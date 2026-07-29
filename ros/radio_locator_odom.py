#!/usr/bin/env python3
"""Fork of follow_me's radio_locator that filters the person estimate in a NON-ROTATING frame.

WHY THIS EXISTS
    The upstream radio_locator (follow_me package) trilaterates the person from robot-mounted UWB
    anchors and runs its Kalman filter in the `locator` frame -- which is rigidly bolted to
    `base_link` (identity static TF) and therefore ROTATES WITH THE ROBOT. When the robot turns in
    place, the person's body-frame coordinates rotate with it, the constant-position filter reads
    that as a huge apparent motion, smooths it, and the estimate lags -- "the follow point swings
    with the rotation and slowly integrates back." Pure rotating-frame artifact.

    This fork changes ONE thing: after the (unchanged) body-frame trilateration, it transforms the
    point into a fixed, non-rotating frame (`filter_frame`, default `odom`, published by
    elevation_node) and runs the Kalman filter THERE. A pure spin now moves the person 0 in odom ->
    zero innovation -> no lag. Real walking is still real motion and gets smoothed as before.

    Everything else -- trilateration, AoA fusion, range Butterworth LPF, weighting -- is copied
    verbatim from upstream so this stays easy to diff and drop once the fix lands upstream. All
    behavioral changes are tagged `# CHANGED:`.

DEPLOYMENT
    Runs ON THE ROBOT (helhest-jr-robot), where the raw UWB topics, their message types, scipy, and
    the base_link->locator TF live -- NOT on the Jetson. It is a plain script (no colcon build, no
    warp, no .venv): scp it over, source the workspace, and run

        python3 radio_locator_odom.py            # zero-config drop-in on /radio/estimate_pose
        python3 radio_locator_odom.py --ros-args -p filter_frame:=map   # A/B against map

    Run it INSTEAD of follow_me's radio_locator (same node/topic). It self-loads follow_me's own
    twr.yaml + radio.yaml from the installed share dir, so no separate params file is needed.

    Dependency: `odom` is published by elevation_node (Jetson) over shared /tf. If that TF isn't
    reaching the robot the node logs `no TF odom<-locator` and skips -- same failure mode as
    elevation_node's own follow callback.
"""

from __future__ import annotations

import os
from collections import deque
from functools import partial

import numpy as np
import rclpy
import rclpy.time
import ros2_numpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.optimize import least_squares
from scipy.signal import butter
from scipy.signal import lfilter
from scipy.signal import lfilter_zi

from tf2_ros import Buffer
from tf2_ros import ConnectivityException
from tf2_ros import ExtrapolationException
from tf2_ros import LookupException
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformListener

from dwm1001_ros_interfaces.msg import UWBMeas
from follow_me_interfaces.msg import HeadingEstimate
from follow_me_interfaces.msg import PositionEstimate
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
from std_msgs.msg import String

CUTOFF = 3.0
fs = 10

np.set_printoptions(precision=3)


# ======================================================================================
# Copied verbatim from follow_me/utils.py -- inlined so the fork is self-contained and does
# NOT import follow_me.utils (that would couple it to their internals and their install path).
# ======================================================================================
class Kalman:
    """Constant-position (random-walk) Kalman filter for a 3D position, with
    innovation-gated adaptive process noise. (Copied from follow_me.utils.)"""

    def __init__(self, x0, P0, q, r, gate_threshold=9.0, max_inflation=200.0):
        self.x = x0
        self.x_cov = P0
        self.H = np.eye(3)
        self.Q = q * np.eye(3)
        self.R = r * np.eye(3)
        self.gate_threshold = gate_threshold
        self.max_inflation = max_inflation

    def set_initial(self, x0, P0):
        self.x = x0
        self.x_cov = P0

    def predict(self, dt):
        # A = I (no motion model); process noise still accrues with elapsed time
        self.x_cov = self.x_cov + self.Q * dt
        return self.x, self.x_cov

    def correct(self, measurement):
        K = np.matmul(
            np.matmul(self.x_cov, self.H.T),
            np.linalg.inv(np.matmul(np.matmul(self.H, self.x_cov), self.H.T) + self.R),
        )
        self.x = self.x + np.matmul(K, measurement - np.matmul(self.H, self.x))
        self.x_cov = np.matmul(
            np.eye(self.x_cov.shape[0]) - np.matmul(K, self.H), self.x_cov
        )
        return self.x, self.x_cov

    def step(self, measurement, dt):
        self.predict(dt)
        innovation = measurement - np.matmul(self.H, self.x)
        S = np.matmul(np.matmul(self.H, self.x_cov), self.H.T) + self.R
        d2 = float(np.matmul(np.matmul(innovation.T, np.linalg.inv(S)), innovation))
        if d2 > self.gate_threshold:
            factor = min(d2 / self.gate_threshold, self.max_inflation)
            self.x_cov = self.x_cov * factor
        return self.correct(measurement)


def declare_kalman_parameters(node, prefix="kalman", q=1.0, r=1.0, gate_threshold=9.0, max_inflation=200.0):
    """Declare a Kalman filter's tuning knobs as ROS params and return them as a dict.
    (Copied from follow_me.utils.)"""
    node.declare_parameter(f"{prefix}_q", q)
    node.declare_parameter(f"{prefix}_r", r)
    node.declare_parameter(f"{prefix}_gate_threshold", gate_threshold)
    node.declare_parameter(f"{prefix}_max_inflation", max_inflation)
    return {
        "q": node.get_parameter(f"{prefix}_q").value,
        "r": node.get_parameter(f"{prefix}_r").value,
        "gate_threshold": node.get_parameter(f"{prefix}_gate_threshold").value,
        "max_inflation": node.get_parameter(f"{prefix}_max_inflation").value,
    }


def attach_kalman_param_callback(node, kalman, prefix="kalman"):
    """Retune a Kalman filter in place on `ros2 param set`. (Copied from follow_me.utils.)"""

    def cb(params):
        for p in params:
            if p.name == f"{prefix}_q":
                kalman.Q = p.value * np.eye(3)
            elif p.name == f"{prefix}_r":
                kalman.R = p.value * np.eye(3)
            elif p.name == f"{prefix}_gate_threshold":
                kalman.gate_threshold = p.value
            elif p.name == f"{prefix}_max_inflation":
                kalman.max_inflation = p.value
        return SetParametersResult(successful=True)

    node.add_on_set_parameters_callback(cb)


def get_transform(node, tf_buffer, tf_from, tf_to, out="matrix", time=None, dur=0.1):
    """Latest transform between two frames. Multiplying a point in `tf_to` by the returned matrix
    gives it in `tf_from`. Returns None on a tf2 lookup failure. (Copied from follow_me.utils.)"""
    if time is None:
        tf_time = rclpy.time.Time()
    else:
        if not isinstance(time, rclpy.time.Time):
            raise TypeError("parameter time has to be rclpy Time")
        tf_time = time

    try:
        t = tf_buffer.lookup_transform(tf_from, tf_to, tf_time, Duration(seconds=dur))
    except (LookupException, ExtrapolationException):
        return None
    except ConnectivityException as ex:
        node.get_logger().error(str(ex))
        return None

    if out == "matrix":
        return ros2_numpy.numpify(t.transform)
    elif out == "tf":
        return t
    else:
        raise ValueError("argument out should be 'matrix' or 'tf'")


def _load_follow_me_config():
    """Load follow_me's own twr.yaml + radio.yaml from its installed share dir, mirroring the
    twr_* param injection that radio.launch.py does -- so this fork needs no separate params file.
    Only works on the robot, where follow_me is installed."""
    try:
        share = get_package_share_directory("follow_me")
    except Exception as exc:  # PackageNotFoundError and friends
        raise RuntimeError(
            "follow_me not found -- run this fork on the robot where follow_me is installed"
        ) from exc
    with open(os.path.join(share, "config", "twr.yaml")) as f:
        twr = yaml.safe_load(f)["/**"]["ros__parameters"]
    with open(os.path.join(share, "config", "radio.yaml")) as f:
        radio = yaml.safe_load(f)["/**"]["ros__parameters"]
    return twr, radio


class Locator(Node):
    def __init__(self):
        # CHANGED: hardcode the "radio" namespace + node name so this is a zero-arg drop-in on
        # /radio/estimate_pose (upstream got these from radio.launch.py's PushRosNamespace).
        super().__init__("radio_locator", namespace="radio")

        # CHANGED: self-load follow_me's twr.yaml + radio.yaml and use the values as param defaults
        # (CLI -p overrides still win), instead of receiving them from a launch file.
        twr_cfg, radio_cfg = _load_follow_me_config()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.declare_parameter("fixed_frame", radio_cfg["fixed_frame"])
        self.declare_parameter("human_frame", radio_cfg["human_frame"])
        # CHANGED: the NON-ROTATING frame the estimate is filtered and published in -- the whole
        # point of this fork. `fixed_frame` stays the body frame the anchors are defined in.
        # NB: on this robot elevation_node names the odom frame "odom_2d" (it inherits /odom_2d's
        # header.frame_id), NOT "odom" -- so map->odom_2d->base_link is the live chain.
        self.declare_parameter("filter_frame", "odom_2d")
        self.declare_parameter("use_3d", radio_cfg.get("use_3d", False))
        self.declare_parameter("use_aoa", radio_cfg.get("use_aoa", False))
        self.declare_parameter("mount_height", radio_cfg.get("mount_height", 0.0))
        self.declare_parameter("max_msg_delay", radio_cfg.get("max_msg_delay", 0.3))
        # values mirrored from the /uwb/twr node parameters (from twr.yaml via _load_follow_me_config)
        self.declare_parameter("twr_ids", twr_cfg["ids"])
        self.declare_parameter("twr_positions", twr_cfg["positions"])
        self.declare_parameter("twr_calibration", twr_cfg["calibration"])
        self.declare_parameter("twr_target", twr_cfg["target"])

        self.fixed_frame = self.get_parameter("fixed_frame").value
        self.human_frame = self.get_parameter("human_frame").value
        self.filter_frame = self.get_parameter("filter_frame").value  # CHANGED
        self.use_3d = self.get_parameter("use_3d").value
        self.use_aoa = self.get_parameter("use_aoa").value
        self.mount_height = self.get_parameter("mount_height").value
        self.max_msg_delay = self.get_parameter("max_msg_delay").value
        if not self.use_3d:
            self.get_logger().warning(
                "radio localisation is set to operate just in the x-y plane"
            )
        self.br = TransformBroadcaster(self)
        self.t = TransformStamped()
        self.t.header.frame_id = self.filter_frame  # CHANGED: broadcast human_frame under odom
        self.t.child_frame_id = self.human_frame
        self.t.transform.rotation.w = 1.0

        self.p = PoseWithCovarianceStamped()
        self.p.header.frame_id = self.filter_frame  # CHANGED: cov pose is in odom
        self.p.pose.pose.orientation.w = 1.0

        # TWR
        self.ids = self.get_parameter("twr_ids").value
        self.coords = self.get_parameter("twr_positions").value
        self.coeffs = self.get_parameter("twr_calibration").value
        self.target = self.get_parameter("twr_target").value
        self.positions = {}
        self.calibration = {}
        self.subscribers = {}
        self.queues = {}
        self.ranges = {}
        self.ranges_avg = {}
        self.uwb_stamps = {}
        self.lp_filter = butter(3, CUTOFF, fs=fs)
        self.zi = {}
        for i in range(len(self.ids)):
            id = self.ids[i]
            self.positions[id] = np.array(
                [
                    [self.coords[3 * i]],
                    [self.coords[3 * i + 1]],
                    [self.coords[3 * i + 2]],
                ]
            )
            self.calibration[id] = [
                self.coeffs[4 * i],
                self.coeffs[4 * i + 1],
                self.coeffs[4 * i + 2],
                self.coeffs[4 * i + 3],
            ]
            self.queues[id] = deque()
            self.ranges[id] = np.nan
            self.ranges_avg[id] = 0.0
            self.uwb_stamps[id] = None
            topic = "/uwb/twr/ID_" + id + "/distances"
            self.subscribers[id] = self.create_subscription(
                UWBMeas, topic, partial(self.range_cb, id=id), 1
            )
            self.zi[id] = lfilter_zi(self.lp_filter[0], self.lp_filter[1])

        # AoA
        self.heading_estimate = None
        self.heading_stamp = None
        if self.use_aoa:
            self.heading_subs = self.create_subscription(
                HeadingEstimate, "/bluetooth/aoa/angle", self.angle_cb, 1
            )

        if self.use_3d:
            self.last_pos = np.array([np.nan, np.nan, np.nan])
        else:
            self.last_pos = np.array([np.nan, np.nan])

        kalman_params = declare_kalman_parameters(
            self,
            q=radio_cfg.get("kalman_q", 1.0),
            r=radio_cfg.get("kalman_r", 0.25),
            gate_threshold=radio_cfg.get("kalman_gate_threshold", 9.0),
            max_inflation=radio_cfg.get("kalman_max_inflation", 200.0),
        )
        self.filter = Kalman(np.array([[0.0], [0.0], [0.0]]), np.eye(3), **kalman_params)
        attach_kalman_param_callback(self, self.filter)
        self.filter_last_time = None
        self.initialised = False

        self.started = False
        self.pub = self.create_publisher(Bool, "/detection_ready", 1)
        self.estimate_pub = self.create_publisher(PositionEstimate, "estimate", 1)
        self.pose_pub = self.create_publisher(PoseStamped, "estimate_pose", 1)
        self.pose_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped, "estimate_pose_cov", 1
        )
        self.sound_pub = self.create_publisher(String, "/log_sound", 1)

        self.get_logger().info(
            f"radio_locator_odom: filtering in {self.filter_frame!r} (anchors in {self.fixed_frame!r})"
        )
        self.get_logger().info("Waiting for average value of the measurements")
        # warm up the filters before starting localisation (callbacks run while spinning)
        self.tim = None
        self._start_timer = self.create_timer(5.0, self._start)

    def _start(self):
        self._start_timer.cancel()
        self.get_logger().info("Starting radio localisation")
        self.tim = self.create_timer(0.1, self.publish_pose)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def range_cb(self, msg, id):
        for m in msg.measurements:
            if m.id != self.target:
                continue
            d = m.dist
            self.ranges[id] = d
            p = self.calibration[id]
            d_cal = p[0] * d**3 + p[1] * d**2 + p[2] * d + p[3]
            d_filt, zi = lfilter(
                self.lp_filter[0], self.lp_filter[1], np.array([d_cal]), zi=self.zi[id]
            )
            self.zi[id] = zi
            self.ranges[id] = d
            self.ranges_avg[id] = float(d_filt)
            self.uwb_stamps[id] = self.now_sec()

    def angle_cb(self, msg):
        self.heading_estimate = msg
        self.heading_stamp = self.now_sec()

    def intersectionPoint(self, guess, init, p_init):
        if self.use_aoa and self.heading_estimate is None:
            self.get_logger().warning("No estimate available")
            return None
        valid_ids = []
        x_t = []
        y_t = []
        z_t = []
        d = []
        age = []

        t = self.now_sec()
        for i in range(len(self.ids)):
            id = self.ids[i]
            meas_age = t - self.uwb_stamps[id]
            if meas_age > self.max_msg_delay:
                self.get_logger().error(
                    "Measurement from TWR tag %s is more than %.2f seconds old, disregarding it"
                    % (id, self.max_msg_delay)
                )
                id_mod = ""
                for j in range(len(id) - 1):
                    id_mod += id[j] + " "
                id_mod += id[-1]
                s = String(
                    data="Warning: Measurement from T W R tag %s is more than %.2f seconds old"
                    % (id_mod, self.max_msg_delay)
                )
                self.sound_pub.publish(s)
                continue
            valid_ids += [id]
            x_t += [[self.positions[id][0][0]]]
            y_t += [[self.positions[id][1][0]]]
            z_t += [[self.positions[id][2][0]]]
            d += [[self.ranges_avg[id]]]
            age += [meas_age]

        if len(valid_ids) < 3:
            self.get_logger().error(
                "Not enough up-to-date TWR measurements (%d) to solve for the position, "
                "at least 3 are required" % len(valid_ids)
            )
            return None

        use_aoa_now = False
        tf = None
        angles = None
        if self.use_aoa:
            heading_age = t - self.heading_stamp
            if heading_age > self.max_msg_delay:
                self.get_logger().warning(
                    "AoA heading estimate is more than %.2f seconds old, disregarding it"
                    % self.max_msg_delay
                )
            else:
                angles = [self.heading_estimate.azimuth, self.heading_estimate.elevation]
                angles_frame = self.heading_estimate.header.frame_id
                tf = get_transform(self, self.tf_buffer, angles_frame, self.fixed_frame)
                if tf is None:
                    self.get_logger().fatal(
                        "No transform between %s and %s" % (self.fixed_frame, angles_frame)
                    )
                    return None
                use_aoa_now = True

        x_t = np.array(x_t)
        y_t = np.array(y_t)
        z_t = np.array(z_t)
        d = np.array(d)

        w = self.weighting_function(d, valid_ids)

        # weight based on how old is the measurement
        w_t = np.zeros((len(valid_ids), 1))
        for i in range(len(valid_ids)):
            if age[i] < 1.0:
                w_t[i, 0] = min(age) / age[i]
        w_t = np.minimum(w_t, 0.5)  # let the maximum difference in magnitude be 2 -> otherwise it could happen, that one measurement overtakes all just because of unlucky timing
        # # w_t = 0.01*np.ones((len(valid_ids), 1))

        print(f"age weights {w_t}")

        def eq(g):
            # TWR
            if self.use_3d:
                x, y, z = g
                f = (x - x_t) ** 2 + (y - y_t) ** 2 + (z - z_t) ** 2 - d**2
            else:
                x, y = g
                f = (
                    (x - x_t) ** 2
                    + (y - y_t) ** 2
                    + (self.mount_height - z_t) ** 2
                    - d**2
                )

            f = w_t * f  # weighting based on the age of the measurement

            if use_aoa_now:
                # AOA
                # perpendicular distance (in 2D) between the candidate point and the
                # ray originating at the AoA sensor with direction given by the
                # measured azimuth angle
                n = np.array([[np.cos(angles[0])], [np.sin(angles[0])]])
                p = np.matmul(tf, np.array([[x], [y], [0], [1]]))[:2, :]
                dist = float(n[0, 0] * p[1, 0] - n[1, 0] * p[0, 0])
                self.get_logger().info(f"aoa {angles[0]}\n aoa pose {n}\n pose {p}\n DIST {dist}")
                f = np.vstack((f, 30.0 * dist))

            return f.flatten().tolist()

        if init:
            best = None
            cost = np.inf
            for p in p_init:
                ans = least_squares(eq, p, loss="soft_l1", verbose=0)
                if ans.success and ans.cost < cost:
                    best = ans.x
                    cost = ans.cost
            return best
        else:
            ans = least_squares(eq, guess, loss="soft_l1", verbose=0)
            self.get_logger().info(f"success: {ans.success}, {ans.status}, {ans.message}")
            self.get_logger().info(f"final residual (incl. AoA): {ans.fun} result pose {ans.x}")

            if ans.success:
                return ans.x
            else:
                return None

    def publish_pose(self):
        for id in self.ids:
            if np.isnan(self.ranges[id]):
                self.get_logger().warning("missing data")
                return

        # find the intersection point (BODY frame -- the anchors are defined in fixed_frame/locator)
        init = False
        p_init = []
        if np.any(np.isnan(self.last_pos)):
            p = self.positions[self.ids[0]]
            if self.use_3d:
                p_init += [
                    np.array([p[0][0] + self.ranges_avg[self.ids[0]], p[1][0], p[2][0]])
                ]
                p_init += [
                    np.array([p[0][0], p[1][0] + self.ranges_avg[self.ids[0]], p[2][0]])
                ]
                p_init += [
                    np.array([p[0][0] - self.ranges_avg[self.ids[0]], p[1][0], p[2][0]])
                ]
                p_init += [
                    np.array([p[0][0], p[1][0] - self.ranges_avg[self.ids[0]], p[2][0]])
                ]
            else:
                p_init += [np.array([p[0][0] + self.ranges_avg[self.ids[0]], p[1][0]])]
                p_init += [np.array([p[0][0], p[1][0] + self.ranges_avg[self.ids[0]]])]
                p_init += [np.array([p[0][0] - self.ranges_avg[self.ids[0]], p[1][0]])]
                p_init += [np.array([p[0][0], p[1][0] - self.ranges_avg[self.ids[0]]])]
            init = True
        x = self.intersectionPoint(self.last_pos, init, p_init)
        if x is None:
            self.get_logger().warning("intersection point not found")
            return
        if not self.use_3d:
            x = np.concatenate((x, np.array([self.mount_height])))

        # CHANGED: lift the body-frame solve into the non-rotating filter_frame BEFORE filtering.
        # This is the whole fix: a pure robot rotation leaves the person put in odom -> no lag.
        # get_transform(from=filter_frame, to=fixed_frame) returns filter_T_locator (maps a locator
        # point into odom). Skip this cycle if the TF isn't available yet (elevation stack down).
        filt_T_loc = get_transform(self, self.tf_buffer, self.filter_frame, self.fixed_frame)
        if filt_T_loc is None:
            self.get_logger().warning(
                f"no TF {self.filter_frame}<-{self.fixed_frame}, skipping this estimate",
                throttle_duration_sec=2.0,
            )
            return
        x_f = np.matmul(filt_T_loc, np.array([[x[0]], [x[1]], [x[2]], [1.0]]))[:3, 0]

        now = self.now_sec()
        if not self.initialised:
            self.filter.set_initial(x_f.reshape(3, 1), np.eye(3))  # CHANGED: seed filter in odom
            self.initialised = True
            self.filter_last_time = now
        else:
            dt = now - self.filter_last_time
            self.filter_last_time = now
            # CHANGED: filter in filter_frame (odom) -- constant-position model is now physically
            # correct (a person's world position really is ~constant over a 0.1 s step).
            x_new, cov = self.filter.step(x_f.reshape(3, 1), dt)
            cov_pose = np.zeros((6, 6))
            cov_pose[:3, :3] = cov[:3, :3]
            x_f = x_new[0:3, :].flatten()

            self.p.header.stamp = self.get_clock().now().to_msg()
            self.p.pose.pose.position.x = float(x_f[0])
            self.p.pose.pose.position.y = float(x_f[1])
            self.p.pose.pose.position.z = float(x_f[2])

            self.p.pose.covariance = cov_pose.flatten().tolist()
            self.pose_cov_pub.publish(self.p)

        # CHANGED: the next least_squares seed / range-weighting stays in the BODY frame, where the
        # anchors live -- so keep last_pos as the RAW body solve (not the odom-filtered point).
        if self.use_3d:
            self.last_pos = x
        else:
            self.last_pos = x[:2]

        # send estimate (CHANGED: now in filter_frame, coordinates x_f)
        est = PositionEstimate()
        est.header.frame_id = self.filter_frame
        est.header.stamp = self.get_clock().now().to_msg()
        est.position_estimate = Point(x=float(x_f[0]), y=float(x_f[1]), z=float(x_f[2]))
        self.estimate_pub.publish(est)

        pose = PoseStamped()
        pose.header.frame_id = self.filter_frame  # CHANGED: was fixed_frame ("locator")
        pose.header.stamp = est.header.stamp
        pose.pose.position = Point(x=float(x_f[0]), y=float(x_f[1]), z=float(x_f[2]))
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        # send tf (CHANGED: filter_frame -> human_frame, coordinates x_f)
        self.t.header.stamp = self.get_clock().now().to_msg()
        self.t.transform.translation.x = float(x_f[0])
        self.t.transform.translation.y = float(x_f[1])
        self.t.transform.translation.z = float(x_f[2])
        self.br.sendTransform(self.t)

        # send ready signal
        if not self.started:
            self.pub.publish(Bool(data=True))
            self.started = True

    def weighting_function(self, d, ids):
        weights = []
        for i in range(len(ids)):
            id = ids[i]
            d_meas = d[i]
            pos = self.positions[id]
            if not self.use_3d:
                pos = pos[:2, :]
            d_pred = np.linalg.norm(pos - self.last_pos[:, None])
            w = 1 / (100 * (d_meas - d_pred) ** 2 + 1e-4)
            weights += [w]
        weights = np.array(weights)  # (n_uwbs, 1)
        return weights


def main(args=None):
    rclpy.init(args=args)
    node = Locator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
