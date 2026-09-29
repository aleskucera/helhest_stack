#!/usr/bin/env python3
"""Odin's on-robot navigation node for ROS 2: map, plan, drive.

The pose is Odin's on-device SLAM (`odom_topic`), trusted as is. Each dToF scan is filtered and
folded into one probabilistic elevation belief (elevation_belief, through
helhest.perception.belief_frame -- the same path the closed-loop sim plans on), and every planning
map is a crop of it. Published, elevation-only (no traversability):

  * `elevation_local`  — the MPPI window (`win_m`, robot-centred crop of the belief), NaN where the
    belief never measured.
  * `elevation_global` — the routing window (`route_m`), likewise.

Frames: sensor -> base (base_frame, static TF) -> odom == map_frame; the node publishes the
identity map -> odom and odom -> base itself.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
import tf2_ros
import warp as wp
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
from helhest.perception import ScanPreprocessor
from helhest.perception.belief_frame import BeliefFrame
from helhest.perception import OutlierFilterConfig
from helhest.perception import StatisticalOutlierFilter
from helhest.perception import transform_points
from helhest.planning.coarse import CoarseRouter
from helhest import dynamics
from helhest.control.command import condition_command
from helhest.control.command import turn_first
from helhest.control.governor import ClearanceGovernor
from helhest.control.command import in_flight_history
from helhest.control.command import JOINT_NAMES
from helhest.control.command import joint_states_to_model
from helhest.control.command import to_engine_order
from helhest.control.mppi import MppiGpu
from helhest.control.terminal import dock_control
from helhest.control.turn_adapt import AdaptiveTurnBoost
from helhest.control.turn_adapt import TurnGainEstimator
from helhest.control.yaw_track import YawRateTracker
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo
from helhest.localization.pose_math import invert_pose
from helhest.localization.pose_math import matrix_to_quaternion
from helhest.planner_config import PLAN_DEFAULTS
from helhest.planner_config import planner_config
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException

from ._pipeline_common import elevation_to_cloud
from ._pipeline_common import pointcloud2_to_xyz_time_array
from ._pipeline_common import quaternion_to_matrix

_EZ = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # world up

_IMU_BUFFER_LEN = 500  # ~5 s of 100 Hz IMU — enough to bracket any cloud stamp
_IMU_MAX_EXTRAP_S = 0.05  # fall back to odom if no IMU sample within this of the cloud stamp


# Construction-time params: a change to any rebuilds the owning object.
_OUTLIER_BUILD = frozenset({"outlier_search_radius_m", "outlier_min_neighbors", "device"})
# Planner is sized to the windows + rollout shape; a change to any rebuilds it (and its
# CUDA graphs), so keep it off the per-frame path.
_PLAN_BUILD = frozenset(
    {
        "win_m",
        "route_m",
        "resolution",
        "plan_batch",
        "plan_horizon",
        "plan_n_theta",
        "plan_lat_coarsen",
        "plan_friction",
        "terrain",
        "k_turn",
        "plan_robust_margin_m",
        "plan_robust_margin_deg",
        "plan_nominal_reset",
        "plan_tau_motor",
        "plan_goal_running",
        "plan_effort",
        "plan_turn",
        "plan_wmax",
        "plan_wmin",
        "plan_straight_frac",
        "plan_spin_frac",
        "plan_spin_min",
        "plan_spin_max",
        "plan_elite_frac",
        "plan_n_mu",
        "plan_mu_adapt",
        "plan_mu_span",
        "plan_mu_tau",
        "plan_saturation",
        "plan_wall_veto",
        "plan_pivot_cost",
        "plan_z_veto",
        "plan_charge_per_sigma",
        "plan_clear_t_react",
        "plan_clear_v_cruise",
        "plan_clear_v_min",
        "plan_clear_lookahead_s",
        "plan_clear_mppi_weight",
        "plan_clear_decel",
        "plan_clear_c0",
        "plan_clear_t_turn",
        "plan_clear_route_turn",
        "plan_clear_v_blind",
        "plan_coarse_block_m",
        "plan_coarse_memory_m",
        "plan_coarse_win_m",
        "plan_bridge_m",
        "plan_turn_boost_adapt",
        "plan_turn_boost_tau",
        "plan_yaw_track",
        "device",
    }
)


@dataclass(frozen=True)
class _MapFrame:
    """One frame's built maps + window geometry, shared by publishing and planning."""

    elev_local: np.ndarray  # (wh, ww) filled, NaN-free — planner terrain
    elev_local_view: np.ndarray  # (wh, ww) NaN in unknown cells — for RViz
    relev_view: np.ndarray  # (rwh, rww) NaN in unknown cells — for RViz
    relev_mem: np.ndarray  # (rwh, rww) blind cells inpainted — cost-to-go routing terrain
    relev_measured: np.ndarray  # (rwh, rww) bool: True where the belief has a measurement
    cell: float
    ex: float
    ey: float
    lxmin: float
    lymin: float
    rxmin: float
    rymin: float
    # the device-resident planner inputs: the MPPI crop and its mask, the routing crop's offset
    dev: dict


def _same_manoeuvre(a: np.ndarray, b: np.ndarray, spin_th: float = 0.25) -> bool:
    """Do two plans belong to the same manoeuvre class -- same turn side, and both spinning or
    both not? Mirrors the keys MPPI's elite uses (mppi._cand_dir_kernel): a spin has wl = -wr so
    its MEAN wheel speed is ~0, which is what separates it from a forward arc of the same
    differential."""
    for u in (a, b):
        if u.ndim != 2 or u.shape[1] < 2:
            return True  # unknown shape: fall back to smoothing, as before
    sa, sb = float(np.mean(a[:, 0] + a[:, 1])) * 0.5, float(np.mean(b[:, 0] + b[:, 1])) * 0.5
    da, db = float(np.mean(a[:, 1] - a[:, 0])), float(np.mean(b[:, 1] - b[:, 0]))
    if (abs(sa) <= spin_th) != (abs(sb) <= spin_th):
        return False  # one is a spin and the other is not
    return not (da * db < 0.0 and min(abs(da), abs(db)) > 0.3)  # opposite turn sides


class NavigationNode(Node):
    """Map with the elevation belief, plan with the cost-to-go and MPPI, drive the robot."""

    def __init__(self) -> None:
        super().__init__("navigation")

        self._declare_parameters()
        self._cache_params()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._bframe = None  # BeliefFrame, created on the first scan
        self._bframe_t: float | None = None  # stamp of the last scan folded into it [s]
        self._bbuf: dict[str, wp.array] = {}  # the belief path's preallocated crops and pools
        self._frame: int = 0  # monotonic processed-frame counter (RViz label, debugging)
        self._rec: dict[str, list] | None = (
            {k: [] for k in ("h", "seen", "blk", "v", "route", "cv", "meta", "goal", "t", "trail")}
            if self.plan_debug_record
            else None
        )
        self._rec_planned: int = 0  # planned frames so far, for the every-Nth rule
        # Latest map->odom correction, cached from the last processed cloud and re-broadcast
        # at the full odom rate (see _odom_tf_callback) so base_link stays dense for TF lookups.
        self._map_T_odom: np.ndarray | None = None
        self.goal_xy: tuple[float, float] | None = None  # planning goal in map frame
        self._prev_cmd = np.zeros(
            3, np.float32
        )  # last published /cmd_joints [L, rear, R] (slew ref)
        # Commands already in flight, ENGINE order (wL, wR, w_rear), oldest first. Length = the
        # delay in whole rollout steps; empty (and unused) when plan_command_delay is 0.
        self._cmd_in_flight: deque[np.ndarray] = deque(maxlen=1)
        self._last_cmd_time: float | None = None  # clock of the last /cmd_joints publish
        self._prev_plan_U: np.ndarray | None = (
            None  # last frame's nominal plan (for plan-consistency EMA)
        )
        self._turn_adapt: AdaptiveTurnBoost | None = (
            None  # optional online turn_boost (gyro feedback)
        )
        self._last_diff_out: float | None = (
            None  # last commanded (wR-wL), paired with the yaw it caused
        )
        self._rev_open = False  # reverse gate state last frame (for transition logging)
        self._last_turn_adapt_time: float | None = None  # clock of the last turn_boost EMA update
        self._yaw_track: YawRateTracker | None = None  # optional fast inner yaw-rate loop
        self._last_yaw_track_time: float | None = None  # clock of the last yaw-loop update
        self._goal_reached = (
            False  # latched at the goal -> idle (no planning) until the goal changes
        )
        self.planner: MppiGpu | None = None
        self.plan_sim: ForwardSimulator | None = None
        self.ctg: CostToGo | None = None
        self.sgrid = None  # routing lattice grid (built)
        self._plan_kr: int = 1
        self._plan_dims: tuple[int, int, int, int, int, int] | None = None
        # (t_sec, quaternion xyzw, angular_velocity xyz in base) -- the yaw-rate loop, the
        # turn-boost adapter and the mu estimate read the latest gyro sample from here.
        self._imu_buffer: deque[tuple[float, np.ndarray, np.ndarray]] = deque(
            maxlen=_IMU_BUFFER_LEN
        )
        self._base_R_gyro: np.ndarray | None = None  # cached static base<-imu rotation for the gyro
        self._prof: dict[str, float] = {}  # per-stage cumulative seconds (profile_stages)
        self._prof_n = 0
        self._prof_t = 0.0
        self._preproc: ScanPreprocessor | None = None  # device scan entry path

        self.device = self._resolve_device(self.get_parameter("device").value)
        self._build_outlier_filter()
        if self.plan_enable:
            self._build_planner()

        # Sensor QoS (best-effort): a best-effort sub receives from reliable and best-effort
        # publishers alike; a reliable one gets nothing from the latter.
        self.create_subscription(Imu, self.imu_topic, self._imu_callback, qos_profile_sensor_data)
        self._wheel_meas: np.ndarray | None = None  # measured [wL, wR, w_rear], model convention
        self._wheel_meas_t: float = 0.0  # node-clock seconds of the last measurement
        self._twist_meas: np.ndarray | None = None  # measured body twist (vx, vy, yaw_rate)
        self._twist_meas_t: float = 0.0
        js_topic = self.get_parameter("joint_states_topic").value
        if js_topic:
            self.create_subscription(
                JointState, js_topic, self._joint_states_callback, qos_profile_sensor_data
            )
        self.create_subscription(
            PoseStamped, self.get_parameter("goal_topic").value, self._goal_callback, 10
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("follow_topic").value, self._follow_callback, 10
        )
        # LiDAR is best-effort (SensorDataQoS); a reliable sub gets nothing from it.
        self.cloud_sub = Subscriber(
            self, PointCloud2, self.lidar_topic, qos_profile=qos_profile_sensor_data
        )
        self.odom_sub = Subscriber(self, Odometry, self.odom_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.cloud_sub, self.odom_sub],
            queue_size=self.get_parameter("sync_queue").value,
            slop=self.get_parameter("sync_slop_s").value,
        )
        self.sync.registerCallback(self._synced_callback)
        # Broadcast the pose TF on EVERY odom message (not just synced/processed frames), so
        # map->base_link is dense enough for RViz to look it up at any cloud stamp — e.g. with
        # base_link as the fixed frame. Same subscriber, extra callback: no second subscription.
        self.odom_sub.registerCallback(self._odom_tf_callback)

        self.pub_local = self.create_publisher(PointCloud2, "elevation_local", 10)
        self.pub_global = self.create_publisher(PointCloud2, "elevation_global", 10)
        # the coarse layer's cost-to-go [m], as a height grid: unreachable blocks are left out
        self.pub_coarse = self.create_publisher(PointCloud2, "coarse_value", 10)
        self.pub_path = self.create_publisher(Path, "planned_path", 10)
        self.pub_path_marker = self.create_publisher(Marker, "planned_path_marker", 10)
        self.pub_frame = self.create_publisher(Marker, "frame_marker", 1)
        self.pub_cmd = self.create_publisher(JointState, self.get_parameter("cmd_topic").value, 10)
        self.pub_turn_boost = self.create_publisher(
            Float32, "turn_boost", 10
        )  # turn_boost in effect (debug)
        # Inner yaw loop, as (reference, measured, correction). The correction is the ONLY
        # one not reconstructible from a bag: /cmd_joints carries the CORRECTED differential,
        # so without this there is no way to tell what the loop did -- or whether it ran.
        self.pub_yaw_track = self.create_publisher(Vector3, "yaw_track", 10)
        self.add_on_set_parameters_callback(self._on_parameters_changed)  # validate (pre-set)
        self.add_post_set_parameters_callback(self._on_parameters_applied)  # apply (post-set)

        self.get_logger().info(
            f"NavigationNode: cloud={self.lidar_topic} odom={self.odom_topic} imu={self.imu_topic} "
            f"map_frame={self.map_frame} win_m={self.win_m} route_m={self.route_m} "
            f"device={self.device}"
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        d = self.declare_parameter
        # ROS / sensors: Odin, the robot's only sensor stack. The pose is Odin's on-device SLAM
        # (odom_topic), trusted as is; the cloud arrives already in odin1_base_link.
        d("lidar_topic", "/odin1/cloud_raw")
        d("odom_topic", "/odin1/odometry")
        d("imu_topic", "/odin1/imu")
        d("base_frame", "odin1_base_link")  # cloud already in this frame -> sensor TF = identity
        d("map_frame", "map")
        d("sync_slop_s", 0.05)
        d("sync_queue", 30)
        d("device", "auto")
        # Height crop on the input scan, in base_frame (robot-relative). Drops ceiling /
        # sub-floor noise before it reaches the map. Bounds are metres in z.
        d("z_crop_enable", True)
        d("z_crop_min", -1.0)
        d("z_crop_max", 0.5)
        # Horizontal range crop on the input scan: drop returns past this xy-distance from the
        # robot. Far returns are sparse grazing-angle ground -- noise that only pollutes the map.
        # Cropped per SCAN so it never enters any stage. 0 disables.
        d("scan_max_range_m", 15.0)
        # Near cut: drop returns closer than this to the sensor (3-D). The belief adopts a
        # higher reading at once and ignores a lower one, and the dense near field feeds that
        # rule ~14 sweeps a second: without the cut the ground under the robot ratchets up
        # 5-7 cm and leaves a raised trail (Robotour drive, 2026-09-29). 0 disables.
        d("scan_min_range_m", 1.0)
        # Robot self-filter: drop the robot's own returns (wheels/body) -- a base_frame box,
        # measured for the Odin mount (self-returns stay fixed in base while the scene moves).
        d("self_filter_enable", True)
        d("self_x_min", -0.05)
        d("self_x_max", 0.55)
        d("self_y_min", -0.75)
        d("self_y_max", 0.75)
        # Isolated-point removal on the input scan (GPU): drops specks with fewer than
        # min_neighbors returns within the radius before they reach the map. An absolute count
        # (6 is safe out to the routing window).
        d("outlier_enable", True)
        d("outlier_search_radius_m", 0.25)
        d("outlier_min_neighbors", 6)
        # Heightmap (live-tunable). resolution/win_m validated on real bags: the finer 0.08 m cell
        # + 12 m fine window let the MPPI actually see berms across its plan (footprint violations
        # 36%->8% vs the old 0.15/8) -- see the Tier-B planner analysis.
        d("resolution", 0.08)
        d("win_m", 12.0)  # MPPI window (robot-centered)
        d("route_m", 16.0)  # routing window (robot-centered)
        # The planner's three maps (routing, MPPI, coarse) are crops of one elevation_belief
        # window, fed with the scan placed by Odin's SLAM pose -- the path drive_sim plans on
        # (perception/belief_frame.py), which also hands the cost-to-go its sigma and drift.
        d("belief_carve_m", 6.0)  # [m] the belief's visibility carve reach; 0 disables it
        d("debug_frames", False)  # INFO-log per-frame diagnostics (debugging)
        d(
            "profile_stages", False
        )  # GPU-synced per-stage timing, logged every 30 frames (debugging)
        # Reject single-sample gyro glitches before they reach the yaw-rate loop and the turn
        # adapter: the Ouster-era /imu/data spiked to >1000 deg/s for one sample (real motion
        # peaks ~300). Cheap insurance on any IMU. 0 disables.
        d("max_gyro_rate_dps", 600.0)
        # Viz
        d("publish_map_tf", True)
        # Close odom->base_link ourselves. helhest_llc publishes the /odom_2d message but
        # broadcasts no TF, so without this base_link is disconnected from map. Default on
        # here; set false if the odom source ever starts publishing it, or TF double-publishes.
        d("publish_odom_tf", True)
        # MPPI planning. Consumes the maps this node already builds: elevation_local as the
        # rollout terrain, elevation_global for the cost-to-go routing field. Goal comes from RViz "2D Nav Goal" on goal_topic. Publishes
        # the intended path (nav_msgs/Path + a thick LINE_STRIP marker).
        d("plan_enable", True)
        d("goal_topic", "/goal_pose")  # RViz "2D Nav Goal" (a one-shot click, latches on reach)
        # GOAL SOURCE: which stream drives the planner. "click" -> goal_topic (RViz); "follow" ->
        # follow_topic, a live pose (e.g. the radio locator) continuously chased -- every update
        # re-targets and the reach latch never sticks, so a MOVING tag is tracked. The follow pose
        # is transformed into map_frame via TF, so it may arrive in any frame (e.g. 'locator').
        # Runtime-switchable: `ros2 param set /navigation goal_source follow` (no rebuild).
        d("goal_source", "click")
        d("follow_topic", "/radio/estimate_pose")  # PoseStamped to chase in "follow" mode
        # FOLLOW STANDOFF: in "follow" mode, aim this far SHORT of the tag (m), back along the
        # robot->tag line. The tag rides on a person = a LiDAR obstacle, so a goal placed ON them is
        # untraversable -- the router can't seed there so the robot "walls off" and holds. Aiming at
        # free space in front fixes that AND is a sane follow gap. 0 = target the tag exactly (walls
        # off on a person). Runtime-tunable via `ros2 param set`.
        d("follow_standoff", 1.5)
        d("plan_batch", PLAN_DEFAULTS["plan_batch"])  # MPPI rollouts B
        # rollout steps T (planning_solver dt = 0.1 s). SHORT is better here: the cost-to-go lattice
        # does the global routing, so a short MPPI just follows it decisively -- sim-validated to reach
        # more (6/6 vs 2/6 stress worlds), cruise ~30% faster, brake cleanly, fit the 12 m local map,
        # and cost ~3.5x less than T=70. A long horizon instead defers braking into its (never-executed)
        # tail and orbits the goal. Raise toward 40-70 only if far-field lookahead is genuinely needed.
        d("plan_horizon", PLAN_DEFAULTS["plan_horizon"])
        # PLAN CONSISTENCY: EMA the nominal plan toward last frame's (shifted one step) so the committed
        # maneuver doesn't jitter on open ground. 0 = off; ~0.3 cut cruise churn ~35% in sim. Too high
        # adds reaction lag to new obstacles/goals.
        d("plan_consistency", 0.3)
        # Cost-to-go heading bins, and the routing grid they live on. These two are chosen
        # TOGETHER, because the lattice arcs turn by {0, +-bins/2, +-bins} bins and reach every
        # heading only when gcd(bins/2, n_theta) == 1, while the step also has to clear one
        # routing cell (= resolution * plan_lat_coarsen). Get the pair wrong and the planner
        # either refuses to build or, before 4dae1be, silently served a heading ring in pieces.
        # See docs/incidents/incident_2026-09-22_lattice-heading-connectivity.md.
        #
        # 24 bins on a 0.24 m cell (coarsen 3) is connected at bins=2: closing_step = 0.2618 m
        # clears the cell and gcd(1, 24) = 1. That is what ros/config/odin.params.yaml
        # has always run, so the DEPLOYED robot was never affected by the split ring -- the
        # defaults here were, at coarsen 4 (0.32 m), which no closing step under a quarter turn
        # both clears and keeps connected. The defaults now match the params file rather than
        # disagreeing with it silently.
        d("plan_n_theta", PLAN_DEFAULTS["plan_n_theta"])
        d("plan_lat_coarsen", 3)  # routing/cost-to-go grid coarsening vs the map cell
        d("plan_n_refine", 3)  # MPPI refine iterations per frame
        d("plan_friction", 0.8)  # uniform rollout friction
        # Wheel envelope half-width [m]. DEFAULT 0.10 = the measured tread, as a yaw-binned
        # CYLINDER: honest laterally, where the sphere reached 0.35 m sideways and refused gaps the
        # robot can straddle. 0.0 = back to the sphere. Costs ~3.9 ms per perception frame to
        # dilate (64 yaw slices) against 0.06 for the sphere. Widen it (0.15, 0.20) to buy lateral
        # margin back without the sphere's fixed 0.35.
        d("plan_wheel_width", PLAN_DEFAULTS["plan_wheel_width"])
        # 'indoor' (K_TURN 0.4, alpha~1.33) or 'outdoor' (K_TURN 1.0, alpha~1.82 -- grass/dirt grips
        # harder so it understeers). ICP-calibrated per environment; see dynamics.k_turn_for.
        d("terrain", "outdoor")
        d(
            "k_turn", -1.0
        )  # explicit turn-gain override (e.g. from scripts/fit_turn_gain.py); <0 = use terrain
        d(
            "plan_robust_margin_m", PLAN_DEFAULTS["plan_robust_margin_m"]
        )  # cost-to-go safety tube: lateral (m) ~ robot half-width;
        # keeps the routed center a footprint-width off berms (validated in the Tier-C closed loop:
        # 0 belly contacts). Tighten in narrow spaces -- it erodes the feasible set both sides.
        d(
            "plan_robust_margin_deg", PLAN_DEFAULTS["plan_robust_margin_deg"]
        )  # cost-to-go safety tube: heading (deg)
        # HARD-block cells within a footprint of a local step taller than this [m]. Catches thin
        # vertical obstacles (sticks/poles) that the robot's settle STRADDLES between its wheel/belly
        # contacts -- those read traversable for most headings, so the router drives through them.
        # 0 = off (settle-only). ~0.2 blocks a stick the robot can't drive over. See costtogo _step_gate.
        # OPT-IN (default off) -- but the reason it was turned off (fae8640) has since been FIXED
        # and this default has not been revisited. It was: relev_mem filled unmeasured cells with
        # 0.0 while the ground sat lower, so every blind/measured frontier read as a step and
        # hard-blocked (a closed ring around the robot). Two days later 7020fd6 replaced that
        # constant fill with an inpaint from measured neighbours, so blind cells now inherit local
        # ground height and the phantom step is gone. Re-enabling (0.2) is probably safe and is the
        # fix for thin obstacles the settle straddles -- VERIFY on a bag first. Note what this
        # relies on: the gate is a LOCAL difference (see costtogo _step_gate), so a map z-origin far
        # from ground -- e.g. Odin, which bootstraps world from its own odom and can start at
        # z = -1.5 m -- does not by itself trip it; only an absolute-height FILL does.
        # Set AT LAUNCH -- a runtime `param set` does NOT take: not in _PLAN_BUILD, and the gate is
        # baked into the CUDA graph.
        d("plan_obstacle_step_m", PLAN_DEFAULTS["plan_obstacle_step_m"])
        d(
            "plan_nominal_reset", PLAN_DEFAULTS["plan_nominal_reset"]
        )  # nominal wheel speed the planner seeds from
        # MODELED ACTUATION LAG [s]: first-order wheel-speed lag inside the MPPI rollouts
        # (SolverParams.tau_motor). Smoothness belongs in the MODEL, not an output filter: an
        # external slew limiter executes something the planner never simulated, and the induced
        # tracking lag (~0.2-0.4 m at 1.4 m/s under heavy filtering) is exactly the graze depth
        # measured against obstacles. With the lag modeled, candidates that need late dodges rank
        # poorly by themselves and the executed motion matches the plan (sim: the pocket trap at
        # 2.1 m/s goes from mass contacts under output filtering to clean in 17 s).
        # -1 (default) = use dynamics.MOTOR_TAU, the bag-MEASURED wheel response (0.19 s) that
        # planning_solver already defaults to -- a 0.0 here would silently OVERRIDE the measured
        # lag back to instantaneous for the node only, while the demos keep it (the exact config
        # drift dynamics.py exists to prevent). 0 = explicitly instantaneous (legacy); > 0 =
        # explicit override. With the lag modeled, relax plan_max_slew toward a safety backstop
        # rather than the smoothness mechanism.
        d("plan_tau_motor", -1.0)
        # MPPI speed knobs (rebuild the planner on change): the robot drives slow because the cost
        # balance prefers it. Raise goal_running (reward progress) and/or lower effort (penalty on
        # wheel-speed^2) to drive faster. plan_max_omega is only the output SAFETY clamp, not speed.
        d(
            "plan_goal_running", PLAN_DEFAULTS["plan_goal_running"]
        )  # cost-to-go V^2 per step -> higher = faster (more progress pull)
        d(
            "plan_effort", PLAN_DEFAULTS["plan_effort"]
        )  # penalize wheel-speed^2 -> lower = faster (less speed penalty)
        # TURN penalty: cost on the wheel differential (wr - wl)^2 -> a real gradient toward STRAIGHT
        # where the goal cost is flat w.r.t. heading (free-heading goal). Cut straight-line wander ~70%
        # (0.42 -> 0.12 m) AND, by killing the near-goal wobble, reached 6/6 stress worlds vs 4/6.
        # Small enough that a genuine need to turn still wins; >~0.1 starts refusing hard turns.
        d("plan_turn", PLAN_DEFAULTS["plan_turn"])
        # HARD speed ceiling: the MPPI wheel-speed sampling box [0, plan_wmax] rad/s. The planner
        # NEVER commands above this regardless of the cost. This is the real top-speed knob.
        # ~1.4 m/s at 4.0; ~1.75 m/s at 5.0 (r=0.35). plan_wmax maps to the REAL wheel speed -- the
        # LLC consumes /cmd_joints as wheel rad/s (see _publish_cmd).
        # TURNING HEADROOM: the turn differential is realized ~1:1 while the wheels have headroom,
        # but collapses as they approach the motor ceiling (the outer wheel, mean+diff/2, pegs). So
        # keep plan_wmax a notch BELOW whatever that ceiling is -- the differential then survives a
        # turn. (The turn "defect" was mostly this saturation, not a fixed drivetrain gain.)
        # WHERE THE CEILING IS, is unsettled. The "~5.3 rad/s" measured 2026-07-15 predates the
        # /cmd_joints unit fix (f056dcc, 2026-07-27) that made plan_wmax map to REAL wheel rad/s, so
        # it is not in today's units; the operator puts the motor limit at ~15 rad/s and the Odin
        # params file runs plan_wmax 6.0 on that basis. The Odin bags don't settle it -- they were
        # recorded at this 4.0 default and never commanded above 4.00. 4.0 is kept as the
        # conservative default; raise it per-robot via a params file once the ceiling is measured.
        d(
            "plan_wmax", PLAN_DEFAULTS["plan_wmax"]
        )  # max per-wheel omega the planner may command [rad/s]
        # REVERSE / POINT-TURN: lower edge of the sampling box. 0.0 = forward-only (the shipped
        # behaviour). NEGATIVE (e.g. -2.5) lets MPPI command reverse arcs and point turns -- but
        # only while the map BEHIND the robot is measured (plan_reverse_clear_m below); with blind
        # ground behind, the effective floor snaps back to 0 for that frame. SAFETY: before enabling
        # on the real robot, verify the LLC drives a small NEGATIVE /cmd_joints backward -- only
        # all-positive-forward has been verified live (control/command.py header).
        d("plan_wmin", PLAN_DEFAULTS["plan_wmin"])
        # Reverse gate: this much ground straight behind base_link (m) must be MEASURED (in the
        # belief) for reverse to unlock this frame. Checked over a robot-width strip each frame.
        d("plan_reverse_clear_m", 1.5)
        # POINT-TURN routing: cost-to-go pivot primitive cost [m-equivalent per heading bin]; > 0
        # lets the router plan pivot-then-drive for goals behind/beside the robot (a skid-steer can
        # rotate in place; the forward-arc-only lattice pretended it can't). 0 = off. At n_theta 16,
        # a half-turn costs 8*pivot_cost m-equivalent -- 0.45 makes pivots win whenever they save
        # ~4 m of looping. Pairs naturally with plan_wmin < 0 but is useful alone.
        # It also reconnects a split heading ring -- the +-1 bin move is the only odd-parity
        # primitive -- which is why plan_n_theta above is chosen as if this were 0. Connectivity
        # must not depend on a price someone may reasonably set to zero.
        d("plan_pivot_cost", PLAN_DEFAULTS["plan_pivot_cost"])
        # the cost-to-go's sigma path (helhest.planner_config), on the belief's measurement sd
        d("plan_z_veto", PLAN_DEFAULTS["plan_z_veto"])
        d("plan_charge_per_sigma", PLAN_DEFAULTS["plan_charge_per_sigma"])
        # The clearance law (helhest/planning/clearance.py); see helhest.planner_config.
        d("plan_clear_t_react", PLAN_DEFAULTS["plan_clear_t_react"])
        d("plan_clear_v_cruise", PLAN_DEFAULTS["plan_clear_v_cruise"])
        d("plan_clear_v_min", PLAN_DEFAULTS["plan_clear_v_min"])
        d("plan_clear_lookahead_s", PLAN_DEFAULTS["plan_clear_lookahead_s"])
        d("plan_clear_mppi_weight", PLAN_DEFAULTS["plan_clear_mppi_weight"])
        d("plan_clear_decel", PLAN_DEFAULTS["plan_clear_decel"])
        d("plan_clear_c0", PLAN_DEFAULTS["plan_clear_c0"])
        d("plan_clear_t_turn", PLAN_DEFAULTS["plan_clear_t_turn"])
        d("plan_clear_route_turn", PLAN_DEFAULTS["plan_clear_route_turn"])
        d("plan_clear_v_blind", PLAN_DEFAULTS["plan_clear_v_blind"])
        # ROBUST-MU replicas: each MPPI candidate is rolled out under this many friction hypotheses
        # spanning the current uncertainty band and ranked by its WORST outcome, so the winner is a
        # plan that works whether the ground grips or slips (the over/understeer sim-to-real gap).
        # Must divide plan_batch. Cost: candidates = plan_batch / n_mu (the rollout wall-time is
        # latency-bound, so 3 replicas cost ~nothing at B=4096). 1 = off.
        d("plan_n_mu", PLAN_DEFAULTS["plan_n_mu"])
        # Friction-uncertainty band half-width (as a FRACTION of plan_friction) the replicas cover
        # when the online estimator is off (or hasn't locked yet). Live-tunable.
        d("plan_mu_span", PLAN_DEFAULTS["plan_mu_span"])
        # ONLINE mu estimation: invert the turn model on (commanded differential, measured gyro yaw)
        # each frame and slow-EMA the effective friction the planner should use -- fixes BOTH
        # understeer and oversteer (the turn-boost hotfix could only fix understeer), and its
        # residual noise sizes the robust-mu band. See control/turn_adapt.TurnGainEstimator.
        d("plan_mu_adapt", False)
        d("plan_mu_tau", 5.0)  # estimator EMA time constant [s]
        # FRICTION-SATURATION certificate weight (0 = off): penalize rollouts whose demanded
        # slope-hold + centripetal + side-slope force exceeds the friction budget. This is the
        # "drive slow where grip is short, fast where it isn't" term -- the model alone gets MORE
        # optimistic as mu drops, so without it low-grip terrain reads as easy.
        d("plan_saturation", PLAN_DEFAULTS["plan_saturation"])
        # HARD wall veto on MPPI's rollouts, over the cost-to-go's wall field (walls eroded by the
        # robust tube; tilt is never in it). Without it MPPI relied on the rollout-infeasibility
        # charge, which cannot help once every candidate already touches: the robot pressed
        # into walls toward the goal. 0 = off.
        d("plan_wall_veto", PLAN_DEFAULTS["plan_wall_veto"])
        # COARSE "which way" layer (planning/coarse.py). Each frame a plan_coarse_win_m crop of
        # the belief is pooled into plan_coarse_block_m blocks, kept in a map anchored to
        # the WORLD (plan_coarse_memory_m across, centred on the map origin) so a dead end the
        # robot drove away from is still there when it matters, and solved to the goal; that value
        # prices the routing window's border. Measured on false_door (a room whose only door faces
        # the goal): without it 0/3, the dead end forgotten once it scrolls out; with it 3/3.
        # Block 0 = off; memory 0 = bound to the raster, forgetting what scrolls out (sim-only).
        d("plan_coarse_block_m", PLAN_DEFAULTS["plan_coarse_block_m"])
        d("plan_coarse_memory_m", PLAN_DEFAULTS["plan_coarse_memory_m"])
        d("plan_coarse_win_m", 20.0)  # node-only: the simulator pools its own belief window
        # A wall seen ending in the sensor's shadow is not a door: an unseen run this short
        # between two sealed blocks is the wall continuing. Two 0.6 m blocks cannot hide a 1.8 m
        # doorway, the narrowest this robot fits through. 0 = off.
        d("plan_bridge_m", PLAN_DEFAULTS["plan_bridge_m"])
        # TURN FIRST (control/command.turn_first): brake the forward speed while the route lies
        # more than this far off the heading, so a big turn is a spin, not an arc that advances
        # into a wall -- forward-only, an arc that turned 52 deg advanced 1.4 m into the robot's
        # own turning clearance and MPPI froze there. 0 = off. Live-tunable.
        d("plan_turn_first_deg", PLAN_DEFAULTS["plan_turn_first_deg"])
        d("plan_turn_first_reach_m", PLAN_DEFAULTS["plan_turn_first_reach_m"])
        # STRAIGHT sampling prior: fraction of MPPI candidates drawn as zero-differential (straight
        # ahead) drives. Straight is usually near-optimal, so seeding it lets the elite lock onto a
        # clean straight command instead of averaging noisy micro-turns -> ~25% less lateral wander on
        # a clear shot, no cost when a turn is actually needed. 0 = off.
        # Wheel-speed CHANGE penalty: the anti-jerk knob. plan_turn penalises the SIZE of a turn
        # and so cannot tell a deliberate repositioning from a wobble; this penalises CHANGING
        # your mind, which is what wobble is. Raised from the 2e-3 library default -- measured
        # closed-loop, it cuts turn-direction flips 0.60 -> 0.36 /s and lateral wander 0.09 ->
        # 0.05 m on a straight shot while leaving the 90 deg turn time unchanged.
        d("plan_smooth", PLAN_DEFAULTS["plan_smooth"])
        d("plan_straight_frac", PLAN_DEFAULTS["plan_straight_frac"])
        # SPIN prior: fraction of MPPI candidates drawn as a turn on the spot (wl = -wr). This is
        # the ONLY sampler band that is exempt from the wmin clamp -- see the clamp in
        # mppi._sample_target_wheel_omega_kernel -- so turning in place does NOT require enabling
        # reverse, and the two capabilities are configured independently.
        #
        # It must be ON whenever plan_pivot_cost > 0, or the router plans pivot-then-drive that
        # the controller cannot sample: MEASURED on sim-demo 2026-09-23, a goal 135 deg off the
        # heading produced 0 deg of turn and 0.08 rad/s of wheel speed -- the robot simply sat
        # there -- while 45 and 90 deg worked. A forward arc covers ~90 deg over 4 m of travel;
        # past that the lattice wants a pivot and forward-only sampling has nothing to offer.
        #
        # It is also the ESCAPE HATCH that keeps the action space non-empty. Forward blocked by an
        # obstacle and reverse locked by the gate below leaves a skid-steer with no legal action --
        # observed as a robot wedged 0.22 m from the world edge, pointing at it, for 40 minutes
        # with a 26-pose plan it could not start. A spin does not translate into unseen ground, so
        # unlike reverse it is safe to leave always available.
        d("plan_spin_frac", PLAN_DEFAULTS["plan_spin_frac"])
        # [rad/s] floor on a spin candidate's wheel speed. MEASURED on the robot 2026-08-10: below
        # about 2 the wheels will not break loose on the spot and a smaller command only strains.
        d("plan_spin_min", PLAN_DEFAULTS["plan_spin_min"])
        # [rad/s] ceiling on a spin candidate's wheel speed; 0 = plan_wmax. Before the turn boost.
        d("plan_spin_max", PLAN_DEFAULTS["plan_spin_max"])
        # CEM elite fraction: MPPI commits the MEAN of the top-k lowest-cost candidates. Because the
        # goal heading is free, small turns near the goal barely change cost -> the elite fills with
        # near-equal micro-turn candidates and their mean WOBBLES. A PEAKIER elite (smaller frac ->
        # average fewer, better candidates) drives markedly straighter: 0.02 -> 0.01 cut lateral wander
        # ~20%, 0.005 ~33%, with 0 contact regressions in sim (even fixed one stress world). Too small
        # (<~0.003) starves the mean. Batch size barely helps by comparison.
        d("plan_elite_frac", PLAN_DEFAULTS["plan_elite_frac"])
        # ACTUATION (drive the robot). plan_actuate publishes wheel commands to a real robot; set
        # it false to run planning as visualization only. All motor-safety conditioning (the
        # left-wheel sign flip, rear-follower, magnitude clamp, slew limit) is in control/command.py.
        d("plan_actuate", True)  # publish /cmd_joints wheel commands
        d("cmd_topic", "/cmd_joints")  # JointState wheel-velocity command topic (to the LLC)
        # WHEEL FEEDBACK: measured wheel velocities from the LLC, used to seed each replan's
        # realized wheel state (motor-lag + body-momentum initial condition). Without it the
        # rollouts plan from wheels-at-rest every frame. Convention/units verified on
        # bags/motors0 + steps_air: all-positive-forward wheel rad/s, same as /cmd_joints
        # (see control/command.joint_states_to_model). When the topic is silent or stale the
        # seed falls back to the last conditioned command. "" disables. Set AT LAUNCH.
        d("joint_states_topic", "/joint_states")
        d(
            "plan_max_omega", 5.0
        )  # hard cap on |wheel velocity| [rad/s] -- the motor safe max (~5, see plan_wmax)
        # hard cap on |d(cmd)/dt| per wheel [rad/s^2]. At DT=0.1s the command may change by
        # max_slew*0.1 per step; 50 let it jump 0->cruise in ONE step (harsh launch, ~5 m/s^2). 6.0
        # ramps 0->~1.3 m/s cruise over ~0.65s (ground ~2.1 m/s^2) -- softer start/stop, still responsive.
        d("plan_max_slew", 6.0)
        # Log the RAW MPPI command next to the conditioned one every Nth planned frame. 0 = off.
        d("plan_debug_cmd", 0)
        # Planner-input dump to /tmp/plan_dump.npz, one-shot. 1 = the next planned frame;
        # 2 = the first frame planned toward the NEXT goal (what you want for "why didn't it turn").
        d("plan_debug_dump", 0)
        # Frame-history recording for the replay page (studies/closed_loop/build_scrub.py): the
        # routing maps, the coarse map and the pose of every Nth planned frame, plus the coarse
        # memory's final layers, written as one npz to this path when the node shuts down. "" = off.
        d("plan_debug_record", "")
        d("plan_debug_record_every", 4)
        # deceleration cap [rad/s^2] -- separate from accel so stops can be firmer than the gentle
        # launch. 12.0 = ground ~4.2 m/s^2, stops from cruise in ~0.32s. None/<=0 would mean symmetric.
        d("plan_max_decel", 12.0)
        # amplify the commanded turn differential. 1.0 = off. REVISED 2026-07-15: post-fix bags showed
        # the differential is realized ~1:1 below the motor ceiling -- the earlier "~half" was SATURATION
        # (over-commanded wheels), not a real drivetrain gain. So boosting over-turns below the limit and
        # worsens saturation at it. Keep at 1.0 while plan_wmax leaves turning headroom (see that
        # param); the fixed-2.0 story in docs/field/turn_differential_hotfix.md is superseded.
        # Transport delay [s] between publishing /cmd_joints and the wheels acting on it. MEASURED
        # at 149-199 ms (scripts/fit_actuator_lag.py); the rollout then plans against commands that
        # land ~2 ticks late instead of instantly. 0.0 = off. Note this makes the planner turn
        # SOONER, not more: it no longer expects a command to bite instantly. If plan_turn_boost is
        # ever raised above 1.0 to compensate for under-turning, re-check it after changing this --
        # the two corrections overlap.
        # Publish /cmd_joints on a TIMER at this rate [Hz] instead of once per point cloud, walking
        # the committed plan between replans. 0 = off (publish once per cloud, as before).
        # Replanning stays at the sensor rate -- a new plan on a 66 ms-old map is barely new
        # information -- but the COMMAND can change faster, which is where the value is: finer slew
        # and goal-brake resolution, and the LLC stays fed if a cloud is dropped. It does NOT reduce
        # the ~175 ms actuator delay, which lives in the LLC velocity loop.
        d("plan_command_delay", dynamics.COMMAND_DELAY)
        d("plan_turn_boost", 1.0)
        # OPTIONAL: self-tune plan_turn_boost online from gyro feedback (control/turn_adapt.py) so the
        # realized yaw matches the plan across terrains + the drivetrain defect -- makes the fixed
        # plan_turn_boost adaptive. False = off (use the fixed value above). When on, plan_turn_boost
        # is the initial guess; plan_turn_boost_tau is the EMA time constant [s] (slow, so it can't
        # fight replanning). Only adapts while turning; clamps to [1, 3].
        d("plan_turn_boost_adapt", False)
        d("plan_turn_boost_tau", 3.0)
        # OPTIONAL fast inner yaw-rate loop (control/yaw_track.py): correct the commanded
        # differential so the REALIZED yaw matches the command actually published. Needs
        # Off by default; see the module docstring for the gains.
        d("plan_yaw_track", False)
        d("plan_yaw_track_kp", 0.4)
        d("plan_yaw_track_ki", 1.0)
        d("plan_yaw_track_deadband", 0.05)  # [rad/s] straight-running idle band
        d("plan_yaw_track_max", 1.5)  # [rad/s] hard clamp on the differential correction
        d("plan_dock_radius", 1.5)  # within this range of the goal: dock (if enabled) or just stop
        d(
            "plan_dock_enable", True
        )  # True = terminal dock; False = just STOP when within dock_radius
        d("plan_reach_radius", 0.3)  # goal reached -> command a (ramped) stop within this range (m)
        # GOAL BRAKE: scale MPPI's forward speed to 0 over the last brake_dist m so the forward-only
        # robot noses in slow and settles AT the goal instead of overshooting/orbiting past it. Cruise
        # speed is untouched beyond brake_dist. 0 = off. Replaces the hard dock/stop radius; validated
        # in sim (~0.2 m settle, zero overshoot) -- see the goal-brake note in control/command.py.
        d("plan_goal_brake_dist", 2.0)  # start braking within this range of the goal (m); 0 = off
        # TURN BRAKE: ceiling on lateral acceleration [m/s^2] -- the robot slows INTO a corner
        # instead of carrying cruise speed through it, then the slew limiter eases it back out.
        # Scales the forward mean AND the turn differential together, so the planned arc is held
        # (scaling mean alone would tighten it). 0 = off. For reference, at plan_wmax 6 and
        # k_turn 1 the HARDEST corner the planner can command is ~1.5 m/s^2 (v 1.05 m/s,
        # wz 1.44 rad/s), so useful values sit below that -- try ~0.6-1.2. Live-tunable.
        d("plan_turn_brake_a_max", 0.0)
        # Look this far ahead along the COMMITTED plan and pre-apply the tightest cap it finds, so
        # the robot brakes BEFORE the corner rather than in it. 0 = reactive only (cap the current
        # command). The plan is plan_horizon * DT long, so this saturates at 2.5 s by default.
        d("plan_turn_brake_lookahead_s", 0.0)
        d("plan_path_width", 0.08)  # intended-path line marker width (m)

    def _cache_params(self) -> None:
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.lidar_topic: str = g("lidar_topic")
        self.odom_topic: str = g("odom_topic")
        self.imu_topic: str = g("imu_topic")
        self.base_frame: str = g("base_frame")
        self.map_frame: str = g("map_frame")
        self.z_crop_enable: bool = g("z_crop_enable")
        self.z_crop_min: float = g("z_crop_min")
        self.z_crop_max: float = g("z_crop_max")
        self.scan_max_range_m: float = g("scan_max_range_m")
        self.scan_min_range_m: float = g("scan_min_range_m")
        self.self_filter_enable: bool = g("self_filter_enable")
        self.self_x_min: float = g("self_x_min")
        self.self_x_max: float = g("self_x_max")
        self.self_y_min: float = g("self_y_min")
        self.self_y_max: float = g("self_y_max")
        self.outlier_enable: bool = g("outlier_enable")
        self.resolution: float = g("resolution")
        self.win_m: float = g("win_m")
        self.route_m: float = g("route_m")
        self.belief_carve_m: float = g("belief_carve_m")
        _max_gyro_dps: float = g("max_gyro_rate_dps")
        # squared rad/s gate, or inf when disabled (0) — compared against |omega|^2 per sample
        self._max_gyro_rate_sq: float = (
            np.deg2rad(_max_gyro_dps) ** 2 if _max_gyro_dps > 0.0 else np.inf
        )
        self.debug_frames: bool = g("debug_frames")
        self.profile_stages: bool = g("profile_stages")
        self.publish_map_tf: bool = g("publish_map_tf")
        self.publish_odom_tf: bool = g("publish_odom_tf")
        self.plan_enable: bool = g("plan_enable")
        self.goal_source: str = g("goal_source")  # "click" | "follow" -- live-switchable
        self.follow_standoff: float = g("follow_standoff")  # stop this far short of the tag (m)
        self.plan_batch: int = g("plan_batch")
        self.plan_horizon: int = g("plan_horizon")
        self.plan_consistency: float = g("plan_consistency")
        self.plan_n_theta: int = g("plan_n_theta")
        self.plan_lat_coarsen: int = g("plan_lat_coarsen")
        self.plan_n_refine: int = g("plan_n_refine")
        self.plan_friction: float = g("plan_friction")
        _ww = float(g("plan_wheel_width"))
        self.plan_wheel_width: float | None = _ww if _ww > 0.0 else None
        self.terrain: str = g("terrain")
        self.k_turn_override: float = g("k_turn")
        self.plan_robust_margin_m: float = g("plan_robust_margin_m")
        self.plan_robust_margin_deg: float = g("plan_robust_margin_deg")
        self.plan_obstacle_step_m: float = g("plan_obstacle_step_m")
        self.plan_nominal_reset: float = g("plan_nominal_reset")
        self.plan_tau_motor: float = g("plan_tau_motor")
        self.plan_goal_running: float = g("plan_goal_running")
        self.plan_effort: float = g("plan_effort")
        self.plan_turn: float = g("plan_turn")
        self.plan_smooth: float = g("plan_smooth")
        self.plan_straight_frac: float = g("plan_straight_frac")
        self.plan_debug_cmd: int = int(g("plan_debug_cmd"))
        self.plan_debug_dump: int = int(g("plan_debug_dump"))
        self.plan_debug_record: str = str(g("plan_debug_record"))
        self.plan_debug_record_every: int = max(1, int(g("plan_debug_record_every")))
        self.plan_spin_frac: float = g("plan_spin_frac")
        self.plan_spin_min: float = g("plan_spin_min")
        self.plan_spin_max: float = g("plan_spin_max")
        self.plan_elite_frac: float = g("plan_elite_frac")
        self.plan_wmax: float = g("plan_wmax")
        self.plan_wmin: float = g("plan_wmin")
        self.plan_reverse_clear_m: float = g("plan_reverse_clear_m")
        self.plan_pivot_cost: float = g("plan_pivot_cost")
        self.plan_z_veto: float = g("plan_z_veto")
        self.plan_charge_per_sigma: float = g("plan_charge_per_sigma")
        self.plan_clear_t_react: float = g("plan_clear_t_react")
        self.plan_clear_v_cruise: float = g("plan_clear_v_cruise")
        self.plan_clear_v_min: float = g("plan_clear_v_min")
        self.plan_clear_lookahead_s: float = g("plan_clear_lookahead_s")
        self.plan_clear_mppi_weight: float = g("plan_clear_mppi_weight")
        self.plan_clear_decel: float = g("plan_clear_decel")
        self.plan_clear_c0: float = g("plan_clear_c0")
        self.plan_clear_t_turn: float = g("plan_clear_t_turn")
        self.plan_clear_route_turn: float = g("plan_clear_route_turn")
        self.plan_clear_v_blind: float = g("plan_clear_v_blind")
        self.plan_n_mu: int = g("plan_n_mu")
        self.plan_mu_span: float = g("plan_mu_span")
        self.plan_mu_adapt: bool = g("plan_mu_adapt")
        self.plan_mu_tau: float = g("plan_mu_tau")
        self.plan_saturation: float = g("plan_saturation")
        self.plan_wall_veto: float = g("plan_wall_veto")
        self.plan_coarse_block_m: float = g("plan_coarse_block_m")
        self.plan_coarse_memory_m: float = g("plan_coarse_memory_m")
        self.plan_coarse_win_m: float = g("plan_coarse_win_m")
        self.plan_bridge_m: float = g("plan_bridge_m")
        self.plan_turn_first_deg: float = g("plan_turn_first_deg")
        self.plan_turn_first_reach_m: float = g("plan_turn_first_reach_m")
        self.plan_actuate: bool = g("plan_actuate")
        self.plan_max_omega: float = g("plan_max_omega")
        self.plan_max_slew: float = g("plan_max_slew")
        self.plan_max_decel: float = g("plan_max_decel")
        self.plan_command_delay: float = g("plan_command_delay")
        self.plan_turn_boost: float = g("plan_turn_boost")
        self.plan_turn_boost_adapt: bool = g("plan_turn_boost_adapt")
        self.plan_turn_boost_tau: float = g("plan_turn_boost_tau")
        self.plan_yaw_track: bool = g("plan_yaw_track")
        self.plan_yaw_track_kp: float = g("plan_yaw_track_kp")
        self.plan_yaw_track_ki: float = g("plan_yaw_track_ki")
        self.plan_yaw_track_deadband: float = g("plan_yaw_track_deadband")
        self.plan_yaw_track_max: float = g("plan_yaw_track_max")
        self.plan_dock_radius: float = g("plan_dock_radius")
        self.plan_dock_enable: bool = g("plan_dock_enable")
        self.plan_reach_radius: float = g("plan_reach_radius")
        self.plan_goal_brake_dist: float = g("plan_goal_brake_dist")
        self.plan_turn_brake_a_max: float = g("plan_turn_brake_a_max")
        self.plan_turn_brake_lookahead_s: float = g("plan_turn_brake_lookahead_s")
        self.plan_path_width: float = g("plan_path_width")

    @staticmethod
    def _resolve_device(name: str) -> wp.context.Device:
        if name == "auto":
            return wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
        return wp.get_device(name)

    # ------------------------------------------------------------------
    # Heavy-object construction
    # ------------------------------------------------------------------

    def _build_outlier_filter(self) -> None:
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        cfg = OutlierFilterConfig(
            search_radius_m=g("outlier_search_radius_m"),
            min_neighbors=g("outlier_min_neighbors"),
        )
        self.outlier_filter = StatisticalOutlierFilter(cfg, device=self.device)

    def _build_planner(self) -> None:
        """Build the MPPI rollout simulator, planner, and cost-to-go, sized to the current
        windows/resolution. Expensive (rollout buffers + CUDA graphs) — only on structural
        param change, never per frame."""
        cell = self.resolution
        ww = wh = int(round(self.win_m / cell))
        rww = rwh = int(round(self.route_m / cell))
        kr = max(1, int(self.plan_lat_coarsen))
        rcny, rcnx, rccell = rwh // kr, rww // kr, cell * kr
        win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
        # terrain-dependent turn gain (outdoor grips harder -> understeers). See dynamics.k_turn_for.
        # explicit k_turn param wins; else the terrain preset (indoor/outdoor)
        if self.k_turn_override >= 0.0:
            kt = self.k_turn_override
            self.get_logger().info(f"planner K_TURN={kt} (k_turn param override)")
        else:
            kt = dynamics.k_turn_for(self.terrain)
            self.get_logger().info(f"planner terrain='{self.terrain}' -> K_TURN={kt}")
        plan_solver = dynamics.planning_solver(k_turn=kt, command_delay=self.plan_command_delay)
        if self.plan_tau_motor >= 0.0:  # -1 = keep dynamics.MOTOR_TAU (the measured default)
            plan_solver.tau_motor = self.plan_tau_motor
        # The mapping plan_* -> planner objects lives in helhest.planner_config, shared with the
        # simulator, so the two cannot build different controllers from the same parameters.
        # Built BEFORE the rollout sim: it decides the batch (rounded to what n_mu divides).
        cfg = planner_config({k: getattr(self, k) for k in PLAN_DEFAULTS})
        self.plan_sim = ForwardSimulator(
            dynamics.robot_params(cfg.wheel_width),
            plan_solver,
            win_grid,
            cfg.batch,
            cfg.horizon,
            self.device,
        )
        self.plan_sim.set_uniform_friction(self.plan_friction)
        # size the in-flight ring to the delay the rollout actually models
        self._cmd_in_flight = deque(
            self._cmd_in_flight, maxlen=max(self.plan_sim.command_delay_steps, 1)
        )
        self.planner = MppiGpu(self.plan_sim, cfg.cost, sampling=cfg.sampling, n_theta=cfg.n_theta)
        self.planner.reset_nominal(cfg.nominal_reset)
        self.planner.set_mu_band(1.0, cfg.mu_span)
        self.governor: ClearanceGovernor | None = None
        if cfg.clearance is not None:
            self.governor = ClearanceGovernor(
                dynamics.robot_params(cfg.wheel_width),
                plan_dt=float(self.planner.cw.dt),
                params=cfg.clearance,
                device=self.device,
            )
        if self.plan_wmin < 0.0:
            self.planner.set_wmin(0.0)  # reverse stays locked until the gate in _plan opens it
        # ONLINE mu estimation: recenter the planner's friction on the realized turn gain (both
        # directions) and size the robust band from the estimator's noise.
        rp = dynamics.robot_params(self.plan_wheel_width)
        if self.plan_mu_adapt:
            self._mu_est = TurnGainEstimator(
                k_turn=kt,
                mu_nominal=self.plan_friction,
                wheel_radius=rp.wheel_radius,
                half_track=rp.half_track,
                dt=dynamics.DT,
                tau_s=self.plan_mu_tau,
            )
            self.get_logger().info(
                f"online mu estimation ON (k_turn={kt}, nominal mu={self.plan_friction}, "
                f"tau={self.plan_mu_tau}s)"
            )
        else:
            self._mu_est = None
        # optional online turn_boost from gyro feedback: alpha = 1 + k_turn*mu matches the plan model.
        if self.plan_turn_boost_adapt:
            rp = dynamics.robot_params(self.plan_wheel_width)
            self._turn_adapt = AdaptiveTurnBoost(
                alpha_model=1.0 + kt * self.plan_friction,
                wheel_radius=rp.wheel_radius,
                half_track=rp.half_track,
                dt=dynamics.DT,
                tau_s=self.plan_turn_boost_tau,
                init=self.plan_turn_boost,
            )
            self.get_logger().info(
                f"adaptive turn_boost ON (alpha_model={1.0 + kt * self.plan_friction:.2f}, "
                f"tau={self.plan_turn_boost_tau}s, init={self.plan_turn_boost})"
            )
        else:
            self._turn_adapt = None
        # Lateral-accel constant for the turn brake: a_lat = lat_gain * mean * diff, from
        # v = R*mean and wz = R*diff/(2*half_track*alpha). alpha = 1 + k_turn*grip/(m*g) is the
        # model's turn resistance; on flat ground with the wheels carrying the full weight that
        # is 1 + k_turn, which is the value the planner itself is tuned against.
        _rp = dynamics.robot_params(self.plan_wheel_width)
        self._lat_gain = _rp.wheel_radius**2 / (2.0 * _rp.half_track * (1.0 + kt))
        # Yaw rate per unit differential, for the inner yaw loop. NOT _lat_gain / R: the turn
        # brake deliberately uses alpha = 1 + k_turn (the mu = 1 worst case) because it is a
        # SAFETY cap and over-braking is harmless. A yaw REFERENCE has to be the planner's own
        # model, alpha = 1 + k_turn*plan_friction, or the loop would steer the robot away from
        # what MPPI actually planned -- about 8% less yaw at the deployed mu 0.8.
        self._yaw_per_diff = _rp.wheel_radius / (
            2.0 * _rp.half_track * (1.0 + kt * self.plan_friction)
        )
        if self.plan_yaw_track:
            self._yaw_track = YawRateTracker(
                yaw_per_diff=self._yaw_per_diff,
                kp=self.plan_yaw_track_kp,
                ki=self.plan_yaw_track_ki,
                deadband=self.plan_yaw_track_deadband,
                max_correction=self.plan_yaw_track_max,
            )
            self.get_logger().info(
                f"yaw-rate loop ON at the plan rate (kp={self.plan_yaw_track_kp}, "
                f"ki={self.plan_yaw_track_ki}, deadband={self.plan_yaw_track_deadband} rad/s)"
            )
        else:
            self._yaw_track = None
        self.ctg = CostToGo(
            GridParams(rcnx, rcny, rccell, 0.0, 0.0),
            dynamics.robot_params(self.plan_wheel_width),
            dynamics.planning_solver(
                k_turn=kt
            ),  # static settle ignores k_turn; passed for consistency
            **cfg.costtogo,
            device=self.device,
        )
        self.planner.cw.lattice_cap = self.ctg._vcap
        self.coarse: CoarseRouter | None = None
        if cfg.coarse["block_m"] > 0.0:
            cw = int(round(self.plan_coarse_win_m / cell))
            memory = None
            if cfg.coarse["memory_m"] > 0.0:
                # anchored on the MAP frame's origin -- the world, through localization -- so it
                # never moves; the robot is near it when the map starts
                nm = int(round(cfg.coarse["memory_m"] / cell))
                memory = GridParams(nm, nm, cell, -0.5 * nm * cell, -0.5 * nm * cell)
            self.coarse = CoarseRouter(
                GridParams(cw, cw, cell, 0.0, 0.0),
                factor=max(1, int(round(cfg.coarse["block_m"] / cell))),
                bridge_m=cfg.coarse["bridge_m"],
                memory_grid=memory,
                device=self.device,
            )
            # the coarse grid's shape is fixed; where it sits in the routing window's frame is
            # passed per frame, since an anchored grid moves under a robot-centred window
            self.ctg.set_coarse(
                GridParams(
                    self.coarse.grid.cells_x,
                    self.coarse.grid.cells_y,
                    self.coarse.grid.cell_size,
                    0.0,
                    0.0,
                )
            )
            self._coarse_cw = cw
            self._coarse_mask = wp.zeros((cw, cw), dtype=wp.float32, device=self.device)
            self.get_logger().info(
                f"coarse layer ON: {self.coarse.grid.cell_size:.2f} m blocks from a "
                f"{self.plan_coarse_win_m:.0f} m raster, "
                + (
                    f"anchored map {cfg.coarse['memory_m']:.0f} m across, "
                    if memory is not None
                    else "window-bound (forgets what scrolls out), "
                )
                + f"shadow bridge {cfg.coarse['bridge_m']:.1f} m"
            )
        # Routing field expressed in the PLANNING window's frame: both windows are robot-centered,
        # so their origins differ by a constant cell offset.
        self.sgrid = GridParams(
            rcnx, rcny, rccell, (ww // 2 - rww // 2) * cell, (wh // 2 - rwh // 2) * cell
        ).build()
        self._plan_kr = kr
        self._plan_dims = (ww, wh, rww, rwh, rcnx, rcny)

    def _goal_callback(self, msg: PoseStamped) -> None:
        """RViz 2D Nav Goal. Assumes the pose is already in map_frame (RViz publishes in its
        fixed frame — set it to the map frame); warns otherwise and uses it as-is."""
        if self.goal_source != "click":
            return  # follow mode owns the goal; set goal_source:=click to drive from RViz
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().warning(
                f"goal frame '{msg.header.frame_id}' != map_frame '{self.map_frame}'; "
                "set the RViz Fixed Frame to the map frame."
            )
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self._prev_plan_U = None  # new goal -> don't smooth against the old goal's plan
        self._goal_reached = False  # new goal -> resume planning
        # plan_debug_dump 2 = "dump the first frame that plans toward the NEXT goal". Arming it
        # from outside and racing the goal in is unreliable: the node is usually still planning
        # toward the previous goal, so "the next planned frame" is the old one.
        if self.plan_debug_dump == 2:
            self._dump_pending = True
            self.plan_debug_dump = 0
        self.get_logger().info(f"goal set: ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f})")

    def _follow_callback(self, msg: PoseStamped) -> None:
        """Follow-me target (e.g. the radio locator on /radio/estimate_pose). Active only while
        goal_source == 'follow': the pose is transformed into map_frame via TF and becomes the live
        goal on every update, so a MOVING tag is continuously chased. The warm start (_prev_plan_U)
        is deliberately NOT reset here -- the target drifts smoothly, so the last plan is still a
        good seed. The reach latch is handled per-frame in the plan loop
        (stop within plan_reach_radius, resume when the tag moves away). Ignored in 'click' mode."""
        if self.goal_source != "follow":
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id, rclpy.time.Time()
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"follow: no TF {msg.header.frame_id!r} -> {self.map_frame!r} ({exc})",
                throttle_duration_sec=2.0,
            )
            return
        p = do_transform_pose(msg.pose, tf)
        px, py = p.position.x, p.position.y
        # Stand off short of the tag: it rides on a person (a LiDAR obstacle), so a goal placed ON
        # them is untraversable and the router walls off. Aim follow_standoff m in front, back along
        # the robot->tag line, into free space. Needs the robot's map pose; if that TF is missing,
        # fall back to the raw tag point (keeps following, may wall off).
        if self.follow_standoff > 0.0:
            try:
                rb = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, rclpy.time.Time()
                )
                rx, ry = rb.transform.translation.x, rb.transform.translation.y
                dx, dy = px - rx, py - ry
                dist = float(np.hypot(dx, dy))
                if dist > self.follow_standoff:
                    s = (dist - self.follow_standoff) / dist  # fraction of the way to the tag
                    px, py = rx + s * dx, ry + s * dy
                else:
                    px, py = rx, ry  # already within standoff -> hold position
            except TransformException:
                pass  # no robot pose -> target the tag directly
        self.goal_xy = (px, py)

    def _on_parameters_changed(self, params) -> SetParametersResult:
        """VALIDATION only -- this runs BEFORE the values are committed.

        `add_on_set_parameters_callback` is a pre-set hook: inside it `get_parameter` still
        returns the OLD value. Everything that re-reads or rebuilds therefore has to happen in
        the POST-set callback below, or a runtime `param set` lands one change late -- the
        rebuild runs on the previous values and the new ones only appear when something else is
        set. That is silent, and it wasted a measurement in this repo's own investigation: a live
        `plan_turn 0.03` was read back as 0.2 and the arm was scored as if it had applied.
        """
        return SetParametersResult(successful=True)

    def _on_parameters_applied(self, params) -> None:
        """POST-set: the values are live now, so cache them and rebuild what depends on them."""
        names = {p.name for p in params}
        try:
            self._cache_params()
            if "device" in names:
                self.device = self._resolve_device(self.get_parameter("device").value)
            if names & _OUTLIER_BUILD:
                self._build_outlier_filter()
            if self.plan_enable and (names & _PLAN_BUILD or self.planner is None):
                self._build_planner()  # (re)build on structural change or first enable
            if "device" in names:  # device moved -> device-resident state is stale
                self._bframe, self._bbuf = None, {}
                self._preproc = None  # buffers live on the old device
        except Exception as exc:  # a bad value must not kill the node
            self.get_logger().error(f"applying parameters failed: {exc}")
            return

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_states_callback(self, msg: JointState) -> None:
        """Measured wheel velocities from the LLC -> the rollouts' realized-wheel seed."""
        om = joint_states_to_model(list(msg.name), list(msg.velocity))
        if om is not None:
            self._wheel_meas = om
            self._wheel_meas_t = self.get_clock().now().nanoseconds * 1e-9

    def _imu_callback(self, msg: Imu) -> None:
        q = msg.orientation
        if q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w > 0.5:  # buffer valid fused orientations
            w = msg.angular_velocity
            # Drop a single-sample gyro glitch: the yaw-rate loop and the turn adapter read this
            # buffer, and one 8000 deg/s spike sample is a phantom 80 deg of yaw.
            if w.x * w.x + w.y * w.y + w.z * w.z > self._max_gyro_rate_sq:
                return
            base_R_imu = self._gyro_base_rotation(msg.header.frame_id)
            if base_R_imu is None:  # IMU->base TF not ready yet — skip until it is
                # Loud, because nothing else is: an IMU frame missing from TF drops every sample,
                # and the yaw-rate loop, the mu estimate and the turn adapter then simply never run
                # (cras_odin_driver's /odin1/imu is in `imu`, which its own TF does not contain).
                self.get_logger().warning(
                    f"IMU frame '{msg.header.frame_id}' has no TF to '{self.base_frame}': gyro "
                    "samples dropped -- yaw-rate loop, mu estimate and turn adapter are OFF",
                    throttle_duration_sec=10.0,
                )
                return
            w_base = base_R_imu @ np.array([w.x, w.y, w.z])
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._imu_buffer.append((t, np.array([q.x, q.y, q.z, q.w]), w_base))

    def _synced_callback(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        try:
            self._process(cloud_msg, odom_msg)
        except Exception as exc:
            # Log the traceback, not just the message: this handler swallows EVERY per-frame
            # failure, and a bare message gives no line to look at -- the whole pipeline can be
            # dying each frame with nothing to point at.
            self.get_logger().error(f"elevation error: {exc}\n{traceback.format_exc()}")

    def _ck(self, label: str) -> None:
        """Profiling checkpoint: sync the GPU (Warp is async) and accrue time since the last _ck."""
        if not self.profile_stages:
            return
        wp.synchronize()
        now = time.perf_counter()
        self._prof[label] = self._prof.get(label, 0.0) + (now - self._prof_t)
        self._prof_t = now

    def _process(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        self._frame += 1
        if self.profile_stages:
            wp.synchronize()
            self._prof_t = time.perf_counter()
        odom_T_base = self._odom_to_matrix(odom_msg)
        scan = self._scan_in_base(cloud_msg)
        if scan is None or scan[0].shape[0] == 0:
            self.get_logger().warning("Empty / untransformable scan — skipping.")
            return
        points_sensor, base_T_sensor = scan
        # Sensor->base transform, z / self-footprint / range rejection and compaction, all on
        # device in ONE pass. Done on the host these were four full-cloud numpy copies (~9.6 ms
        # for a 131k-point sweep); the cloud now crosses to the GPU once and stays there.
        scan_buf, n_scan = self._scan_preproc(points_sensor, base_T_sensor)
        if n_scan == 0:
            self.get_logger().warning("crop/self-filter removed all points — check bounds.")
            return
        self._ck("preproc")
        # The pose is Odin's on-device SLAM, trusted as is: the map frame is the odometry frame.
        world_T_base = odom_T_base
        scan_wp = self._denoise(scan_buf[:n_scan])
        self._ck("denoise")
        world_scan = transform_points(scan_wp, len(scan_wp), world_T_base)
        stamp = cloud_msg.header.stamp.sec + cloud_msg.header.stamp.nanosec * 1e-9
        self._belief_update(world_scan, world_T_base @ base_T_sensor, stamp)
        self._ck("belief_update")
        mf = self._build_maps(world_T_base)
        self._ck("build_maps")
        if mf is not None:
            self._publish_maps(mf, cloud_msg.header.stamp)
            self._ck("publish_maps")
            if self.plan_enable and self.goal_xy is not None and self.planner is not None:
                self._plan(mf, world_T_base, cloud_msg.header.stamp)
        self._cache_map_correction(world_T_base, odom_msg)
        self._ck("cache_tf")
        if self.profile_stages:
            self._prof_n += 1
            if self._prof_n % 30 == 0:
                parts = " ".join(
                    f"{k}={1000 * v / self._prof_n:.1f}"
                    for k, v in sorted(self._prof.items(), key=lambda kv: -kv[1])
                )
                total = 1000 * sum(self._prof.values()) / self._prof_n
                self.get_logger().info(
                    f"PROFILE avg ms/frame (n={self._prof_n}) total={total:.1f} | {parts}"
                )

    # ------------------------------------------------------------------
    # The planning maps: one elevation belief, cropped (perception/belief_frame.py)
    # ------------------------------------------------------------------

    def _belief_update(
        self, world_scan: wp.array, world_T_sensor: np.ndarray, stamp: float
    ) -> None:
        """Fold this frame's scan into the belief (created on the first scan)."""
        ex, ey = float(world_T_sensor[0, 3]), float(world_T_sensor[1, 3])
        if self._bframe is None:
            cell = self.resolution
            span = max(self.route_m, self.plan_coarse_win_m)
            # On the world lattice the anchored coarse memory uses (origin -memory/2, whole cells),
            # so its raster pools into the memory's blocks exactly; both recenter in whole cells.
            lat0 = 0.0
            coarse = getattr(self, "coarse", None)
            if coarse is not None and coarse.persistent:
                lat0 = float(coarse.grid.origin_x)
            x0 = lat0 + round((ex - 0.5 * span - lat0) / cell) * cell
            y0 = lat0 + round((ey - 0.5 * span - lat0) / cell) * cell
            n = int(round(span / cell))
            self._bframe = BeliefFrame(
                (x0, x0 + n * cell, y0, y0 + n * cell),
                cell,
                carve_range=self.belief_carve_m,
                device=self.device,
            )
            self._bframe_t = None
        dt = 0.0 if self._bframe_t is None else max(0.0, stamp - self._bframe_t)
        self._bframe_t = stamp
        self._bframe.update(world_scan, world_T_sensor[:3, 3].copy(), (ex, ey), dt)

    def _belief_buf(self, name: str, n: int) -> wp.array:
        b = self._bbuf.get(name)
        if b is None or b.shape[0] != n:
            b = self._bbuf[name] = wp.zeros((n, n), dtype=wp.float32, device=self.device)
        return b

    def _build_belief_maps(self, world_T_base: np.ndarray) -> _MapFrame:
        """The belief path: every planning map is a crop of the one belief window, on its lattice.
        The device crops are what the planner reads; the numpy views are for RViz and recording."""
        bf = self._bframe
        h, m, _, _ = bf.layers()
        n = bf.measured.shape[0]
        cell = self.resolution
        rww = int(round(self.route_m / cell))
        lww = int(round(self.win_m / cell))
        off_r, off_l = n // 2 - rww // 2, n // 2 - lww // 2
        route_h = bf.crop(h, off_r, off_r, self._belief_buf("route_h", rww))
        route_m = bf.crop(m, off_r, off_r, self._belief_buf("route_m", rww))
        local_h = bf.crop(h, off_l, off_l, self._belief_buf("local_h", lww))
        local_m = bf.crop(m, off_l, off_l, self._belief_buf("local_m", lww))
        relev_mem = route_h.numpy()
        rmeasured = route_m.numpy() > 0.5
        elev_local = local_h.numpy()
        lmeasured = local_m.numpy() > 0.5
        return _MapFrame(
            elev_local=elev_local,
            elev_local_view=np.where(lmeasured, elev_local, np.nan).astype(np.float32),
            relev_view=np.where(rmeasured, relev_mem, np.nan).astype(np.float32),
            relev_mem=relev_mem,
            relev_measured=rmeasured,
            cell=cell,
            ex=float(world_T_base[0, 3]),
            ey=float(world_T_base[1, 3]),
            lxmin=bf.xmin + off_l * cell,
            lymin=bf.ymin + off_l * cell,
            rxmin=bf.xmin + off_r * cell,
            rymin=bf.ymin + off_r * cell,
            dev={"local_h": local_h, "local_m": local_m, "off_r": off_r},
        )

    def _build_maps(self, world_T_base: np.ndarray) -> _MapFrame | None:
        """This frame's planning maps: crops of the belief window (see `_build_belief_maps`)."""
        return None if self._bframe is None else self._build_belief_maps(world_T_base)

    def _publish_maps(self, mf: _MapFrame, stamp) -> None:
        self._publish_grid(self.pub_local, mf.elev_local_view, mf.lxmin, mf.lymin, mf.cell, stamp)
        self._publish_grid(self.pub_global, mf.relev_view, mf.rxmin, mf.rymin, mf.cell, stamp)
        self._publish_frame_marker(mf, stamp)

    def _publish_frame_marker(self, mf: _MapFrame, stamp) -> None:
        """A floating text label with the current processed-frame index (over the robot), so
        the exact frame is readable in RViz — e.g. to pin down when tracking goes off."""
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.map_frame
        m.ns = "frame"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(mf.ex)
        m.pose.position.y = float(mf.ey)
        m.pose.position.z = 2.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.6  # text height (m)
        m.color = ColorRGBA(r=1.0, g=1.0, b=0.2, a=1.0)
        m.text = f"#{self._frame}"
        self.pub_frame.publish(m)

    def _record_frame(
        self,
        mf: _MapFrame,
        V: wp.array,
        world_T_base: np.ndarray,
        eyaw: float,
        d_goal: float,
        stamp,
    ) -> None:
        """Keep this planned frame for the replay page, in drive_sim's npz layout.

        The page draws every layer over ONE lattice, so the routing field (kr x kr coarser than
        the raster) is expanded back onto the raster's cells and padded to its size. The command
        stored is the one published on the PREVIOUS planned frame: this frame's is not
        conditioned yet at this point, and the difference is one frame.
        """
        rec = self._rec
        rec["trail"].append([mf.ex, mf.ey])
        self._rec_planned += 1
        if (self._rec_planned - 1) % self.plan_debug_record_every != 0:
            return
        rwh, rww = mf.relev_mem.shape
        kr = self._plan_kr
        vh = V.numpy()
        n_theta = vh.shape[2]
        cap = float(self.ctg._vcap)

        def expand(a: np.ndarray, fill: float) -> np.ndarray:
            a = np.repeat(np.repeat(a, kr, axis=0), kr, axis=1)
            out = np.full((rwh, rww), fill, np.float32)
            out[: a.shape[0], : a.shape[1]] = a[:rwh, :rww]
            return out

        rec["h"].append(np.asarray(mf.relev_mem, np.float32))
        rec["seen"].append(np.asarray(mf.relev_measured, np.uint8))
        rec["blk"].append(expand(self.ctg.blocked.numpy().mean(2), 1.0))
        rec["v"].append(expand(vh.min(2), cap))
        rec["route"].append(expand((vh < 0.9 * cap).mean(2), 0.0))
        rec["cv"].append(
            np.zeros((1, 1), np.float32)
            if self.coarse is None
            else self.coarse.V.numpy()[:, :, 0].astype(np.float32)
        )
        R = world_T_base[:3, :3]
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))  # nose-up = NEGATIVE
        rcell = mf.cell * kr
        c, r = int((mf.ex - mf.rxmin) / rcell), int((mf.ey - mf.rymin) / rcell)
        t = int(round((eyaw % (2.0 * np.pi)) / (2.0 * np.pi / n_theta))) % n_theta
        v_here = (
            float(vh[r, c, t]) if 0 <= r < vh.shape[0] and 0 <= c < vh.shape[1] else float("nan")
        )
        rec["meta"].append(
            [
                self._rec_planned - 1,
                mf.ex,
                mf.ey,
                eyaw,
                float(self._prev_cmd[0]),  # /cmd_joints order: left, rear, right
                float(self._prev_cmd[2]),
                d_goal,
                mf.rxmin,
                mf.rymin,
                roll,
                pitch,
                v_here,
            ]
        )
        rec["goal"].append([float(self.goal_xy[0]), float(self.goal_xy[1])])
        rec["t"].append(stamp.sec + stamp.nanosec * 1e-9)

    def _record_save(self) -> None:
        """Write the recording, plus the coarse map's own layers, so its sealing can be audited
        offline."""
        rec = self._rec
        if rec is None or not rec["meta"]:
            return
        t = np.asarray(rec["t"], np.float64)
        out: dict = dict(
            plan_config=np.array(json.dumps({k: getattr(self, k) for k in PLAN_DEFAULTS})),
            dt=float(np.median(np.diff(t))) if len(t) > 1 else 0.0,
            trail=np.asarray(rec["trail"], np.float64),
            goal=np.asarray(rec["goal"][-1], np.float64),
            cell=float(self.resolution),
            reached=bool(self._goal_reached),
            coarse_cell=(0.0 if self.coarse is None else self.coarse.grid.cell_size),
            coarse_memory=(self.coarse is not None and self.coarse.persistent),
            coarse_bounds=np.array(
                (
                    [self.coarse.grid.origin_x, self.coarse.grid.origin_y]
                    if self.coarse is not None
                    else [0.0, 0.0]
                ),
                np.float64,
            ),
            **{f"hist_{k}": np.asarray(v) for k, v in rec.items() if k != "trail"},
        )
        if self.coarse is not None:
            for k in ("passable", "seen", "coverage", "bridged", "floor"):
                out[f"coarse_{k}"] = getattr(self.coarse, k).numpy()
            out["coarse_V"] = self.coarse.V.numpy()[:, :, 0]
        np.savez_compressed(self.plan_debug_record, **out)
        self.get_logger().info(
            f"plan recording -> {self.plan_debug_record} ({len(rec['meta'])} frames)"
        )

    def destroy_node(self) -> bool:
        if self._rec is not None:
            try:
                self._record_save()
            except Exception as e:  # a diagnostic must never block the shutdown
                self.get_logger().warn(f"plan recording failed: {e}")
        return super().destroy_node()

    def _publish_grid(self, pub, elev: np.ndarray, xmin: float, ymin: float, cell: float, stamp):
        pub.publish(elevation_to_cloud(elev, xmin, ymin, cell, stamp, self.map_frame))

    # ------------------------------------------------------------------
    # MPPI planning (visualization only)
    # ------------------------------------------------------------------

    def _reverse_clear(self, mf: _MapFrame, yaw: float) -> bool:
        """True when a robot-width strip straight behind base_link (0.3 .. plan_reverse_clear_m)
        is >= 95% MEASURED in the belief -- the map-knowledge gate for reverse driving.
        Out-of-window samples count as blind."""
        cell = mf.cell
        meas = mf.relev_measured
        ds = np.arange(0.3, self.plan_reverse_clear_m + 1e-6, cell)
        lats = np.arange(-0.6, 0.6 + 1e-6, cell)
        # p = base - d*fwd + lat*left, fwd = (cos, sin), left = (-sin, cos)
        px = mf.ex - np.cos(yaw) * ds[:, None] - np.sin(yaw) * lats[None, :]
        py = mf.ey - np.sin(yaw) * ds[:, None] + np.cos(yaw) * lats[None, :]
        cols = np.floor((px - mf.rxmin) / cell).astype(int)
        rows = np.floor((py - mf.rymin) / cell).astype(int)
        inb = (rows >= 0) & (rows < meas.shape[0]) & (cols >= 0) & (cols < meas.shape[1])
        hit = np.zeros(px.shape, bool)
        hit[inb] = meas[rows[inb], cols[inb]]
        return bool(hit.mean() >= 0.95)

    def _turn_brake_lookahead(self, turn_boost: float) -> float:
        """Tightest turn-brake speed scale over the next `plan_turn_brake_lookahead_s` of the plan.

        The reactive brake in `condition_command` only sees the step being published, so it slows
        IN the corner. The committed plan already says what the next plan_horizon*DT holds, so
        scanning it and pre-applying the worst cap makes the robot shed speed BEFORE the corner.
        Returns 1.0 (no anticipation) when disabled or when nothing ahead exceeds the ceiling.
        """
        if self.plan_turn_brake_a_max <= 0.0 or self.plan_turn_brake_lookahead_s <= 0.0:
            return 1.0
        if self.planner is None:
            return 1.0
        U = self.planner.nominal()  # [T, 2] committed (wL, wR), model convention
        n = min(len(U), max(1, int(round(self.plan_turn_brake_lookahead_s / dynamics.DT))))
        scale = 1.0
        for k in range(n):
            wl_k, wr_k = float(U[k][0]), float(U[k][1])
            mean_k = 0.5 * (wl_k + wr_k)
            diff_k = (wr_k - wl_k) * float(turn_boost)  # match what the output will actually send
            a_lat = abs(self._lat_gain * mean_k * diff_k)
            if a_lat > self.plan_turn_brake_a_max:
                scale = min(scale, math.sqrt(self.plan_turn_brake_a_max / a_lat))
        return scale

    def _plan(self, mf: _MapFrame, world_T_base: np.ndarray, stamp) -> None:
        """Run MPPI toward the goal on this frame's maps; publish the intended path.

        Terrain = the belief's MPPI crop; the routing cost-to-go is solved on its routing crop,
        pooled to the routing cell, with the belief's measurement sd and drift.
        """
        gx, gy = self.goal_xy
        ez = float(world_T_base[2, 3])
        eyaw = float(np.arctan2(world_T_base[1, 0], world_T_base[0, 0]))
        _, _, _, _, rcnx, rcny = self._plan_dims
        kr = self._plan_kr
        state_l = np.array([mf.ex - mf.lxmin, mf.ey - mf.lymin, eyaw], np.float32)
        goal_l = (gx - mf.lxmin, gy - mf.lymin)
        goal_r = (gx - mf.rxmin, gy - mf.rymin)

        # --- goal reached: announce once, then IDLE (skip cost-to-go + MPPI) until the goal changes.
        # Keep publishing a (ramped) stop each frame so the LLC stays fed at rest. Resumes on a new goal.
        # In "follow" mode the latch is NOT sticky: _goal_reached tracks "within radius" per-frame,
        # so the robot stops on top of a stationary tag but resumes the instant the tag moves away.
        d_goal = float(np.hypot(gx - mf.ex, gy - mf.ey))
        within = d_goal < self.plan_reach_radius
        if self.goal_source == "follow":
            if within and not self._goal_reached:
                self.get_logger().info(
                    f"follow: within {self.plan_reach_radius:.2f} m -- holding at tag"
                )
            self._goal_reached = within  # per-frame, not latched -> chases again when the tag moves
        elif not self._goal_reached and within:
            self._goal_reached = True
            self.get_logger().info(
                f"REACHED goal (d={d_goal:.2f} m) -- stopping; idle until a new goal is set."
            )
        if self._goal_reached:
            if self._yaw_track is not None:
                self._yaw_track.reset()  # at rest: a held integrator is a lurch on the next goal
            if self.plan_actuate:
                cmd = condition_command(
                    0.0,
                    0.0,
                    self._prev_cmd,
                    max_omega=self.plan_max_omega,
                    max_slew=self.plan_max_slew,
                    max_decel=self.plan_max_decel,
                    dt=self._command_dt(dynamics.DT),
                    turn_boost=self.plan_turn_boost,
                )
                self._prev_cmd = cmd
                self._publish_cmd(cmd)
            return
        with wp.ScopedDevice(self.device):
            self.plan_sim.set_terrain(mf.dev["local_h"])
            self._ck("plan:set_terrain")
            # Seed the rollouts' REALIZED initial state -- without this every replan planned from
            # wheels-at-rest and zero body twist (command_history only covers in-flight COMMANDS).
            # Wheel speed: measured /joint_states when fresh, else the last conditioned command.
            # Body twist: the odometry's child-frame twist when fresh, else derived from the
            # wheel seed through the turn model (vy unobservable there -> 0).
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if self._wheel_meas is not None and now_s - self._wheel_meas_t < 0.3:
                wheel_seed = self._wheel_meas
            else:
                wheel_seed = to_engine_order(self._prev_cmd)
            self.plan_sim.set_initial_wheel_omega(wheel_seed)
            if self._twist_meas is not None and now_s - self._twist_meas_t < 0.3:
                twist_seed = self._twist_meas
            else:
                alpha = 1.0 + self.plan_sim.solver.k_turn * self.plan_friction
                twist_seed = dynamics.twist_from_wheels(wheel_seed, alpha)
            self.plan_sim.set_initial_twist(twist_seed)
            self._ck("plan:seed_state")
            # REVERSE gate (only when plan_wmin < 0): give the cost kernel this frame's observed-
            # cell mask (reversing over blind cells is penalized) and unlock the negative sampling
            # floor only while a robot-width strip behind base_link is measured in the belief --
            # no rear sensor, so reverse may only use REMEMBERED ground.
            # the governor needs the same mask: it slows the robot over ground nobody has measured
            if self.plan_wmin < 0.0 or self.governor is not None:
                self.planner.set_measured(mf.dev["local_m"])
            if self.plan_wmin < 0.0:
                rev_open = self._reverse_clear(mf, eyaw)
                self.planner.set_wmin(self.plan_wmin if rev_open else 0.0)
                if rev_open != self._rev_open:
                    self.get_logger().info(
                        f"reverse {'UNLOCKED' if rev_open else 'locked'} "
                        f"(map behind {'measured' if rev_open else 'blind'})"
                    )
                self._rev_open = rev_open
            # the routing grid: the belief pooled kr x kr, sigma and drift beside the height
            bf, off_r = self._bframe, mf.dev["off_r"]
            Hc = self._belief_buf("pool_h", rcny)
            Mc = self._belief_buf("pool_m", rcny)
            route_sd = self._belief_buf("pool_sd", rcny)
            route_drift = self._belief_buf("pool_drift", rcny)
            bf.pool(off_r, off_r, kr, Hc, Mc, route_sd, route_drift)
            # WHICH WAY -- the coarse layer, pooled from the belief window (wider than the routing
            # window) and remembered across frames; its value prices the routing window's border.
            vc = None
            coarse_origin = None
            if self.coarse is not None:
                cw = self._coarse_cw
                cell = mf.cell
                # the belief window's centred cw x cw crop, on its (and the memory's) lattice
                off_c = bf.measured.shape[0] // 2 - cw // 2
                cxmin, cymin = bf.xmin + off_c * cell, bf.ymin + off_c * cell
                if self.coarse.persistent:
                    cx0, cy0 = self.coarse.grid.origin_x, self.coarse.grid.origin_y
                else:
                    cx0, cy0 = cxmin, cymin
                c_h = bf.crop(bf.height, off_c, off_c, self._belief_buf("coarse_h", cw))
                c_m = bf.crop(bf.measured, off_c, off_c, self._coarse_mask)
                if self.coarse.persistent:
                    vc = self.coarse.solve(c_h, c_m, (gx, gy), (cxmin, cymin))
                else:
                    vc = self.coarse.solve(c_h, c_m, (gx - cxmin, gy - cymin))
                coarse_origin = (cx0 - mf.rxmin, cy0 - mf.rymin)
                self._ck("plan:coarse")
            V = self.ctg.compute(
                Hc,
                goal_r,
                measured=Mc,
                sigma=route_sd,
                drift=route_drift,
                coarse_value=vc,
                coarse_origin=coarse_origin,
            )
            self._ck("plan:ctg")
            if vc is not None:
                # for RViz: the coarse cost-to-go as a height grid, unreachable blocks left out
                cvn = vc.numpy()[:, :, 0]
                self._publish_grid(
                    self.pub_coarse,
                    np.where(cvn < 0.9 * self.coarse.solver._inf, cvn, np.nan).astype(np.float32),
                    cx0,
                    cy0,
                    self.coarse.grid.cell_size,
                    stamp,
                )
            if self._rec is not None:
                self._record_frame(mf, V, world_T_base, eyaw, d_goal, stamp)
            # ONE-SHOT PLANNER DUMP. Set plan_debug_dump to 1 and the next planned frame writes
            # everything the planner was given -- routing elevation, the MEASURED mask, the goal
            # and pose in the routing frame, and V -- so an offline probe can be run on exactly
            # what the node saw. Every "why did it not move" question this session has come down
            # to a difference between the live belief and a synthetic map, and there was no way
            # to close that gap without this.
            if self.plan_debug_dump == 1 or getattr(self, "_dump_pending", False):
                self.plan_debug_dump = 0
                self._dump_pending = False
                try:
                    np.savez_compressed(
                        "/tmp/plan_dump.npz",
                        elevation=Hc.numpy(),
                        measured=Mc.numpy(),
                        V=V.numpy(),
                        goal_r=np.asarray(goal_r, np.float32),
                        # MPPI's own frame: the rollouts start at state_l and chase goal_l, and V
                        # is placed into this frame through sgrid's origin. Without all three an
                        # offline probe cannot line the value field up with the robot.
                        state_l=np.asarray(state_l, np.float32),
                        goal_l=np.asarray(goal_l, np.float32),
                        sgrid_origin=np.asarray(
                            [self.sgrid.origin_x, self.sgrid.origin_y], np.float32
                        ),
                        cell=np.float32(self.resolution * max(1, int(self.plan_lat_coarsen))),
                        # the fine terrain the ROLLOUTS drive on, at the map resolution
                        elev_local=np.asarray(mf.elev_local, np.float32),
                        fine_cell=np.float32(self.resolution),
                    )
                    self.get_logger().info("plan dump -> /tmp/plan_dump.npz")
                except Exception as e:  # a diagnostic must never take the planner down
                    self.get_logger().warn(f"plan dump failed: {e}")
            # V with a way out of every no-route pose (CostToGo._escape_kernel): where V is capped
            # MPPI otherwise follows a straight line to the goal, into walls and dead ends
            self.planner.set_lattice(self.ctg.V_escape, self.sgrid)
            if self.planner.cw.veto > 0.0:  # walls are a hard no for the controller too
                self.planner.set_veto(self.ctg.hazard, self.sgrid)
            if self.planner.cw.clear_time > 0.0:  # the wall-distance map the clearance cost reads
                self.planner.update_clearance()
                self._ck("plan:clear_map")
            self._load_command_history()
            self.planner.replan(state_l, goal_l, int(self.plan_n_refine))
            self._ck("plan:replan")
        # PLAN CONSISTENCY: EMA the nominal toward last frame's plan, shifted one step forward (the
        # receding horizon) so the committed maneuver is stable frame-to-frame instead of jittering on
        # open ground. Feeds the next replan's warm-start too. plan_consistency = 0 disables it.
        if self.plan_consistency > 0.0:
            U = self.planner.nominal()
            if self._prev_plan_U is not None and self._prev_plan_U.shape == U.shape:
                shifted = np.roll(self._prev_plan_U, -1, axis=0)
                shifted[-1] = self._prev_plan_U[-1]
                # ...but never ACROSS a manoeuvre class. MPPI's elite mean is mode-coherent by
                # construction (it will not average a spin with an arc, or a left-passer with a
                # right-passer); this EMA sits OUTSIDE it and would undo that, blending a spin
                # with last frame's forward arc into neither. Smoothing is for jitter within a
                # manoeuvre, not for the decision to change manoeuvre -- so on a class change the
                # new plan is taken whole and the history restarts from it.
                if _same_manoeuvre(U, shifted):
                    U = (1.0 - self.plan_consistency) * U + self.plan_consistency * shifted
                    self.planner.set_nominal(U)
            self._prev_plan_U = U.copy()
        # candidate 0 is the committed nominal rollout; window-local -> map coords.
        # Rollout 0 ONLY -- it is the nominal (mppi.py: "candidate 0 keeps the nominal"), and the
        # published path is the only consumer. Reading the whole [T+1, B] tensor pulled 1.2 MB to
        # the host every frame to use 312 bytes of it. Warp copies the strided column directly, so
        # this is a device-side slice rather than a host-side one: 0.63 -> 0.17 ms, same values.
        nominal_xy = self.planner.sim.controlled[:, 0].numpy()  # [T+1, 3] = (x, y, yaw)
        self._ck("plan:readback")
        origin = np.array([mf.lxmin, mf.lymin], np.float32)
        self._publish_path(nominal_xy[:, :2] + origin, ez, stamp)
        self._ck("plan:pub_path")

        # --- ACTUATION: turn the plan into a conditioned /cmd_joints command (default OFF) ---
        if not self.plan_actuate:
            return
        d = float(np.hypot(gx - mf.ex, gy - mf.ey))  # robot -> goal distance
        from_plan = False  # True only on the MPPI branch, whose command IS a plan to walk
        if d < self.plan_reach_radius:
            wl, wr = 0.0, 0.0  # reached -> stop (the slew limiter ramps the command down)
        elif self.plan_dock_enable and d < self.plan_dock_radius:
            # terminal dock; with reverse unlocked it may also PIVOT toward an off-axis goal
            u = dock_control(
                state_l,
                goal_l,
                wmax=self.plan_max_omega,
                wmin=self.plan_wmin if self._rev_open else 0.0,
            )
            wl, wr = float(u[0]), float(u[1])
        else:
            # MPPI drives; the goal brake (in condition_command) bleeds off speed on the final
            # approach so it settles instead of orbiting. With the dock disabled this branch covers
            # the whole reach_radius..inf band -- the continuous brake replaces the hard stop-radius.
            from_plan = True
            u0 = self.planner.nominal()[0]  # first committed step (wL, wR), model convention
            wl, wr = float(u0[0]), float(u0[1])
            wl_raw, wr_raw = wl, wr  # before the yaw loop and the conditioner touch them
            if self.plan_turn_first_deg > 0.0:
                # spin first when the route lies well behind (control/command.turn_first); the way
                # on is read off the routing field around the robot, in the routing window's frame
                bearing = self.ctg.descent_bearing(
                    mf.ex - mf.rxmin, mf.ey - mf.rymin, self.plan_turn_first_reach_m
                )
                if np.isfinite(bearing):
                    err = (bearing - eyaw + np.pi) % (2.0 * np.pi) - np.pi
                    wl, wr = turn_first(
                        wl,
                        wr,
                        err,
                        start_deg=self.plan_turn_first_deg,
                        # last PUBLISHED differential, [L, rear, R] -> R - L
                        prev_diff=float(self._prev_cmd[2] - self._prev_cmd[0]),
                        # measured ground speed from the odometry twist, when fresh
                        speed=(
                            float(np.hypot(self._twist_meas[0], self._twist_meas[1]))
                            if self._twist_meas is not None and now_s - self._twist_meas_t < 0.3
                            else None
                        ),
                    )
            if self.governor is not None:
                # only as fast as the room along the next second of the plan allows
                sim = self.planner.sim
                wl, wr = self.governor.cap(
                    wl, wr, sim.controlled, sim.elevation, self.planner.measured, sim.grid
                )
                self._ck("plan:governor")
        # rear-follower + goal brake + turn boost + magnitude clamp + slew limit, all in control/command.py
        turn_boost = (
            self._turn_adapt.turn_boost if self._turn_adapt is not None else self.plan_turn_boost
        )
        if from_plan:
            # The adaptive turn_boost pairs a commanded differential with the yaw it produced, so
            # it belongs on the PLAN cadence -- once per committed plan, not per publish.
            if (
                self._turn_adapt is not None
                and self._imu_buffer
                and self._last_diff_out is not None
            ):
                self._turn_adapt_update(self._last_diff_out, float(self._imu_buffer[-1][2][2]))
            self._last_diff_out = float(self._prev_cmd[2] - self._prev_cmd[0])
        # INNER YAW LOOP. The reference is the UNCORRECTED plan through the same conditioner:
        # post-turn-brake and post-slew, but free of the loop's own output. Referencing the
        # corrected command makes reference and measurement both scale with the correction, so the
        # error has no fixed point and the loop inflates the turn even at zero model error
        # (measured: peak yaw 0.517 -> 0.575 rad/s on correctly-modelled ground).
        ref_cmd = None
        if self._yaw_track is not None and from_plan:
            ref_cmd = self._conditioned(wl, wr, d, turn_boost)
            half = 0.5 * self._yaw_track.correction  # differential only; the planner owns speed
            wl, wr = wl - half, wr + half
        elif self._yaw_track is not None:
            self._yaw_track.reset()  # stopping, docking or held: do not carry an integrator over
        # RAW PLAN vs PUBLISHED. Everything between the two -- yaw loop, turn boost, goal brake,
        # magnitude clamp, slew limit -- can only be told apart from outside by luck, and a turn
        # that fails is exactly the case where you need to know which side lost it. Throttled, so
        # it costs nothing on the straight-line cruise that dominates a run.
        if from_plan and self.plan_debug_cmd:
            self._dbg_n = getattr(self, "_dbg_n", 0) + 1
            if self._dbg_n % max(1, int(self.plan_debug_cmd)) == 0:
                self.get_logger().info(
                    f"cmd: mppi ({wl_raw:+.2f},{wr_raw:+.2f}) d={wr_raw - wl_raw:+.2f}"
                    f" -> yaw-loop ({wl:+.2f},{wr:+.2f}) -> goal {d:.2f} m"
                )
        cmd = condition_command(
            wl,
            wr,
            self._prev_cmd,
            max_omega=self.plan_max_omega,
            max_slew=self.plan_max_slew,
            max_decel=self.plan_max_decel,
            dt=self._command_dt(dynamics.DT),
            turn_boost=turn_boost,
            goal_dist=d,
            brake_dist=self.plan_goal_brake_dist,
            turn_brake_a_max=self.plan_turn_brake_a_max,
            lat_gain=self._lat_gain,
            turn_brake_scale=self._turn_brake_lookahead(turn_boost),
        )
        self._prev_cmd = cmd
        self._publish_cmd(cmd)
        if ref_cmd is not None:
            self._yaw_track_update(ref_cmd, cmd, turn_boost)
        self.pub_turn_boost.publish(
            Float32(data=float(turn_boost))
        )  # turn_boost in effect (debug/monitor)
        # ONLINE ADAPTATION (optional): pair the PREVIOUS command's differential with the yaw it
        # produced (this frame's gyro) and slow-update -- only while genuinely turning.
        # _turn_adapt compensates at the COMMAND (boost); _mu_est recenters the MODEL (both
        # directions) and sizes the robust-mu band from its residual noise.
        if (self._turn_adapt is not None or self._mu_est is not None) and self._imu_buffer:
            yaw_meas = float(self._imu_buffer[-1][2][2])
            if self._last_diff_out is not None:
                if self._turn_adapt is not None:
                    self._turn_adapt_update(self._last_diff_out, yaw_meas)
                if self._mu_est is not None:
                    center, span = self._mu_est.update(self._last_diff_out, yaw_meas)
                    self.planner.set_mu_band(center, span)
            self._last_diff_out = float(
                cmd[2] - cmd[0]
            )  # condition_command [L, rear, R] -> (wR - wL)

    def _conditioned(self, wl: float, wr: float, goal_dist: float, turn_boost: float):
        """Run the output conditioner without publishing. Pure, so the yaw loop can call it on the
        uncorrected plan to get its reference, and the caller then runs it again with the
        correction folded in."""
        return condition_command(
            wl,
            wr,
            self._prev_cmd,
            max_omega=self.plan_max_omega,
            max_slew=self.plan_max_slew,
            max_decel=self.plan_max_decel,
            dt=self._command_dt(dynamics.DT),
            turn_boost=turn_boost,
            goal_dist=goal_dist,
            brake_dist=self.plan_goal_brake_dist,
            turn_brake_a_max=self.plan_turn_brake_a_max,
            lat_gain=self._lat_gain,
            turn_brake_scale=self._turn_brake_lookahead(turn_boost),
        )

    def _yaw_track_update(self, ref_cmd: np.ndarray, cmd: np.ndarray, turn_boost: float) -> None:
        """Close the yaw loop: reference from the uncorrected intent, saturation from what went out.

        Both are post-turn-brake, so the loop tracks the braked arc rather than fighting the brake
        to recover the yaw the brake deliberately removed.
        """
        assert self._yaw_track is not None
        if not self._imu_buffer:
            return
        self._yaw_track.set_gains(  # live-tunable: these are not in _PLAN_BUILD
            self.plan_yaw_track_kp,
            self.plan_yaw_track_ki,
            self.plan_yaw_track_deadband,
            self.plan_yaw_track_max,
        )
        t_imu, _, w_base = self._imu_buffer[-1]
        now = float(self.get_clock().now().nanoseconds) * 1e-9
        dt = dynamics.DT
        if self._last_yaw_track_time is not None:
            dt = max(now - self._last_yaw_track_time, 1e-4)
        self._last_yaw_track_time = now
        # /cmd_joints order is (left, rear, right), so the differential is [2] - [0].
        #
        # DIVIDE OUT turn_boost. condition_command multiplies the differential by it, so a
        # reference read straight off its output scales with the boost -- and the boost then
        # cancels from the loop's error, since the measurement scales with it too. That makes the
        # loop blind to exactly the correction turn_boost was set to apply. It is worse than
        # useless at boost != 1: the reference would sit turn_boost x above the yaw the planner
        # asked for, so the loop would drive the robot to over-turn by that factor. The planner's
        # intent is the UNBOOSTED differential; the boost is compensation for a drivetrain loss,
        # not part of the plan.
        yaw_ref = float(ref_cmd[2] - ref_cmd[0]) / max(turn_boost, 1e-3) * self._yaw_per_diff
        yaw_meas = float(w_base[2])  # base-frame gyro z; /odin1/imu carries yaw on +z
        corr = self._yaw_track.update(
            yaw_ref,
            yaw_meas,
            dt,
            age=now - t_imu,
            saturated=bool(np.max(np.abs(cmd)) >= self.plan_max_omega - 1e-3),
        )
        self.pub_yaw_track.publish(Vector3(x=yaw_ref, y=yaw_meas, z=float(corr)))

    def _load_command_history(self) -> None:
        """Copy the commands still in flight into the rollout buffer, oldest first.

        Row k acts on rollout step k while k < command_delay_steps, so this is what makes the plan
        start from what the wheels are ABOUT to do rather than from what we are about to ask.
        Written in place into the buffer the captured MPPI graph already reads, so no re-capture.
        Before enough commands exist (startup) the oldest entry is repeated, i.e. the robot is
        assumed to have been holding it.
        """
        n = int(self.plan_sim.command_delay_steps)
        if n <= 0:
            return
        self.plan_sim.command_history.assign(
            in_flight_history(self._cmd_in_flight, n, int(self.plan_sim.batch_size))
        )

    def _command_dt(self, expected: float) -> float:
        """Seconds since the last /cmd_joints publish, for the rate limiter.

        `condition_command` sizes its slew and decel caps as rate * dt, so handing it a NOMINAL
        period that does not match the real publish interval scales every cap by the ratio. On
        Odin the cloud arrives every ~69 ms while dynamics.DT is 0.1, which made every limit ~45%
        looser than its parameter said -- plan_max_slew 2.0 was really acting as 2.9 rad/s^2.

        Clamped to [0.25x, 2x] of `expected`. A long gap is not a licence to jump: after a stalled
        frame the command should still ramp over the next few ticks rather than stepping by
        whatever the elapsed time would allow.
        """
        if self._last_cmd_time is None:
            return expected
        now = float(self.get_clock().now().nanoseconds) * 1e-9
        return float(np.clip(now - self._last_cmd_time, 0.25 * expected, 2.0 * expected))

    def _turn_adapt_update(self, diff_cmd: float, yaw_meas: float) -> None:
        """Feed the adaptive turn_boost, timed by the ACTUAL interval between updates.

        `tau_s` is a wall-clock time constant, so handing the EMA a nominal period that does not
        match the real update rate scales the constant by the ratio. This runs at the PLAN rate,
        ~69 ms on Odin rather than the nominal dynamics.DT of 0.1, which would otherwise make the
        loop ~31% faster than its parameter says.

        Clamped to [0.25x, 4x] of nominal. Unlike the command rate limiter, a long gap here is
        legitimately worth a bigger blend -- more time really has passed -- so the upper bound is
        loose and only guards against a stalled-frame outlier.
        """
        assert self._turn_adapt is not None
        now = float(self.get_clock().now().nanoseconds) * 1e-9
        dt = None
        if self._last_turn_adapt_time is not None:
            gap = now - self._last_turn_adapt_time
            dt = float(np.clip(gap, 0.25 * dynamics.DT, 4.0 * dynamics.DT))
        self._last_turn_adapt_time = now
        self._turn_adapt.update(diff_cmd, yaw_meas, dt=dt)

    def _publish_cmd(self, cmd: np.ndarray) -> None:
        """Publish the conditioned [left, rear, right] wheel command to /cmd_joints.

        `cmd` is in WHEEL rad/s (the planner/model convention) and the LLC now consumes /cmd_joints
        as wheel rad/s directly, so we publish it as-is. (Before 2026-07-27 the LLC misread the
        command as motor rev/s and we scaled by 22.5/(2*pi); that compensation was dropped once the
        LLC was fixed -- verified on the robot: realized wheel speed == commanded.)

        Stamped with the current clock (not the sensor stamp) so an LLC deadman sees a fresh
        command. VELOCITY ONLY: position/effort are left empty. Filling them with inf breaks
        serialization across the micro-ROS/XRCE bridge, so the LLC never receives the command
        (found live on the robot 2026-07-10)."""
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(JOINT_NAMES)
        m.velocity = [float(v) for v in cmd]
        self.pub_cmd.publish(m)
        self._cmd_in_flight.append(to_engine_order(cmd))
        self._last_cmd_time = float(self.get_clock().now().nanoseconds) * 1e-9

    def _publish_path(self, xy: np.ndarray, z: float, stamp) -> None:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.map_frame
        for x, y in xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)
        # Same path as a thick LINE_STRIP marker (nav_msgs/Path renders as 1px GL lines).
        m = Marker()
        m.header = path.header
        m.ns = "planned_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = float(self.plan_path_width)
        m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)  # magenta: reads over the green height map
        m.pose.orientation.w = 1.0
        m.points = [Point(x=float(x), y=float(y), z=z) for x, y in xy]
        self.pub_path_marker.publish(m)

    # ------------------------------------------------------------------
    # IMU gravity vector -> up-in-base
    # ------------------------------------------------------------------

    def _gyro_base_rotation(self, frame_id: str) -> np.ndarray | None:
        """Cached base_R_imu (rotation only) from the static IMU mount TF, or None if not ready yet.

        The gyro angular_velocity arrives in the IMU frame; rotating it into base_frame keeps the
        yaw axis correct whatever the mount. It is static, so it is looked up once and cached.
        """
        if self._base_R_gyro is not None:
            return self._base_R_gyro
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, rclpy.time.Time())
        except TransformException:
            return None
        r = tf.transform.rotation
        self._base_R_gyro = quaternion_to_matrix(r.x, r.y, r.z, r.w)[:3, :3]
        return self._base_R_gyro

    def _odom_to_matrix(self, odom_msg: Odometry) -> np.ndarray:
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        T = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        T[0, 3], T[1, 3], T[2, 3] = p.x, p.y, p.z
        return T

    def _scan_in_base(self, cloud_msg: PointCloud2) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warning(f"sensor->base TF lookup failed: {exc}")
            return None
        t = transform.transform.translation
        r = transform.transform.rotation
        base_T_sensor = quaternion_to_matrix(r.x, r.y, r.z, r.w)
        base_T_sensor[0, 3], base_T_sensor[1, 3], base_T_sensor[2, 3] = t.x, t.y, t.z
        points, _ = pointcloud2_to_xyz_time_array(cloud_msg)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32), base_T_sensor
        # Points stay in the SENSOR frame and in float32: the transform is fused into the
        # device gate kernel, so casting to float64 here would only double the upload.
        return np.ascontiguousarray(points, dtype=np.float32), base_T_sensor

    def _denoise(self, scan_wp: wp.array) -> wp.array:
        """GPU-native isolated-point removal on the base-frame scan (device in/out): strips
        specks with too few neighbours (see StatisticalOutlierFilter)."""
        if not self.outlier_enable or len(scan_wp) == 0:
            return scan_wp
        return self.outlier_filter.apply(scan_wp)

    def _scan_preproc(
        self, points_sensor: np.ndarray, base_T_sensor: np.ndarray
    ) -> tuple[wp.array, int]:
        """Sensor->base transform + z / self / range gates + compaction, on device.

        Returns `(buffer, count)`; the buffer is owned by the preprocessor and is valid only until
        the next call.
        """
        if self._preproc is None or self._preproc.max_points < points_sensor.shape[0]:
            self._preproc = ScanPreprocessor(int(points_sensor.shape[0]), device=self.device)
        buf, count = self._preproc.run(
            points_sensor,
            base_T_sensor,
            z_range=(self.z_crop_min, self.z_crop_max) if self.z_crop_enable else None,
            self_box=(
                (self.self_x_min, self.self_x_max, self.self_y_min, self.self_y_max)
                if self.self_filter_enable
                else None
            ),
            max_range=self.scan_max_range_m,
            min_range=self.scan_min_range_m,
        )
        return buf, count

    def _make_tf(self, mat: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
        """Marshal a 4x4 parent_T_child pose into a stamped TF message."""
        qx, qy, qz, qw = matrix_to_quaternion(mat)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(mat[0, 3])
        tf.transform.translation.y = float(mat[1, 3])
        tf.transform.translation.z = float(mat[2, 3])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        return tf

    def _cache_map_correction(self, world_T_base: np.ndarray, odom_msg: Odometry) -> None:
        """Cache the map->odom correction from a processed cloud; _odom_tf_callback broadcasts it."""
        self._map_T_odom = world_T_base @ invert_pose(self._odom_to_matrix(odom_msg))

    def _odom_tf_callback(self, odom_msg: Odometry) -> None:
        """Broadcast the pose TF at the full odom rate so base_link is dense and fresh.

        map->odom re-uses the last processed cloud's correction (it changes slowly between
        clouds); odom->base_link is this message's raw pose. Both are re-stamped at the odom
        time, so a lookup at any cloud stamp finds a bracketing sample instead of extrapolating.
        """
        stamp = odom_msg.header.stamp
        # Latest measured body twist (REP-105: Odometry.twist is child/base-frame; the odin
        # driver's twist-frame fix guarantees it) -> seeds the rollouts' init_twist in _plan.
        tw = odom_msg.twist.twist
        self._twist_meas = np.array([tw.linear.x, tw.linear.y, tw.angular.z], np.float32)
        self._twist_meas_t = self.get_clock().now().nanoseconds * 1e-9
        if self.publish_map_tf and self._map_T_odom is not None:
            self.tf_broadcaster.sendTransform(
                self._make_tf(self._map_T_odom, self.map_frame, odom_msg.header.frame_id, stamp)
            )
        # Close odom->base_link when the odom source broadcasts no TF (see publish_odom_tf).
        if self.publish_odom_tf:
            odom_T_base = self._odom_to_matrix(odom_msg)
            self.tf_broadcaster.sendTransform(
                self._make_tf(odom_T_base, odom_msg.header.frame_id, self.base_frame, stamp)
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
