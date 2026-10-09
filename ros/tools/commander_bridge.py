#!/usr/bin/env python3
"""Stand in for crl_commander so the Robotour mission stack drives navigation_node.

The Robotour follower (robot_mission_planner, `gps` mode) hands a route to crl_commander: the
waypoints on /goal_sequence (PoseArray, FP_ECEF), a single goal on /goal_waypoint, the mode over
/crl_commander/switch_mode, and it watches /crl_commander/state. crl_commander then plans with
naex and drives through the smac path follower. This node offers the same interface and instead
gives navigation_node one goal at a time on /goal_pose, in its map frame (odom_odin), converted
through TF: FP_ECEF -> FP_POI -> FP_VRTK -> base_link -> odin1_base_link -> odom_odin.

Mirrors crl_commander (adam_ws, 674dcac) where the follower can tell the difference:
  * state strings: the mode name, STOP / GOTO / SEQUENCE; SEQUENCE drops to STOP after the last
    waypoint, which the follower reads as "sequence finished".
  * a waypoint is reached inside a goal_reached_dist_x x goal_reached_dist_y box in the robot's
    frame (2.5 x 2.5 m); a sequence starts at the waypoint AFTER the nearest one; a waypoint is
    skipped after sequence_point_timeout_sec (180 s).
  * waypoints are re-converted into the map frame every tick: the earth -> map link drifts.
Differs from crl_commander on purpose:
  * earth-frame waypoints are lifted to the robot's ellipsoidal height before the conversion. The
    follower sends GPX points at altitude 0, ~450 m under the robot, and the Fixposition and the
    Odin disagree on tilt by ~1 deg, which moved every goal ~9 m (kolecko, 2026-09-30).
  * the nearest-waypoint search at the start leaves the last waypoint out: on a loop route the
    robot starts next to both ends, and "nearest is the last" ended the mission before it began.
Left out: crl_commander's skip-ahead past optional waypoints, the gpx sequence source, and the
explore / WEX / follow-me modes (refused).

Runs where crl_commander's interfaces are built (the NUC, `source ~/workspaces/adam_ws/install/
setup.bash`). Without them it still runs, with no services -- for replays and tests.

    python3 commander_bridge.py [--ros-args -p map_frame:=odom_odin ...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def reached(
    robot_xyyaw: tuple[float, float, float],
    goal_xy: tuple[float, float],
    box_x: float,
    box_y: float,
) -> bool:
    """crl_commander's arrival test: the goal inside a box around the robot, in the robot's frame
    (x ahead/behind, y left/right)."""
    x, y, yaw = robot_xyyaw
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    ahead = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return abs(ahead) <= box_x and abs(left) <= box_y


WGS84_A = 6378137.0  # [m]
WGS84_E2 = 6.69437999014e-3  # first eccentricity squared


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """(lat [rad], lon [rad], ellipsoidal height [m]) of an ECEF point."""
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(6):  # converges to < 1 mm near the surface
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        h = r / math.cos(lat) - n
        lat = math.atan2(z, r * (1.0 - WGS84_E2 * n / (n + h)))
    return lat, lon, h


def geodetic_to_ecef(lat: float, lon: float, h: float) -> tuple[float, float, float]:
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return (
        (n + h) * math.cos(lat) * math.cos(lon),
        (n + h) * math.cos(lat) * math.sin(lon),
        (n * (1.0 - WGS84_E2) + h) * math.sin(lat),
    )


def lift_to_height(
    point_ecef: tuple[float, float, float], height: float
) -> tuple[float, float, float]:
    """The same latitude and longitude at `height` [m] above the ellipsoid. Along the ellipsoid
    normal, not the geocentric radius: over ~450 m the two differ by up to ~1.5 m sideways."""
    lat, lon, _ = ecef_to_geodetic(*point_ecef)
    return geodetic_to_ecef(lat, lon, height)


def start_index(
    robot_xy: tuple[float, float], waypoints_xy: list[tuple[float, float]], from_next: bool
) -> int | None:
    """Where a sequence starts: the nearest waypoint, or the one after it; None = no waypoints.
    The last waypoint is left out of the nearest search, so a loop route (its ends side by side)
    starts at its beginning; a robot already at the end of an open route is sent to the last
    waypoint, reaches it at once and the sequence completes."""
    if not waypoints_xy:
        return None
    d = [math.hypot(wx - robot_xy[0], wy - robot_xy[1]) for wx, wy in waypoints_xy]
    candidates = range(max(1, len(d) - 1))
    i = min(candidates, key=d.__getitem__) + (1 if from_next else 0)
    return min(i, len(waypoints_xy) - 1)


@dataclass
class Walker:
    """Which waypoint of a sequence is active; the ROS node supplies positions in one frame."""

    box_x: float = 2.5
    box_y: float = 2.5
    timeout_s: float = 180.0  # <= 0 = never skip
    loop: bool = False
    index: int | None = None
    since: float = 0.0

    def begin(
        self,
        robot_xy: tuple[float, float],
        waypoints_xy: list[tuple[float, float]],
        from_next: bool,
        now: float,
    ) -> bool:
        self.index = start_index(robot_xy, waypoints_xy, from_next)
        self.since = now
        return self.index is not None

    def step(
        self,
        robot_xyyaw: tuple[float, float, float],
        waypoints_xy: list[tuple[float, float]],
        now: float,
    ) -> str:
        """Advance past a reached or timed-out waypoint. Returns 'active', 'advanced' or 'done'."""
        if self.index is None:
            return "done"
        hit = reached(robot_xyyaw, waypoints_xy[self.index], self.box_x, self.box_y)
        late = self.timeout_s > 0.0 and now - self.since > self.timeout_s
        if not (hit or late):
            return "active"
        self.index += 1
        self.since = now
        if self.index >= len(waypoints_xy):
            if not self.loop:
                self.index = None
                return "done"
            self.index = 0
        return "advanced"


def main() -> None:
    import rclpy
    from geometry_msgs.msg import PoseArray
    from geometry_msgs.msg import PoseStamped
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import QoSProfile
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_geometry_msgs import do_transform_pose_stamped
    from tf2_ros import Buffer
    from tf2_ros import TransformListener

    try:
        from crl_commander.srv import ConfigureSequenceMode
        from crl_commander.srv import SwitchMode
    except ImportError:
        ConfigureSequenceMode = SwitchMode = None  # noqa: N806

    class CommanderBridge(Node):
        def __init__(self) -> None:
            super().__init__("crl_commander")  # the name the follower's services live under
            p = self.declare_parameter
            self.map_frame = p("map_frame", "odom_odin").value
            self.robot_frame = p("robot_frame", "base_link").value
            # waypoints in this frame are lifted to the robot's height (see the module docstring)
            self.earth_frame = p("earth_frame", "FP_ECEF").value
            self.goal_topic = p("goal_topic", "/goal_pose").value
            self.walker = Walker(
                box_x=p("goal_reached_dist_x", 2.5).value,
                box_y=p("goal_reached_dist_y", 2.5).value,
                timeout_s=p("sequence_point_timeout_sec", 180.0).value,
                loop=p("sequence_loop", False).value,
            )
            self.from_next = p("sequence_start_from_next", True).value
            # [m] re-send the active goal once its converted position moved this far: every new
            # /goal_pose restarts navigation_node's plan, so small GNSS/Odin drift must not
            self.resend_m = p("goal_resend_m", 1.0).value
            # replays without the follower: take the mode a message implies
            self.auto_mode = p("auto_mode", SwitchMode is None).value

            self.tf = Buffer(cache_time=Duration(seconds=30.0))
            TransformListener(self.tf, self)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(PoseStamped, "/goal_waypoint", self.on_waypoint, latched)
            self.create_subscription(PoseArray, "/goal_sequence", self.on_sequence, latched)
            self.pub_goal = self.create_publisher(PoseStamped, self.goal_topic, 10)
            self.pub_state = self.create_publisher(String, "~/state", 10)
            if SwitchMode is not None:
                self.create_service(SwitchMode, "~/switch_mode", self.on_switch)
                self.create_service(
                    ConfigureSequenceMode, "~/configure_sequence_mode", self.on_configure
                )
            else:
                self.get_logger().warning("crl_commander interfaces not found: no services")

            self.mode = "STOP"
            self.goto_goal: PoseStamped | None = None
            self.sequence: list[PoseStamped] = []
            self.seq_fresh = False
            self.sent_xy: tuple[float, float] | None = None
            self.sent_key = None
            self.create_timer(0.2, self.tick)
            self.create_timer(1.0, self.publish_state)
            self.get_logger().info(
                f"commander bridge: goals -> {self.goal_topic} in {self.map_frame}"
            )

        # ---------------------------------------------------------------- inputs
        def on_waypoint(self, msg: PoseStamped) -> None:
            self.goto_goal = msg
            self.sent_key = None
            if self.auto_mode:
                self.set_mode("GOTO")

        def on_sequence(self, msg: PoseArray) -> None:
            self.sequence = []
            for pose in msg.poses:
                ps = PoseStamped()
                ps.header.frame_id = msg.header.frame_id
                ps.pose = pose
                self.sequence.append(ps)
            self.seq_fresh = True
            self.get_logger().info(
                f"sequence of {len(self.sequence)} waypoints in '{msg.header.frame_id}'"
            )
            if self.auto_mode:
                self.set_mode("SEQUENCE")

        def on_switch(self, req: object, res: object) -> object:
            mode = req.mode.lower()
            if mode in ("stop", "goto", "sequence", "force_goto"):
                self.set_mode("GOTO" if mode == "force_goto" else mode.upper())
                res.success, res.message = True, f"mode {self.mode}"
            else:
                res.success, res.message = False, f"mode '{req.mode}' not supported by the bridge"
                self.get_logger().warning(res.message)
            return res

        def on_configure(self, req: object, res: object) -> object:
            if req.source not in ("topic", ""):
                res.success, res.message = False, "only the topic sequence source is supported"
                return res
            self.walker.loop = bool(req.loop)
            res.success, res.message = True, f"source topic, loop {req.loop}"
            return res

        # ---------------------------------------------------------------- helpers
        def set_mode(self, mode: str) -> None:
            if mode == self.mode and mode != "SEQUENCE":
                return
            self.get_logger().info(f"mode {self.mode} -> {mode}")
            self.mode = mode
            self.sent_key = None
            if mode == "SEQUENCE":
                self.seq_fresh = True
            if mode == "STOP":
                self.stop_robot()
            self.publish_state()

        def publish_state(self) -> None:
            self.pub_state.publish(String(data=self.mode))

        def robot_height(self) -> float | None:
            """The robot's ellipsoidal height [m], from earth_frame -> robot_frame."""
            try:
                t = self.tf.lookup_transform(self.earth_frame, self.robot_frame, Time())
            except Exception as e:
                self.get_logger().warning(
                    f"no TF {self.robot_frame} -> {self.earth_frame}: {e}",
                    throttle_duration_sec=5.0,
                )
                return None
            p = t.transform.translation
            return ecef_to_geodetic(p.x, p.y, p.z)[2]

        def to_map(self, ps: PoseStamped, height: float | None) -> tuple[float, float] | None:
            if ps.header.frame_id == self.earth_frame:
                if height is None:
                    return None  # unlifted, the goal would land ~9 m off
                lifted = PoseStamped()
                lifted.header = ps.header
                lifted.pose.orientation = ps.pose.orientation
                q = ps.pose.position
                x, y, z = lift_to_height((q.x, q.y, q.z), height)
                lifted.pose.position.x, lifted.pose.position.y, lifted.pose.position.z = x, y, z
                ps = lifted
            try:
                t = self.tf.lookup_transform(self.map_frame, ps.header.frame_id, Time())
            except Exception as e:  # TF not there yet: wait, as the commander does
                self.get_logger().warning(
                    f"no TF {ps.header.frame_id} -> {self.map_frame}: {e}",
                    throttle_duration_sec=5.0,
                )
                return None
            out = do_transform_pose_stamped(ps, t)
            return (out.pose.position.x, out.pose.position.y)

        def robot(self) -> tuple[float, float, float] | None:
            try:
                t = self.tf.lookup_transform(self.map_frame, self.robot_frame, Time())
            except Exception:
                return None
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return (t.transform.translation.x, t.transform.translation.y, yaw)

        def send(self, key: tuple, xy: tuple[float, float]) -> None:
            moved = (
                self.sent_xy is None
                or math.hypot(xy[0] - self.sent_xy[0], xy[1] - self.sent_xy[1]) > self.resend_m
            )
            if key == self.sent_key and not moved:
                return
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.map_frame
            msg.pose.position.x, msg.pose.position.y = float(xy[0]), float(xy[1])
            msg.pose.orientation.w = 1.0
            self.pub_goal.publish(msg)
            self.sent_key, self.sent_xy = key, xy
            self.get_logger().info(f"goal {key} -> ({xy[0]:.2f}, {xy[1]:.2f}) in {self.map_frame}")

        def stop_robot(self) -> None:
            """navigation_node has no stop command: a goal where it stands is reached at once."""
            r = self.robot()
            if r is not None:
                self.send(("stop",), (r[0], r[1]))

        # ---------------------------------------------------------------- loop
        def tick(self) -> None:
            if self.mode == "STOP":
                return
            r = self.robot()
            if r is None:
                self.get_logger().warning(
                    f"no TF {self.robot_frame} -> {self.map_frame}: cannot place the robot",
                    throttle_duration_sec=5.0,
                )
                return
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.mode == "GOTO":
                if self.goto_goal is None:
                    return
                xy = self.to_map(self.goto_goal, self.robot_height())
                if xy is not None:
                    self.send(("goto", id(self.goto_goal)), xy)
                return
            # SEQUENCE: convert every waypoint each tick -- the earth -> map link drifts
            height = self.robot_height()
            pts = [self.to_map(w, height) for w in self.sequence]
            if not pts or any(p is None for p in pts):
                return
            if self.seq_fresh:
                self.seq_fresh = False
                if not self.walker.begin(r[:2], pts, self.from_next, now):
                    self.get_logger().info("empty sequence: nothing to do")
                    self.set_mode("STOP")
                    return
                self.get_logger().info(
                    f"sequence starts at waypoint {self.walker.index}/{len(pts) - 1}"
                )
            state = self.walker.step(r, pts, now)
            if state == "done":
                self.get_logger().info("sequence complete")
                self.set_mode("STOP")
                return
            if state == "advanced":
                self.get_logger().info(f"waypoint {self.walker.index}/{len(pts) - 1}")
            self.send(("seq", self.walker.index), pts[self.walker.index])

    rclpy.init()
    node = CommanderBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
