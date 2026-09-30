#!/bin/bash
# Open-loop /cmd_vel test: drive forward, turn around in place, drive back -- recorded, so the
# LLC's own yaw loop can be judged on the SAME command every run.
#
# Usage:  ./turn_test.sh [options] <name>       record to ~/bags/<name>_1, _2, ... (next free)
#         ./turn_test.sh -n <name>              dry run: print the programme, publish nothing
#
#   -v SPEED   forward speed [m/s]                    (default 0.5)
#   -w RATE    yaw rate of the turn [rad/s]           (default 0.6)
#   -d DIST    each straight [m]                      (default 3.0)
#   -a ACC     forward acceleration [m/s^2]           (default 0.5)
#   -A ACC     yaw acceleration [rad/s^2]             (default 1.0)
#   -r         turn right (default left)
#   -n         dry run
#
# The programme: straight DIST, stop 1 s, turn 180 deg in place, stop 1 s, straight DIST back.
# Every segment is a trapezoid (ramp at the given acceleration, hold, ramp down), so nothing
# steps; the command is open loop and identical every run, which is the point.
#
# SAFETY. navigation_node must NOT be running: it publishes /cmd_vel too and the two would fight.
# The script refuses to start while anything else publishes /cmd_vel. It counts down 3 s before
# moving, and sends zeros on any exit -- Ctrl-C included. Needs ~4 m of clear ground ahead.
set -euo pipefail

SPEED=0.5; RATE=0.6; DIST=3.0; ACC=0.5; YAW_ACC=1.0; SIDE=1; DRY=0
while getopts "v:w:d:a:A:rnh" opt; do
  case $opt in
    v) SPEED=$OPTARG ;; w) RATE=$OPTARG ;; d) DIST=$OPTARG ;;
    a) ACC=$OPTARG ;; A) YAW_ACC=$OPTARG ;; r) SIDE=-1 ;; n) DRY=1 ;;
    *) sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
  esac
done
shift $((OPTIND - 1))
NAME="${1:-}"
[[ -n "$NAME" || $DRY -eq 1 ]] || { echo "usage: turn_test.sh [options] <name>   (-h for options)"; exit 1; }

source ~/.rosrc >/dev/null 2>&1 || true
source ~/workspaces/helhest_ws/install/setup.bash >/dev/null 2>&1 || true

# The command programme and its publisher. Kept in one file with the recording so a run is one
# command on the robot; the profile itself is plain Python below.
drive() {
  python3 - "$SPEED" "$RATE" "$DIST" "$ACC" "$YAW_ACC" "$SIDE" "$1" <<'PY'
import math
import sys
import time

speed, rate, dist, acc, yaw_acc, side, go = map(float, sys.argv[1:8])
HZ = 20.0  # command rate: an LLC deadman always sees a fresh command


def trapezoid(total: float, peak: float, ramp: float) -> list[float]:
    """Samples at HZ of a profile that covers `total` (distance or angle) at most at `peak`,
    ramping at `ramp`; a triangle when the segment is too short to reach the peak."""
    peak = min(peak, math.sqrt(total * ramp))
    t_ramp = peak / ramp
    t_hold = (total - peak * t_ramp) / peak
    n = int(round((2 * t_ramp + t_hold) * HZ))
    out = []
    for i in range(n + 1):
        t = i / HZ
        out.append(min(peak, ramp * t, ramp * max(0.0, 2 * t_ramp + t_hold - t)))
    return out


def pause(seconds: float) -> list[tuple[float, float]]:
    return [(0.0, 0.0)] * int(round(seconds * HZ))


straight = [(v, 0.0) for v in trapezoid(dist, speed, acc)]
turn = [(0.0, side * w) for w in trapezoid(math.pi, rate, yaw_acc)]
prog = pause(1.0) + straight + pause(1.0) + turn + pause(1.0) + straight + pause(1.0)

dur = len(prog) / HZ
side_name = "left" if side > 0 else "right"
print(f"programme: {dist:.1f} m at {speed:.2f} m/s (acc {acc:.2f}), turn 180 deg {side_name} at "
      f"{rate:.2f} rad/s (acc {yaw_acc:.2f}), {dist:.1f} m back -- {dur:.1f} s")
print(f"  distance per straight {sum(v for v, _ in straight) / HZ:.2f} m, "
      f"turn {math.degrees(sum(abs(w) for _, w in turn) / HZ):.0f} deg")
if not go:
    sys.exit(0)

import rclpy  # noqa: E402  (only needed when driving)
from geometry_msgs.msg import TwistStamped  # noqa: E402

rclpy.init()
node = rclpy.create_node("turn_test")
pub = node.create_publisher(TwistStamped, "/cmd_vel", 10)


def send(v: float, w: float) -> None:
    m = TwistStamped()
    m.header.stamp = node.get_clock().now().to_msg()
    m.header.frame_id = "base_link"
    m.twist.linear.x = float(v)
    m.twist.angular.z = float(w)
    pub.publish(m)


try:
    for i in range(3, 0, -1):
        print(f"  moving in {i} ...", flush=True)
        for _ in range(int(HZ)):
            send(0.0, 0.0)
            time.sleep(1.0 / HZ)
    t0 = time.monotonic()
    for k, (v, w) in enumerate(prog):
        send(v, w)
        time.sleep(max(0.0, t0 + (k + 1) / HZ - time.monotonic()))
    print("  done")
finally:
    for _ in range(int(HZ)):  # a second of zeros, whatever happened above
        send(0.0, 0.0)
        time.sleep(1.0 / HZ)
    node.destroy_node()
    rclpy.shutdown()
PY
}

if [[ $DRY -eq 1 ]]; then
  drive 0
  echo "dry run -- nothing published, nothing recorded."
  exit 0
fi

# Nothing else may publish /cmd_vel (navigation_node with cmd_output twist does).
PUBS=$(ros2 topic info /cmd_vel 2>/dev/null | awk '/Publisher count/ {print $3}')
if [[ "${PUBS:-0}" != "0" ]]; then
  echo "something already publishes /cmd_vel ($PUBS publisher(s)) -- stop navigation_node first."
  exit 1
fi

N=1
while [[ -e "$HOME/bags/${NAME}_$N" ]]; do N=$((N + 1)); done
DEST="$HOME/bags/${NAME}_$N"
mkdir -p "$HOME/bags"

TOPICS=(
  /cmd_vel                  # what this script sent
  /joint_setpoints          # what the LLC turned it into, per wheel
  /joint_states             # what the wheels did: velocity and effort
  /debug/angle /debug/setpoint /debug/error   # the LLC's yaw loop, as far as it publishes it
  /estop_active
  /odin1/imu                # 400 Hz gyro: the yaw rate actually achieved
  /odin1/odometry           # Odin pose: where the robot actually went
  /odom_2d                  # wheel odometry
  /tf /tf_static
)
QOS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)/rosbag2_qos.yaml"

echo "recording -> $DEST"
ros2 bag record -o "$DEST" -s mcap --storage-preset-profile zstd_fast \
  --qos-profile-overrides-path "$QOS" --topics "${TOPICS[@]}" >/tmp/turn_test_record.log 2>&1 &
REC=$!
stop_recording() {
  kill -INT "$REC" 2>/dev/null || true
  wait "$REC" 2>/dev/null || true
  echo "bag: $DEST"
}
trap stop_recording EXIT
sleep 2  # let the recorder subscribe before anything moves

drive 1
sleep 1  # the tail of the stop
