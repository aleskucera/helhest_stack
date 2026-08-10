#!/usr/bin/env python3
"""Drive a scripted calibration manoeuvre, so the steps are clean and the holds are exact.

    ros2 run ... / python3 ros/calibrate_drive.py relax --go
    python3 ros/calibrate_drive.py relax            # dry run: print the program, publish nothing

`record_odin.sh` only RECORDS; the manoeuvre was a description for a human on the sticks. That is
good enough for steady-state turning, where all that matters is holding a command for a few
seconds, and not good enough for `relax`, whose whole measurement is a response TIME read off a
differential step. A hand-made step has an uncertain onset, an uncertain amplitude and a different
mean speed every repeat, which is exactly the noise the fit cannot absorb.

Start the bag first, then this. Two safety points that are not optional:

  * The planner must NOT be publishing at the same time. Set `plan_actuate:=false` (or do not run
    elevation_node); two publishers on /cmd_joints fight and neither manoeuvre is what you think.
  * Nothing publishes until `--go`. Ctrl-C, or any exit, sends zeros.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# [left, rear, right], wheel rad/s, matching helhest.control.command.JOINT_NAMES and the LLC's
# units since f056dcc. VELOCITY ONLY -- filling position/effort breaks serialisation across the
# micro-ROS bridge and the LLC silently never receives the command (found live, 2026-07-10).
JOINT_NAMES = ("left_wheel_j", "rear_wheel_j", "right_wheel_j")
RATE_HZ = 20.0  # command rate; well above the plan rate so an LLC deadman always sees a fresh one


def block(seconds: float, mean: float, diff: float) -> tuple[float, float, float]:
    """One held command: (duration, wL, wR) from a mean forward speed and a turn differential."""
    return (seconds, mean - 0.5 * diff, mean + 0.5 * diff)


def program(name: str) -> list[tuple[float, float, float]]:
    """The manoeuvre, as a list of held commands."""
    out: list[tuple[float, float, float]] = []
    if name == "calibrate":
        out.append(block(3.0, 0.0, 0.0))
        for mean in (1.0, 2.0, 3.0):  # forward gain: straight, held long enough to be steady
            out += [block(5.0, mean, 0.0), block(2.0, 0.0, 0.0)]
        for mean in (1.0, 2.0):  # steady-state turn gain, both directions
            for diff in (0.5, 1.0, 2.0):
                out += [block(5.0, mean, diff), block(2.0, mean, 0.0),
                        block(5.0, mean, -diff), block(2.0, mean, 0.0)]
            out.append(block(2.0, 0.0, 0.0))
        for _ in range(3):  # standing starts: the actuator step response, tau_motor
            out += [block(3.0, 2.0, 0.0), block(3.0, 0.0, 0.0)]
    elif name == "relax":
        # The discriminator. The SAME differential step from three forward speeds: a lag keyed to
        # time gives the same response time at every speed, one keyed to distance gives tau =
        # sigma / v and so scales as 1/v. Chrono prefers the distance form (sigma ~0.15 m); this
        # is what confirms or kills it on the robot. Mean speed is settled before each step so the
        # step is in the DIFFERENTIAL only.
        for mean in (0.5, 1.0, 2.0):
            out.append(block(3.0, mean, 0.0))
            for _ in range(3):
                out += [block(4.0, mean, 1.5), block(3.0, mean, 0.0),
                        block(4.0, mean, -1.5), block(3.0, mean, 0.0)]
            out.append(block(2.0, 0.0, 0.0))
    elif name == "compact":
        # `relax` needs a sports field: 41 x 29 m, measured by running it through the engine. This
        # gets the same measurement on a patch, by SPINNING IN PLACE. What sets a tyre's relaxation
        # is the speed its contact patch travels over the ground, R * mean|omega|, which is nonzero
        # in a spin even though the body does not translate -- so the speed sweep survives and the
        # footprint collapses to the robot's own turning circle. Contact speed spans 0.70-1.40 m/s
        # here. The sweep would like to go lower for a wider lever on the tau = sigma/v fit, but a
        # spin does not break static friction below ~2 rad/s on this drivetrain (found in the field
        # 2026-08-10: 0.5 and 1.0 rad/s did not move it, 2.0 did), so the low end is dropped.
        #
        # The caveat is real: a spin has every wheel skidding laterally, which is not the regime
        # the planner spends its time in, so sigma fitted here should be checked against a couple
        # of driving steps before it is trusted. `--pause` stops between blocks so a short driving
        # segment can be added by hand on whatever run-up the site allows.
        for w in (2.0, 3.0, 4.0):
            hold = max(2.0, 8.0 * 0.15 / (0.35 * w))  # ~8 relaxation lengths of contact travel
            out.append(block(2.0, 0.0, 0.0))
            for _ in range(3):
                out += [(hold, -w, w), block(2.0, 0.0, 0.0),
                        (hold, w, -w), block(2.0, 0.0, 0.0)]
    elif name == "slope":
        # Driven ON a slope of 10 deg or more; the operator points the robot, this holds the
        # command steady. Across-slope is the case that matters -- zero lateral load transfer in
        # the old model against 0.405 m g in Chrono.
        for _ in range(2):
            out += [block(5.0, 1.5, 0.0), block(3.0, 0.0, 0.0)]
        out += [block(5.0, 1.5, 1.0), block(3.0, 0.0, 0.0),
                block(5.0, 1.5, -1.0), block(3.0, 0.0, 0.0)]
    else:
        raise SystemExit(f"unknown program '{name}'")
    return out


def describe(blocks, max_omega: float) -> float:
    total = sum(b[0] for b in blocks)
    peak = max(max(abs(b[1]), abs(b[2])) for b in blocks)
    print(f"  {len(blocks)} held commands, {total:.0f} s total, "
          f"peak wheel |omega| {peak:.2f} rad/s")
    print(f"  peak ground speed ~{0.35 * peak:.2f} m/s; clamp is {max_omega:.2f} rad/s")
    if peak > max_omega:
        print(f"  NOTE: {peak:.2f} exceeds the clamp and will be limited to {max_omega:.2f}")
    return total


class Driver(Node):
    def __init__(self, blocks, max_omega: float):
        super().__init__("calibrate_drive")
        self.pub = self.create_publisher(JointState, "/cmd_joints", 10)
        self.blocks = blocks
        self.max_omega = max_omega

    def send(self, wl: float, wr: float) -> None:
        lim = self.max_omega
        wl, wr = max(-lim, min(lim, wl)), max(-lim, min(lim, wr))
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(JOINT_NAMES)
        m.velocity = [float(wl), float(0.5 * (wl + wr)), float(wr)]  # rear at the mean
        self.pub.publish(m)

    def run(self, pause: bool = False) -> None:
        period = 1.0 / RATE_HZ
        t_start = time.monotonic()
        for i, (dur, wl, wr) in enumerate(self.blocks):
            t_end = time.monotonic() + dur
            print(f"  [{time.monotonic() - t_start:6.1f}s] block {i + 1}/{len(self.blocks)}: "
                  f"wL {wl:+5.2f}  wR {wr:+5.2f}  ({dur:.1f} s)", flush=True)
            if pause and i > 0:
                self.stop()
                input("      [paused] reposition if needed, then press Enter...")
                t_end = time.monotonic() + dur
            while time.monotonic() < t_end:
                self.send(wl, wr)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)

    def stop(self) -> None:
        for _ in range(int(RATE_HZ)):  # half a second of explicit zeros, so the LLC cannot coast
            self.send(0.0, 0.0)
            time.sleep(1.0 / RATE_HZ)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("program", choices=("calibrate", "relax", "compact", "slope"))
    ap.add_argument("--go", action="store_true", help="actually publish (default is a dry run)")
    ap.add_argument("--max-omega", type=float, default=4.0, help="per-wheel clamp [rad/s]")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--pause", action="store_true",
                    help="wait for Enter between blocks, to reposition on a small patch")
    args = ap.parse_args()

    blocks = program(args.program)
    print(f"program '{args.program}':")
    total = describe(blocks, args.max_omega)
    if args.program == "relax":
        print("  MEASURED footprint ~41 x 29 m -- use `compact` unless you have that")
    if args.program == "compact":
        print("  spins in place: footprint is about the robot's own turning circle")
    if not args.go:
        print("\ndry run -- nothing published. re-run with --go to drive.")
        return

    print(f"\n  START THE BAG FIRST, and make sure plan_actuate is off.")
    for k in range(args.countdown, 0, -1):
        print(f"  driving in {k}...", flush=True)
        time.sleep(1.0)

    rclpy.init()
    node = Driver(blocks, args.max_omega)
    try:
        node.run(args.pause)
        print(f"  done, {total:.0f} s")
    except KeyboardInterrupt:
        print("\n  interrupted -- stopping", flush=True)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
