"""The reference curves the robot published (/reference_path_marker), with the plan, the pose and
the goals, cut out of a bag into one npz per bag -- the raw material for judging them.

    uv run --no-project --with rosbags --with numpy python extract.py <out_dir> <bag> [<bag> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def yaw_of(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def extract(bag: Path) -> dict[str, np.ndarray]:
    topics = {
        "/reference_path_marker",
        "/planned_path",
        "/odin1/odometry",
        "/goal_pose",
        "/cmd_vel",
    }
    ref_t, ref_xy, plan_t, plan_xy = [], [], [], []
    odo, goals, cmd = [], [], []
    with AnyReader([bag]) as rd:
        conns = [c for c in rd.connections if c.topic in topics]
        for c, t, raw in rd.messages(connections=conns):
            m = rd.deserialize(raw, c.msgtype)
            ts = t * 1e-9
            if c.topic == "/reference_path_marker":
                ref_t.append(ts)
                pts = np.array([[p.x, p.y] for p in m.points], np.float32).reshape(-1, 2)
                ref_xy.append(pts if m.action == 0 else np.zeros((0, 2), np.float32))
            elif c.topic == "/planned_path":
                plan_t.append(ts)
                plan_xy.append(
                    np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses], np.float32)
                )
            elif c.topic == "/odin1/odometry":
                p, q, tw = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
                odo.append([ts, p.x, p.y, yaw_of(q), tw.linear.x, tw.angular.z])
            elif c.topic == "/goal_pose":
                goals.append([ts, m.pose.position.x, m.pose.position.y])
            elif c.topic == "/cmd_vel":
                cmd.append([ts, m.twist.linear.x, m.twist.angular.z])

    def pack(seq: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = np.array([len(a) for a in seq], np.int32)
        return n, (np.concatenate(seq) if seq else np.zeros((0, 2), np.float32))

    rn, rxy = pack(ref_xy)
    pn, pxy = pack(plan_xy)
    return dict(
        ref_t=np.array(ref_t),
        ref_n=rn,
        ref_xy=rxy,
        plan_t=np.array(plan_t),
        plan_n=pn,
        plan_xy=pxy,
        odom=np.array(odo),
        goals=np.array(goals).reshape(-1, 3),
        cmd=np.array(cmd).reshape(-1, 3),
    )


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for b in map(Path, sys.argv[2:]):
        d = extract(b)
        np.savez_compressed(out / f"{b.name}.npz", **d)
        print(
            f"{b.name}: {len(d['ref_t'])} refs ({int((d['ref_n'] < 3).sum())} empty), "
            f"{len(d['plan_t'])} plans, {len(d['odom'])} odom, {len(d['goals'])} goals"
        )


if __name__ == "__main__":
    main()
