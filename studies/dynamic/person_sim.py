"""Walk a scripted 'person' across the robot's view and dump clouds + ground truth.

The point is a carve measurement with LABELS. On a bag you can see that a cell is tall and
guess why; here the mover's position is an input, so every cell it ever occupied is known
exactly, and so is every cell that is genuinely ground.

The person is an existing static box relocated each frame -- the ray-cast kernel recomputes
`X_wb * shape_transform[i]` per scan with no acceleration structure, so moving a shape is
visible to the sensor immediately and costs nothing. The box chosen is the static shape
FURTHEST from the robot, so removing it from its original place cannot confound the near field.
"""

import argparse
import numpy as np
import warp as wp
from examples.helhest_junior.odin_sim.sim import build_sim, ODIN_MOUNT_XYZ
from examples.helhest_junior.odin_sim.sensor import OdinSensor

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="pillars")
ap.add_argument("--frames", type=int, default=260)
ap.add_argument("--rate", type=float, default=14.5)  # the real sensor's rate
ap.add_argument("--speed", type=float, default=1.0)  # [m/s] walking
ap.add_argument("--cross-x", type=float, default=3.0)  # [m] ahead of the robot
ap.add_argument("--from-y", type=float, default=-3.5)
ap.add_argument("--to-y", type=float, default=3.5)
ap.add_argument("--settle", type=int, default=40)
ap.add_argument("--out", default="/local/kuceral4/tmp/person.npz")
a = ap.parse_args()

dt = 1.0 / a.rate
sim = build_sim(world=a.world, dt=dt, viewer=False)
sensor = OdinSensor(sim.model, 0, ODIN_MOUNT_XYZ, seed=0)
m = sim.model
st0 = m.shape_transform.numpy().copy()
sb = m.shape_body.numpy()

for _ in range(a.settle):  # let the robot settle onto the terrain
    sim.step()
robot = sim.current_state.body_q.numpy()[0].copy()

static = np.where(sb < 0)[0]
d = np.hypot(st0[static, 0] - robot[0], st0[static, 1] - robot[1])
person = int(static[int(np.argmax(d))])
print(f"world={a.world}  robot at ({robot[0]:.2f}, {robot[1]:.2f})")
print(f"using static shape {person} as the person; it started {d.max():.1f} m away")

span = a.to_y - a.from_y
clouds, robots, people, stamps = [], [], [], []
for k in range(a.frames):
    t = k * dt
    # walk across at constant speed, then hold at the far side
    y = a.from_y + min(span, a.speed * t)
    px, py = robot[0] + a.cross_x, robot[1] + y
    tf = st0.copy()
    tf[person, 0], tf[person, 1], tf[person, 2] = px, py, 0.85
    m.shape_transform.assign(tf)
    sim.step()
    pts = sensor.scan(sim.current_state)
    clouds.append(pts.astype(np.float32))
    robots.append(sim.current_state.body_q.numpy()[0].copy())
    people.append([px, py, 0.85])
    stamps.append(t)
    if k % 40 == 0:
        print(f"  frame {k:>4d}  t={t:5.2f}s  person at ({px:5.2f}, {py:6.2f})  {len(pts)} returns")

n = np.array([len(c) for c in clouds], np.int32)
np.savez_compressed(
    a.out,
    points=np.concatenate(clouds, 0),
    counts=n,
    robot=np.array(robots, np.float64),
    person=np.array(people, np.float64),
    stamp=np.array(stamps, np.float64),
    mount=np.array(ODIN_MOUNT_XYZ, np.float64),
    person_hx=0.2,
    person_hy=0.2,
    person_h=1.7,
    world=a.world,
)
print(f"\nwrote {a.out}: {len(clouds)} frames, {n.sum()} points total")
