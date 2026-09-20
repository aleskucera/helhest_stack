"""Walk a scripted 'person' across the robot's view and dump clouds + ground truth.

The point is a carve measurement with LABELS. On a bag you can see that a cell is tall and
guess why; here the mover's position is an input, so every cell it ever occupied is known
exactly, and so is every cell that is genuinely ground.

The person is an existing static box relocated each frame -- the ray-cast kernel recomputes
`X_wb * shape_transform[i]` per scan with no acceleration structure, so moving a shape is
visible to the sensor immediately and costs nothing. The box chosen is the static shape
FURTHEST from the robot, so removing it from its original place cannot confound the near field.

The other boxes are SUNK rather than switched off. `build_sim(solid_obstacles=False)` would
also take away the shape we borrow -- with no obstacles and no walls the only static shape left
is the TERRAIN MESH, so the script would walk the ground itself out from under the robot, which
then falls forever and scans nothing but its own hull. Building the world whole and dropping the
unwanted boxes 50 m keeps the terrain, and the pillars worlds stamp their obstacles ONTO flat
ground (`ground_only` clears those cells back to 0.0), so what is left behind is clear ground
and not a pit.
"""

import argparse
import numpy as np
import warp as wp
from examples.helhest_junior.odin_sim.sim import build_sim, ODIN_MOUNT_XYZ
from examples.helhest_junior.odin_sim.sensor import OdinSensor

ap = argparse.ArgumentParser()
ap.add_argument("--world", default="pillars")
ap.add_argument(
    "--obstacles",
    action="store_true",
    help="leave the world's pillars and walls standing. OFF by default: the person\n"
    "must walk over CLEAR ground or its ghost is indistinguishable from a wall",
)
ap.add_argument("--frames", type=int, default=260)
ap.add_argument("--rate", type=float, default=14.5)  # the real sensor's rate
ap.add_argument("--speed", type=float, default=1.0)  # [m/s] walking
ap.add_argument("--cross-x", type=float, default=3.0)  # [m] ahead of the robot
ap.add_argument("--from-y", type=float, default=-3.5)
ap.add_argument("--to-y", type=float, default=3.5)
ap.add_argument("--settle", type=int, default=40)
ap.add_argument(
    "--person-max-half",
    type=float,
    default=0.6,
    help="[m] largest half-extent a borrowed box may have to pass as a person",
)
ap.add_argument("--out", default="/local/kuceral4/tmp/person_clear.npz")
a = ap.parse_args()

dt = 1.0 / a.rate
# always build the world whole -- see the module docstring for why turning the obstacles off
# at BUILD time takes the terrain with them
sim = build_sim(world=a.world, dt=dt, viewer=False)
sensor = OdinSensor(sim.model, 0, ODIN_MOUNT_XYZ, seed=0)
m = sim.model
st0 = m.shape_transform.numpy().copy()
sb = m.shape_body.numpy()

for _ in range(a.settle):  # let the robot settle onto the terrain
    sim.step()
robot = sim.current_state.body_q.numpy()[0].copy()

# The first static shape is the terrain mesh (add_terrain runs before add_obstacles);
# borrowing it would teleport the ground away from under the robot. The rest are the world's
# obstacle boxes and its perimeter walls -- and a wall is 10 m long, so "furthest from the
# robot" alone picks one of those. Demand a person-sized footprint as well.
scale = m.shape_scale.numpy()
static = np.where(sb < 0)[0][1:]
compact = np.array([i for i in static if max(scale[i, 0], scale[i, 1]) <= a.person_max_half])
if not len(compact):
    sizes = "  ".join(f"{i}:{scale[i,0]:.2f}x{scale[i,1]:.2f}" for i in static)
    raise SystemExit(
        f"world {a.world!r} has no box within {a.person_max_half} m half-extent\n"
        f"  static shapes: {sizes}"
    )
d = np.hypot(st0[compact, 0] - robot[0], st0[compact, 1] - robot[1])
person = int(compact[int(np.argmax(d))])
hx, hy, hz = scale[person]  # the label must be the box's TRUE extent
print(f"world={a.world}  robot at ({robot[0]:.2f}, {robot[1]:.2f}, {robot[2]:.2f})")
print(
    f"using static shape {person} as the person; it started {d.max():.1f} m away, "
    f"half-extent {hx:.2f} x {hy:.2f} m, {2*hz:.2f} m tall"
)
if not a.obstacles:
    st0[[i for i in static if i != person], 2] -= 50.0  # sink, do not delete
    m.shape_transform.assign(st0)
    print(f"sank {len(static) - 1} other static shapes; the person walks over clear ground")

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
    person_hx=float(hx),
    person_hy=float(hy),
    person_h=float(2 * hz),
    world=a.world,
)
print(f"\nwrote {a.out}: {len(clouds)} frames, {n.sum()} points total")
