# Field calibration — what to do with the robot outside

Branch: **`engine/exact-arc-integration`** (not `tier1-certificates` — see TIER1_REPORT.md §0).

Everything below exists to settle parameters that are currently fitted against a SIMULATOR
(Project Chrono) rather than measured on the robot. Chrono is an independent instrument, not
ground truth; these bags are the arbiter. Where a number below has a "target", that is what the
model currently predicts — the point of the trip is to find out whether it is right.

---

## 0. Thirty seconds that beat an hour of driving

**Look at the rear wheel before you do anything else.**

1. Does the hub **swivel** about a vertical axis, or is its axle rigidly parallel to the front pair?
2. If it swivels, roughly how much **trail** — the horizontal distance from the swivel axis down to
   where the tyre touches the ground?

Write the answer down. In Chrono this single fact moves the turn gain from α 1.10 (free caster) to
α 3.17 (fixed axle), and the robot's measured 2.20 sits between them — so it is currently the
largest single unknown in the yaw model, and no amount of driving resolves it faster than looking.
Take a photo of the mounting.

---

## 1. Before you leave

- [ ] Robot is on the branch above, or at least publishing `/joint_states`.
- [ ] Laptop can reach the robot (`bags/fetch_bag.sh` uses `robot@192.168.18.5`).
- [ ] Space in mind: see the footprint table in §3. `compact` needs almost nothing; `calibrate`
      driven manually needs room to hold a straight line for five seconds at a time.
- [ ] Tape measure or a phone, for the rear-wheel trail and the slope angle.

---

## 2. Pre-flight, on site — the check that saves the trip

**`/joint_states` must be live.** Every fit scores MEASURED wheel speeds against the gyro. With
only `/cmd_joints` you cannot separate what the drivetrain did from what the vehicle did, which is
exactly the ambiguity that left the turn gain reading 2.20 on measured wheels and 3.06 on
commanded ones.

```bash
ros2 topic hz /joint_states      # steady rate, and velocity non-zero when you turn a wheel by hand
ros2 topic hz /joint_setpoints
ros2 topic hz /imu/data          # or /ouster/imu -- the yaw ground truth
```

**`plan_actuate` must be OFF** for any scripted or manual drive:

```bash
# either don't run elevation_node at all, or
ros2 run ... elevation_node --ros-args -p plan_actuate:=false
```

Two publishers on `/cmd_joints` fight and the manoeuvre is not what you think it is. This is the
failure that silently ruins a bag — it looks fine until the fit makes no sense.

---

## 3. The recordings

Start the bag FIRST, then drive. `record_odin.sh <name>` records; Ctrl-C stops and finalises.

| # | scenario | footprint | time | driven how | settles |
|---|---|---|---|---|---|
| 1 | `compact` | **1.9 × 1.8 m** | 142 s | scripted | σ, the relaxation length |
| 2 | `calibrate` | 49 × 22 m scripted, or any long manual drive | ~3 min | **manual is fine** | α, forward gain, τ_motor |
| 3 | `slope` | 11 × 4 m on a ≥10° slope | 32 s | manual or scripted | the load-transfer fix |

Do them in that order: `compact` is safest and shakes out any plumbing problem before the rest.

### 3.1 `compact` — the one a human cannot do

```bash
./ros/record_odin.sh compact              # terminal 1
python3 ros/calibrate_drive.py compact    # terminal 2: DRY RUN first, prints the program
python3 ros/calibrate_drive.py compact --go
```

Spins in place at four wheel speeds. It measures whether the yaw lag is keyed to TIME or to
DISTANCE, which needs identical steps at different speeds — a hand-made step has an uncertain
onset, an uncertain amplitude and a different speed every repeat, and the fit cannot absorb that.

Spinning works because what sets a tyre's relaxation is the speed its CONTACT PATCH travels over
the ground, which is non-zero in a spin even though the body does not translate. Contact speed
spans 0.18–1.40 m/s here, a wider range than the driving version manages.

`--pause` waits for Enter between blocks if you need to reposition.

### 3.2 `calibrate` — drive it yourself

```bash
./ros/record_odin.sh calibrate            # then drive
```

Manual is fine. The fits read measured wheel speeds, so they do not care whether a human or a
script put the wheels there. **One discipline makes or breaks this bag:**

> **Hold each command still for 3–5 seconds.** Not continuous stick correction.

That is the entire reason the existing archive cannot fit α: the planner holds a turn for a median
of **3 ms**, so there is no steady state anywhere in it. A human deliberately holding still is good
data; a human constantly correcting produces the same useless soup.

Over the drive, get all of:

- **straight, held ~5 s**, at three clearly different speeds (slow / medium / fast)
- **steady arcs, held ~5 s**, both directions, at two speeds — both directions matters, it is what
  catches the left-wheel sign asymmetry this project has been bitten by before
- **a few standing starts** — from rest, push to a speed and hold. Gives τ_motor.
- **a few deliberately sharp turn-onsets at clearly different speeds** — snap the stick rather than
  easing it. This is the transfer check for σ: does the value fitted from spinning hold when
  actually driving? `/joint_setpoints` is recorded, so the real onset can be found and fitted
  against rather than assumed.

Gaps, stops, obstacles and traffic cost nothing — the fits segment on the holds.

### 3.3 `slope` — if you can find one

```bash
./ros/record_odin.sh slope
```

≥10° of tilt. Straight up, straight down, and **ACROSS in both directions**, plus a turn while on
the cross-slope. Hold each 4–5 s.

The across-slope runs are the ones that matter. Nothing in the entire bag archive exceeds 5.4° of
tilt, so the `normal_loads` load-transfer fix is validated against Chrono and against nothing else
— and it is exactly there that the old model predicted **zero** lateral transfer where Chrono
predicts 0.405 m g. Measure the slope angle with a phone and write it down.

---

## 4. Write these down per bag

μ changes every answer, so **one bag per surface** and label it.

- surface (concrete / gravel / grass / mud) and how wet
- slope angle if any
- anything odd: a wheel slipping visibly, a stall, the robot fighting you

---

## 5. Back at the desk

```bash
bags/fetch_bag.sh <name>                            # pull from the robot

./ros/calibrate_turn.sh fit ~/bags/<name>           # alpha -- standalone, no GPU or lidar needed
python scripts/fit_actuator_lag.py ~/bags/<name>    # tau_motor
python scripts/fit_traction.py    ~/bags/<name>     # L/K
python scripts/replay_traction.py ~/bags/<name>     # trajectory scoring against the gyro
```

What each should come out at, and what it decides:

| quantity | current value | where it came from | if the bag disagrees |
|---|---|---|---|
| α (turn gain) | 2.20 | bags, two IMUs to 1% | re-decide `k_turn`; the planner ships 0.6 (α 1.48) and under-turns ~1.5x |
| forward gain | 0.906–0.925 | bags, SLAM odometry | it is a real loss; the legacy model gives 1.000 by construction |
| τ_motor | 0.19 s | bags, 34 step responses | already solid; this is a re-confirmation |
| **σ (relaxation length)** | **0.15 m** | **fitted to CHRONO only** | **this is the trip's main question** |
| L/K | 8–15 | bags, quasi-static samples only | still conflicted with the forward gain |

**σ is the one to look at first.** Under the distance hypothesis the response time scales as
`τ = σ/v`, so across the four `compact` speeds it should vary by roughly 8×. If instead the
response time is about the same at every speed, the lag is time-keyed, σ is wrong, and
`yaw_relax_len` should be dropped in favour of `yaw_tau`.

`replay_traction.py` currently scores legacy / shear / shear+momentum. It should gain the two yaw-lag
variants so σ and τ get scored against the gyro rather than against Chrono — worth writing once
there is a real `compact` bag to write it against.

---

## 6. What ruins a bag

In rough order of how often it will happen:

1. **`plan_actuate` left on** — two publishers on `/cmd_joints`, manoeuvre is not what you drove.
2. **`/joint_states` not publishing** — no measured wheel speeds, nothing is fittable.
3. **Commands never held still** — no steady state, so no α. The single most common failure.
4. **Only one turn direction** — hides the drivetrain sign asymmetry.
5. **Mixed surfaces in one bag** — μ differs, the fit averages two different robots.
