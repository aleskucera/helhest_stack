# Incident Report — Half the Planner's Headings Were Unreachable

| | |
|---|---|
| **Date** | 2026-09-22 |
| **Component** | `helhest.planning.costtogo` (arc step selection) + `terrain_value_field.control_set` |
| **Reported by** | Operator, watching the `pocket` replay: *"it seems fine in the pocket. It just doesn't seem to turn to the goal when it can."* |
| **Severity** | High — silent. 1 of 6 stress worlds unreachable; the other 5 paying a 9–24% detour tax without failing. |
| **Root cause** | The arc step was chosen for *closure* and never checked for *connectivity*. At `bins=4`, `n_theta=16` the motion primitives turn by `{2, 4}` bins, which generates only the even headings — so half of every cell's heading bins were unreachable by any sequence of legal moves. |
| **Fixed in** | helhest `4dae1be`, tvf `07ac277`, verified `c1f50d9` |
| **Regression guard** | `tests/planning/test_heading_connectivity.py` |

---

## 1. TL;DR

The cost-to-go planner's state is `(row, col, heading bin)`. Its moves are arcs of a fixed length,
and the length is derived so that each arc lands exactly on a heading bin — otherwise the table
records a pose the robot never occupies. That derivation searched for the smallest *closing* step
that cleared a couple of cells.

**Closing is necessary and not sufficient.** The arcs turn by `{0, ±bins/2, ±bins}` bins, so
repeated they reach only the multiples of `bins/2`. That is every heading **iff
`gcd(bins/2, n_theta) == 1`**. Point turns are the only `±1` move and this planner prices them
out, so nothing bridged the gap.

The shipped configuration — `min_turn_radius = 0.5`, `n_theta = 16`, `cell = 0.2` — put the
`bins=2` step at **0.3927 m** against a **0.40 m** guard. It missed by **2%**, escalated to
`bins=4`, and the heading ring silently split in two.

The consequence was not a crash. It was a value field in which the odd heading bins held the
"no route" cap (`90.0 m`) with `blocked` reading **0** — not vetoed, never visited. On `pocket`
the bearing to the goal fell on bin 9, so the field declared the one heading pointing at the goal
a dead end, and the controller drove past a clear 4.5 m run at it for sixty consecutive frames.

---

## 2. Impact

| world | before (frames to goal) | after | |
|---|---|---|---|
| gap | 187 | **170** | −9% |
| slalom | 376 | **322** | −14% |
| pillars | 246 | **224** | −9% |
| pocket | **did not reach** (3.46 m short, 700 frames) | **308** | fixed |
| ridge | 262 | **199** | −24% |
| bumpy | 313 | **257** | −18% |

5/6 → 6/6. The second column is the more important finding: **every world that already reached
got there 9–24% sooner.** The split ring was not a `pocket` defect that happened to surface
there — it was taxing every run. `pocket` was simply the only world that *needed* a heading the
orphaned half of the ring owned, so it was the only one that failed outright rather than quietly
paying.

Measured across all six worlds before the fix, the odd heading bins had a finite value roughly
**half as often** as the even ones:

| world | even bins with a route | odd bins |
|---|---|---|
| gap | 70.8% | 37.1% |
| slalom | 56.7% | 31.4% |
| pillars | 47.7% | 25.5% |
| pocket | 26.9% | 14.3% |
| ridge | 66.0% | 34.1% |
| bumpy | 40.2% | 20.1% |

---

## 3. Background — what the lattice is

The planner does not work in continuous space. It discretises to **states**, and a state is a
place *plus a heading*:

```
state = (row, col, heading bin)
```

For these runs: a 50×50 grid of 0.2 m cells and `n_theta = 16` bins, so one bin is 22.5°, and bin
`t` means exactly `t × 22.5°`. That is 40,000 states, and value iteration assigns each one a
number V = "metres of driving from here to the goal".

From each state the robot has five moves — arcs of length `step` driven at five turn rates:

```python
_TURN_FRACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)   # of the sharpest turn available
```

The sharpest turn is set by the minimum turn radius R: an arc of length `step` at radius R turns
by `step / R`. **The arc must end on a heading bin**, or the table records a heading the robot
never reaches — off by up to half a bin on every move, compounding, with feasibility then
evaluated at a pose the robot will not occupy. So `step` is derived:

```
step = R × (2π / n_theta) × bins          # closing_step()
```

which makes the sharpest turn exactly `bins` bins. `bins` is forced **even**, because the
half-rate turn is `bins/2` and that has to land on a bin too.

---

## 4. Symptom

On the `pocket` world the robot orbited its own goal at a radius of 4–6 m for the full 700-frame
budget, ending 3.46 m short. The trajectory looked *reasonable* — it was not stuck, not
oscillating, not climbing anything. It simply never turned in.

The operator's reading was exactly right, and better than the first three hypotheses tested
against it.

---

## 5. Investigation

### 5.1 The planner was cleared first

For frames 416–472 — sixty consecutive frames — from the robot's **actual pose and heading**:

- the straight line to the goal was completely clear (`max blocked fraction 0.00`, ground flat to
  0.07 m) over 4.3–5.2 m;
- `blocked` at the robot's own cell and heading was `0`;
- `V` at the robot's own cell and heading was `≈ 5.2` — essentially the straight-line distance;
- **`trace_optimal` returned a 7–8 step plan ending at (9.10, −0.10): the goal itself.**

So the planner was handing back a correct, short, unobstructed plan, and the robot drove past it.

### 5.2 Three hypotheses measured away

Each of these was plausible, and each was wrong. Recorded so they are not re-investigated:

| hypothesis | test | result |
|---|---|---|
| The per-pose veto term is over-penalising the entrance | re-run with `--veto 0` | fails identically (2.73 m) — **not the veto** |
| MPPI is the problem; a pure-pursuit follower would reach | re-run with `--controller carrot` | fails *differently* — climbs the wall, body z to 1.10 m. Not a clean control, and not an exoneration |
| The value field is unstable frame to frame | track V at a **fixed world point** across 100 frames | stable: 4.40 → 4.53. **Not spatial instability** |
| The half-cell / grid-merge bugs (`518876a`, `e721469`) were to blame | re-run the sweep at HEAD | pocket essentially unchanged, 3.71 → 3.46 m |

### 5.3 The instrument that found it

The recorded diagnostic field was `hist_v = V.min(2)` — the minimum over *all* headings. That
hides a route which exists only at a heading the robot is not at, and it is why the defect
survived so long in the replay data.

Dumping the **whole heading vector** at one cell, instead of its minimum:

```
heading    0    22    45    67    90   112   135   157   180   202   225   247   270   292   315   337
V        6.9  90.0   7.1  90.0   6.3  90.0   5.2  90.0   5.2  90.0   5.1  90.0   5.2  90.0   5.9  90.0
blocked    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
```

Every odd bin at the cap, with `blocked` zero on every one of them. Nothing was vetoed. Value
iteration had simply never arrived.

`90.0` is not a coincidence — it is exactly `_vcap = 1.5 × (50 + 50) × 0.2 × (1 + 2.0)`, the
unreachable sentinel.

---

## 6. Root cause

### 6.1 The arithmetic

At `bins = 4`, write out what the five turn fractions do to the heading bin:

```
hard right    Δt = −4   (−90°)
gentle right  Δt = −2   (−45°)
straight      Δt =  0
gentle left   Δt = +2   (+45°)
hard left     Δt = +4   (+90°)
```

**Every one is even.** Start at bin 6 (135°):

```
step 1:  6 ± {0,2,4}      →  {2, 4, 6, 8, 10}
step 2:  those ± {0,2,4}  →  {0, 2, 4, 6, 8, 10, 12, 14}
step 3:  ... the same set. Closed.
```

Eight of sixteen headings, forever. Bins 1, 3, 5, … are not blocked by terrain and not vetoed by
tilt — they are **not connected by any sequence of legal moves**.

### 6.2 The general rule

"Even/odd" is the special case. The moves change the bin by multiples of `bins/2`, so the
reachable headings form the subgroup of ℤ_`n_theta` generated by `bins/2`, and the number of
disconnected pieces is:

```
components = gcd(bins/2, n_theta)
```

| | | |
|---|---|---|
| `bins=2` | `gcd(1,16) = 1` | connected — all 16 headings |
| `bins=4` | `gcd(2,16) = 2` | **two rings: evens and odds** ← shipped |
| `bins=6` | `gcd(3,16) = 1` | connected — even `bins`, and *fine*, because 3 ⟂ 16 |
| `bins=8` | `gcd(4,16) = 4` | four rings |

It was never about parity. `bins=6` is even and connected.

### 6.3 How that `bins` got chosen

```python
bins = 2
while closing_step(...) < 2.0 * cell_size and bins + 2 <= bins_max:
    bins += 2
```

`closing_step(16, 0.5, 2) = 0.3927 m`. The guard was `2 × 0.2 = 0.40 m`. **A 2% miss**, and the
search escalated into a split ring.

The one primitive that would have bridged it is the point turn (`±1` bin, rotating in place).
`terrain_value_field` includes those by default. `helhest` passes `pivot_cost = 0.0` → "infinite
price, leave them out". The safety net had been removed, and the step search never knew it had
been relying on it.

### 6.4 Why it destroyed `pocket` specifically

MPPI's horizon is 25 × (1/14.5 s) = 1.7 s ≈ **1.1 m of travel**, while the goal was **4.4 m**
away. It therefore cannot "see" the goal; it scores each rollout by the value field **at the
rollout's final pose, heading included**. Half of those final headings land on a `90.0`.

The bearing to the goal on the frames that mattered was 203°–214°, which nearest-bins to
**202.5° = bin 9 = odd = capped**. The heading that pointed at the goal was precisely the one the
field called hopeless. Measured, at the end of a straight rollout versus a turn-toward-goal
rollout:

```
frame 440:  straight 8.79   turn  4.64   → turn worth −4.15
frame 448:  straight 8.79   turn  4.53   → turn worth −4.27
frame 464:  straight 7.63   turn 90.00   → turn worth +82
frame 472:  straight 5.18   turn 90.00   → turn worth +85
```

Same manoeuvre, same open ground, advice inverting every few frames. Averaged over samples MPPI
commanded a wheel difference of **−0.2** — a token turn — and sailed past.

---

## 7. Why it stayed hidden

Three independent reasons, all worth fixing as a class:

1. **It degraded instead of failing.** The goal cell is seeded at *every* heading, so states on
   the orphaned ring still took a value wherever they could drive straight in. That is why odd
   bins measured 21% finite rather than 0% — the field got worse everywhere instead of breaking
   somewhere.
2. **The existing test checked the wrong invariant.** `test_closure.py` verified that every arc
   lands on a bin. It did, perfectly. Nothing checked that the arcs *reach* every bin.
3. **The recorded diagnostic was a minimum over headings.** `hist_v = V.min(2)` shows a cell as
   "has a route" if *any* of its 16 headings does. The defect is invisible in every replay, every
   figure, and every panel built from that field.

---

## 8. The fix

**`terrain_value_field` `07ac277` — refuse a disconnected table where it is built.**
`arc_control_set` already warned about a non-closing step. It now computes the subgroup the turn
deltas generate — derived from `turns`, so it survives a change to `_TURN_FRACTIONS` — and
**raises** when the point turns are off and `gcd(n_theta, *deltas) > 1`. It raises rather than
warns because the solver has no way to notice: this is a malformed table, not a tuning choice.

**`helhest` `4dae1be` — choose a connected `bins`, and derive the guard.**
The step search now only accepts `bins` with `gcd(bins/2, n_theta) == 1`, and raises if none fits
under the quarter-turn cap rather than falling back to a split one. The ways out at that point —
a coarser heading ring, or admitting the point turns the skid-steer actually has — are decisions,
not defaults.

The `2 × cell_size` guard was a round number. The bound that actually guarantees an arc leaves
its own state is the **half-diagonal**: a state sits at a cell centre, and the farthest in-cell
point is `cell × √2/2`, in the diagonal directions. The fix takes one whole cell — that, with
margin — which is the invariant `test_every_arc_moves_at_least_one_cell` already asserts.

This configuration now selects `bins=2`. Bin 9 at the frame in question reads **4.82 m** against
a true distance of 4.64 m, instead of 90.0.

**Cost:** 3.4 ms versus 2.9 ms per solve, on a frame where planning is already only ~7% of the
budget.

**What was deliberately *not* done:** fixing it by enabling point turns. It works — it is what
tvf's default relies on, and a skid-steer genuinely can rotate in place. But then the lattice's
connectedness depends on a *price*, and the next person to set `pivot_cost = 0.0` for a perfectly
sound reason (no pivot-then-drive on a slope) silently loses half the heading space again.
Connectivity is a structural invariant; it should not be a side effect of a cost knob.
`tests/planning/test_heading_connectivity.py` pins that point turns *would* reconnect a split
ring, precisely so the choice stays deliberate.

---

## 9. A second defect found by the same check

`tests/planning/test_closure.py` carried `(n_theta=24, cell=0.50)` in its config list, and it
passed.

That configuration has **no connected lattice at all** under the quarter-turn cap: `bins=2` turns
30° over 0.2618 m, which is half a cell and snaps back onto its own state; `bins=4` generates
every 2nd heading; `bins=6` every 3rd. Under the old search it selected `bins=6` and shipped a
lattice in **three disconnected pieces**, with two thirds of every cell's headings unreachable on
open ground — and the suite was satisfied, because closure was all anyone checked.

It now has its own test asserting the raise, and explaining why.

---

## 10. What to carry forward

1. **A structure that *closes* is not necessarily *connected*.** Any lattice built from repeating
   primitives has a reachability question, and it is answered by the subgroup its moves generate,
   not by whether each move is well-formed.
2. **Validate at the point of construction, not at the point of use.** The connectivity check
   belongs in `arc_control_set`, where every caller hits it, not in one caller's step search.
3. **Watch for "safe by default" that a caller can disable.** tvf's default includes the point
   turns and is always connected. The hazard existed only for a caller that turned them off, and
   nothing connected those two facts until it cost a run.
4. **Prefer derived bounds to round ones.** `2 × cell_size` was a plausible number that caused
   this. The requirement it was standing in for — an arc must leave its own cell — has an exact
   answer.
5. **Beware diagnostics that reduce over the axis where the bug lives.** `V.min(2)` made a
   heading-space defect structurally invisible in every recorded artifact. When a field is
   indexed by `(place, orientation)`, a figure that shows only `place` cannot show an
   orientation bug.
6. **The operator's description was the most accurate hypothesis on the table.** "It doesn't turn
   to the goal when it can" located the defect in the *turn*, while three measured hypotheses were
   looking at the veto, the controller, and the map.
