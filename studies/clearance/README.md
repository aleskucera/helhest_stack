# Clearance-law study harness

Closed-loop sim runs of the clearance law (`helhest/planning/clearance.py`) with drive_sim.

- `worlds.py` registers three study worlds (2.4 m and 2.6 m corridors started 0.5 m off centre,
  a 2.4 m L-bend) into `helhest.worlds` for the run. They are not part of the stress set.
- `run.py` is drive_sim plus the study worlds and `--set plan_key=value` overrides.
- `batch.sh jobs.txt out_dir` runs a job list 8 at a time, alternating GPUs.
- `analyze.py out_dir arm1 arm2 ...` compares arms: frames, closest pass, turning near walls.

A job list compares arms by tag, three runs each, for example:

```
slalom_base_r1 slalom
slalom_c20_r1 slalom --set plan_clear_c0=0.2
```

Compare arms only within one machine: dasenka's ostrich physics differs from the laptop's.
