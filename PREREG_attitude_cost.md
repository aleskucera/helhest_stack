
## VERDICTS (held-out, single pass, 2026-08-31; artifacts
## studies/out/v2/ @spires; runner log therein)

Gate: 21 of 22 traverses kept; excluded 2023-07-20_19-12-27 (p99 |dz/dt|
0.62 m/s), as pre-stated. Unflagged windows: 317 foresight, 323 hindsight.
Freeze conditions C1/C1b/C2/C3 all held (C2 refit a=0.105, b=0.971 vs F1
a=0.110, b=1.021, within tolerance).

- V2-1 PASS both conditions: clark-corr's E error below fosm's in 301/317
  (p = 2.7e-69) and 311/323 (p = 2.7e-76). Median rel err E:
  1.85e-3 / 8.46e-4 (clark-corr) vs 2.31e-2 / 1.48e-2 (fosm).
- V2-2 PASS both conditions: clark-corr has the lowest median CVaR error
  outright (0.0045 foresight, 0.0022 hindsight; mc-32 0.0085 / 0.0068;
  step-form 0.352 / 0.377). No tie clause needed.
- V2-3 PASS both conditions: below mc-32 on both moments in 231/317
  (p = 1.8e-16) and 255/323 (p = 1.4e-26).
- Deficit correction, held-out: raw sd-ratio 0.939 / 0.965 -> corrected
  0.996 / 1.014. The design-fitted law transferred.
- Fan characterization: 284 / 289 scoreable fans; interpretation as
  pre-stated in section 6.
