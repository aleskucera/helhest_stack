"""Every pinned constant of the E2 pipeline, in one place, with its provenance.

These are the values PREREG_baseprod.md marks [PIN-AT-FREEZE] and the values FREEZE.json
records. Nothing in the scored path may introduce a constant that is not here.

The two calibration constants that were FITTED on design data -- the mount-pitch offset and
the stereo noise model -- are pinned to their rehearsal-v3 values rather than refitted at run
time. That is the freeze semantics: fitted once, on design data, applied unchanged. The runner
still refits both and records the refit beside the pinned value (`pitch_refit_check`,
`sigma_refit_check` in FREEZE.json) so a drift between code versions is visible; the refit
never enters the build.
"""
from __future__ import annotations

# --- depth intrinsics: CameraInfo, recovered in theory/BASEPROD_BLOCKER.md -------------------
FX = FY = 422.28125
CX = 425.4715881347656
CY = 236.69622802734375
IMG_W, IMG_H = 848, 480
DEPTH_SCALE = 1e-3          # 16-bit PNG is millimetres
DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)   # already-rectified stream

# --- camera mount: rehearsal_v3/fit_pitch.json, common to both design traverses ---------------
NOMINAL_PITCH_DEG = 20.0    # tf_static link_mast -> camera_bottom_screw_frame
MOUNT_PITCH_OFFSET_DEG = -1.4092
TOTAL_MOUNT_PITCH_DEG = NOMINAL_PITCH_DEG + MOUNT_PITCH_OFFSET_DEG   # 18.5908

# --- stereo noise model sigma_z(r) = a + b r^2, re-measured at the fitted pitch (build_v3) ----
SIG_A = 0.05185
SIG_B = 0.00343

# --- belief / fusion -------------------------------------------------------------------------
CELL = 0.10                 # [m] map cell
R_MIN, R_MAX = 0.35, 4.0    # [m] usable slant-range band
STRIDE = 4                  # pixel decimation
ALPHA = 0.0                 # Fankhauser keep-the-highest semantics
MAX_VARIANCE = 1.0e9        # the maxVariance clamp is LIFTED (spike Item 2)
PAD_M = R_MAX + 1.0

# --- windowing -------------------------------------------------------------------------------
# 4 m is the depth camera's range cap (R_MAX). The first design phase's C2 failure showed a
# 10 m window is structurally unforeseeable by a 4 m sensor: from before its own start such a
# window is never observed past ~4 m, so foresight retention clustered at 0.40-0.55 and 11 of
# 17 windows fell below the retention floor. The window length now matches the perception
# horizon, which is also the horizon a planner predicts over.
WIN_LEN_M = 4.0             # contiguous along-track window
LEGACY_WIN_LEN_M = 10.0     # the pre-edit length, kept ONLY for the C1 code-parity check
LOOKBACK_M = 7.0            # frames from this far behind the window start may fuse into it
TRACK_SMOOTH_FIXES = 9      # centred rolling mean on the RTK track before arc length
MIN_FRAMES_PER_WINDOW = 5

# --- geometry --------------------------------------------------------------------------------
R_WHEEL = 0.075             # [m] Gerdes et al. arXiv:2411.04700 Fig. 10 (15 cm diameter)
WHEEL_PATCH_R = 0.05        # [m] DSM disc averaged per wheel in the registration
N_BLOCKS = 5                # registration block-CV folds

# --- QC flags: PREREG_baseprod.md "QC flags". NOT alterable by the runner. --------------------
MIN_RETAINED = 0.50         # window EXCLUDED below this
FLAG_MIN_OBS_FRAC = 0.70    # window FLAGGED below this
FLAG_MAX_SIGMA_M = 0.10     # window FLAGGED above this
FLAG_MAX_INVALID_FRAC = 0.05

# --- arms ------------------------------------------------------------------------------------
# Every variance-predicting arm; the recalibration is applied identically to all of them.
# mean-map is a point prediction with no predicted variance and is handled separately.
ARMS_VARIANCE = ("clark", "clark-diag", "fosm", "step-form", "mc-2", "mc-32")
ARMS_NO_MC = ("clark", "clark-diag", "fosm", "step-form")
ARMS_LEGACY = ("clark", "clark-diag", "fosm")   # the pre-edit set, for the C1 parity check
FIT_ARM = "clark"

# step-form (the deployed practice, fan2021step). PINNED FORMULA:
#   E  = cost(mean map)                          == the mean-map point prediction
#   sd = sum_n |c_eff[n]| * sqrt(C[u(n), u(n)])
# where n runs over contact stencil nodes, c_eff[n] is the same per-step settle weight the cost
# itself uses, and u(n) = u_idx[n, argmax_j means[n, j]] is the MEAN-MAP WINNER cell of node n.
# It takes the winner cell's own marginal sd -- NO max (no moment-matched maximum of the
# candidates) -- and sums sds rather than forming a quadratic form -- NO covariance (the
# perfect-correlation, per-step Gaussian-CVaR proxy).
STEP_FORM_FORMULA = (
    "E = cost(mean map); sd = sum over contact nodes n of |c_eff[n]| * sqrt(C[u(n), u(n)]), "
    "u(n) = u_idx[n, argmax_j means[n, j]] the mean-map winner cell of node n; no max, no "
    "covariance (per-step Gaussian-CVaR proxy)")

N_MC_DRAWS = 20_000         # E2-iii: >= 20,000 draws per window
MC_SEED = 20260828
MC_CHUNK = 250
# mc-N arms: E and sd are the sample statistics of N draws SUBSAMPLED without replacement from
# the reference draws, with this fixed seed. N = 2 is the equal-compute point (the estimator
# costs 1.33 rollouts); N = 32 is a generous real-time budget.
MC_SUBSAMPLE_SEED = 20260828
MC_N_ARMS = (2, 32)

# CVaR reported under the belief referee: the mean of the upper (1 - q) tail of the cost.
# For a Gaussian (E, sd): CVaR_q = E + sd * phi(Phi^-1(q)) / (1 - q).
CVAR_Q = 0.9
CVAR_GAUSS_FACTOR = 1.7549833193248682   # phi(Phi^-1(0.9)) / 0.1

# the sd-deficit-vs-contest-depth curve: fixed bin edges on the cost-weighted median alpha
ALPHA_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, float("inf"))

# --- freeze conditions: PREREG_baseprod.md "Authorized autonomous execution" ------------------
# C1 changed meaning when the window length dropped to 4 m: reproducing v3 is no longer the
# same experiment. It is now a two-part CODE-PARITY condition -- (a) the 10 m hindsight scoring
# path still reproduces rehearsal v3 exactly, so the change is a configuration change and not a
# code change; (b) the 4 m pipeline runs both conditions end to end with finite outputs.
C1_REL_TOL = 1.0e-6
C2_MIN_UNFLAGGED_FRAC = 0.60

# --- E2 criteria bands (evaluated only on held-out data; recorded here, never altered) --------
E2_I_COV1_BAND = (0.55, 0.85)
E2_II_SD_RATIO_BAND = (0.7, 1.4)
E2_III_MAX_MEDIAN_REL_ERR = 0.05        # (a) MEAN: clark's median relative error
E2_III_SIGN_TEST_ALPHA = 0.05
E2_III_SD_BAND = (0.85, 1.05)           # (b) SD: clark's sd-ratio to the belief MC, absolute
# Pooling choice for E2-iii(b), fixed here in advance: the criterion is read on the MEDIAN of
# the per-window sd-ratio over unflagged windows. Mean and p5/p95 are reported beside it.
E2_III_SD_POOL = "median"

# --- banned corrections (PREREG_baseprod.md "Arms and recalibration") ------------------------
BANNED_CORRECTIONS = (
    "per-window truth-fitted mean correction",
    "per-traverse truth-fitted mean correction",
    "per-traverse range-flat vertical constant (sized at ~0.08 m on t1 / ~0.00 m on t2; "
    "would cut residual rms to ~0.17 m and k to ~165 -- BANNED, it is a truth-fitted mean)",
    "any refit of (k, tau) on held-out data",
    "any per-traverse camera pitch fitted against truth",
)
