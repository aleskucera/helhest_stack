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
WIN_LEN_M = 10.0            # contiguous along-track window
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

# --- scoring ---------------------------------------------------------------------------------
ARMS_SCORED = ("clark", "clark-diag", "fosm")
FIT_ARM = "clark"
N_MC_DRAWS = 20_000         # E2-iii: >= 20,000 draws per window
MC_SEED = 20260828
MC_CHUNK = 250

# --- freeze conditions: PREREG_baseprod.md "Authorized autonomous execution" ------------------
C1_REL_TOL = 1.0e-6         # v3 hindsight reproduction tolerance
C2_MIN_UNFLAGGED_FRAC = 0.60

# --- E2 criteria bands (evaluated only on held-out data; recorded here, never altered) --------
E2_I_COV1_BAND = (0.55, 0.85)
E2_II_SD_RATIO_BAND = (0.7, 1.4)
E2_III_MAX_MEDIAN_REL_ERR = 0.05
E2_III_SIGN_TEST_ALPHA = 0.05

# --- banned corrections (PREREG_baseprod.md "Arms and recalibration") ------------------------
BANNED_CORRECTIONS = (
    "per-window truth-fitted mean correction",
    "per-traverse truth-fitted mean correction",
    "per-traverse range-flat vertical constant (sized at ~0.08 m on t1 / ~0.00 m on t2; "
    "would cut residual rms to ~0.17 m and k to ~165 -- BANNED, it is a truth-fitted mean)",
    "any refit of (k, tau) on held-out data",
    "any per-traverse camera pitch fitted against truth",
)
