from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Knobs — edit these to change sources, labels, and output path.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# BAG 0
TRAJ_A_CSV: pathlib.Path = _REPO_ROOT / "data" / "traj_20260723_095447.csv"
TRAJ_B_CSV: pathlib.Path = _REPO_ROOT / "data" / "ekf_traj_20260723_164718.csv"     # EKF

# BAG 1
#TRAJ_A_CSV: pathlib.Path = _REPO_ROOT / "data" / "traj_20260722_155940.csv"
#TRAJ_B_CSV: pathlib.Path = _REPO_ROOT / "data" / "traj_20260722_160247.csv"     # EKF

# BAG 2
# TRAJ_A_CSV: pathlib.Path = _REPO_ROOT / "data" / "traj_20260720_142618.csv"
# TRAJ_B_CSV: pathlib.Path = _REPO_ROOT / "data" / "traj_20260722_131017.csv"     # EKF

LABEL_A: str = "old localization"
LABEL_B: str = "ekf"

# Show a third trajectory from the EKF predict (pre-update) state.
SHOW_EKF_PRED: bool = True
LABEL_PRED: str = "ekf pred"

TITLE = "ROSBAG 0"
# ---------------------------------------------------------------------------

# EKF CSV columns: t_sec,x_pred_m,y_pred_m,psi_pred_rad,x_upd_m,y_upd_m,psi_upd_rad
_EKF_HEADER_PREFIX = "x_pred_m"


def _load_xy(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_m, y_m) from a plain traj CSV (columns: t_sec,x_m,y_m,psi_rad)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2))
    return data[:, 0], data[:, 1]


def _load_ekf_xy(
    path: pathlib.Path,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Return ((x_upd, y_upd), (x_pred, y_pred)) from an EKF traj CSV.

    The pred pair is None when SHOW_EKF_PRED is False.
    Columns: t_sec,x_pred_m,y_pred_m,psi_pred_rad,x_upd_m,y_upd_m,psi_upd_rad
    """
    # pred: cols 1,2  /  upd: cols 4,5
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 4, 5))
    upd = data[:, 2], data[:, 3]
    pred = (data[:, 0], data[:, 1]) if SHOW_EKF_PRED else None
    return upd, pred


def _is_ekf_csv(path: pathlib.Path) -> bool:
    """Return True when the CSV header contains x_pred_m (EKF format)."""
    with path.open() as f:
        header = f.readline()
    return _EKF_HEADER_PREFIX in header


def main() -> None:
    x_a, y_a = _load_xy(TRAJ_A_CSV)

    # Load TRAJ_B — branches on CSV format
    x_pred, y_pred = None, None
    if _is_ekf_csv(TRAJ_B_CSV):
        (x_b, y_b), pred_pair = _load_ekf_xy(TRAJ_B_CSV)
        if pred_pair is not None:
            x_pred, y_pred = pred_pair
    else:
        x_b, y_b = _load_xy(TRAJ_B_CSV)
        if SHOW_EKF_PRED:
            print(
                f"[loc_pipeline_comparison] SHOW_EKF_PRED=True but {TRAJ_B_CSV.name} "
                "is a plain traj CSV (no pred columns) — skipping pred overlay."
            )

    fig, ax = plt.subplots(figsize=(8, 7))

    color_a = "#1f77b4"  # blue
    color_b = "#ff7f0e"  # orange
    color_p = "#2ca02c"  # green — EKF pred

    # Trajectory lines
    ax.plot(x_a, y_a, color=color_a, linewidth=1.2, label=LABEL_A)
    ax.plot(x_b, y_b, color=color_b, linewidth=1.2, label=LABEL_B)
    if x_pred is not None:
        ax.plot(x_pred, y_pred, color=color_p, linewidth=1.2, label=LABEL_PRED)

    # Start markers (filled circle, black edge)
    ax.scatter(x_a[0], y_a[0], marker="o", s=80, color=color_a, edgecolors="black", zorder=5)
    ax.scatter(x_b[0], y_b[0], marker="o", s=80, color=color_b, edgecolors="black", zorder=5)
    if x_pred is not None:
        ax.scatter(
            x_pred[0], y_pred[0], marker="o", s=80, color=color_p, edgecolors="black", zorder=5
        )

    # End markers (X)
    ax.scatter(x_a[-1], y_a[-1], marker="X", s=80, color=color_a, edgecolors="black", zorder=5)
    ax.scatter(x_b[-1], y_b[-1], marker="X", s=80, color=color_b, edgecolors="black", zorder=5)
    if x_pred is not None:
        ax.scatter(
            x_pred[-1], y_pred[-1], marker="X", s=80, color=color_p, edgecolors="black", zorder=5
        )

    # Shared legend entries for start / end conventions
    ax.scatter([], [], marker="o", s=80, color="grey", edgecolors="black", label="Start")
    ax.scatter([], [], marker="X", s=80, color="grey", edgecolors="black", label="End")

    ax.set_title(TITLE)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
