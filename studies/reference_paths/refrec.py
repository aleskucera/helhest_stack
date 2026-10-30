"""Record the reference stage's inputs and outputs every frame, in the layout the node's
plan_ref_record writes, so one offline replayer (replay_ref.py) reads both."""

from __future__ import annotations

import json

import numpy as np


class RefRecorder:
    def __init__(self) -> None:
        self.rows: dict[str, list] = {}

    def add(self, refpath, V, blocked, clear, sgrid, state_l, goal_l, n_ref, origin, t) -> None:
        rows = {
            "t": float(t),
            "V": V.numpy(),
            "blocked": (blocked.numpy() > 0.5).astype(np.uint8),
            "clear": clear.numpy(),
            "state": np.asarray(state_l, np.float32)[:3],
            "goal": np.asarray(goal_l, np.float32)[:2],
            "sgrid": np.array(
                [sgrid.origin_x, sgrid.origin_y, sgrid.cell_size, sgrid.cells_x, sgrid.cells_y]
            ),
            "origin": np.asarray(origin, np.float64)[:2],
            "n_ref": int(n_ref),
            "curve": refpath.pts.numpy()[: max(int(n_ref), 0)],
            "taut": refpath.taut.numpy()[: int(refpath.n_taut.numpy()[0])],
            "desc": refpath.desc.numpy()[: int(refpath.n_desc.numpy()[0])],
            "cold": int(refpath.cold.numpy()[0]),
        }
        for k, v in rows.items():
            self.rows.setdefault(k, []).append(v)

    def save(self, path: str, refpath, **extra: np.ndarray) -> None:
        out = {}
        for k, v in self.rows.items():
            if k in ("curve", "taut", "desc"):  # ragged: concatenated, with the row lengths
                out[f"{k}_n"] = np.array([len(a) for a in v], np.int32)
                out[k] = np.concatenate(v) if v else np.zeros((0, 2), np.float32)
            else:
                out[k] = np.asarray(v)
        out["params"] = np.array(json.dumps(refpath.params.__dict__))
        out["half_width"] = np.array(refpath.half_width)
        out["rear"] = np.array(refpath.rear)
        np.savez_compressed(path, **out, **extra)
