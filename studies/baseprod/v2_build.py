"""v2 belief build (PREREG_attitude_cost.md section 5): the v1 pipeline
verbatim with exactly two constants changed at call time --- R_MAX = 3.0 m
and ALPHA = 1.0 --- following the patch discipline vis/rebuild_e2.py
verified (at pinned constants that path reproduces the committed v1 window
products exactly, so any difference is the two constants and nothing else).

Outputs land in out/windows/<condition>_v2/<traverse>/.

    BASEPROD_ROOT=... PYTHONPATH=<studies> python -m baseprod.v2_build design
    BASEPROD_ROOT=... PYTHONPATH=<studies> python -m baseprod.v2_build <traverse> [<traverse>...]
"""
from __future__ import annotations

import json
import sys
import time

from . import constants as K

R_MAX_V2, ALPHA_V2 = 3.0, 1.0


def apply_v2_constants():
    K.R_MAX = R_MAX_V2
    K.ALPHA = ALPHA_V2


def build_v2(names, conditions=("foresight", "hindsight")):
    apply_v2_constants()
    from .build import build          # reads K.* at call time
    metas = []
    for name in names:
        for cond in conditions:
            t0 = time.time()
            meta = build(name, cond, tag=f"{cond}_v2")
            meta["v2_constants"] = {"R_MAX": K.R_MAX, "ALPHA": K.ALPHA}
            metas.append(meta)
            print(f"[{time.strftime('%F %T')}] built {name} {cond}: "
                  f"{meta.get('n_windows')} windows in {time.time()-t0:.0f}s",
                  flush=True)
    return metas


def main():
    from .paths import DESIGN_TRAVERSES, OUT
    args = sys.argv[1:]
    names = list(DESIGN_TRAVERSES) if args == ["design"] else args
    if not names:
        print(__doc__)
        sys.exit(1)
    metas = build_v2(names)
    outp = OUT / "v2" / "build_meta.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(outp.read_text()) if outp.exists() else []
    existing.extend(metas)
    outp.write_text(json.dumps(existing, indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
