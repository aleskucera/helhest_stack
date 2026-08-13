#!/usr/bin/env bash
# Run a script against the SOURCE-BUILT Chrono (vehicle + SCM), which the conda pychrono lacks.
#
#   scripts/chrono_env.sh scripts/chrono_vehicle_model.py --terrain scm --out /tmp/veh_scm.json
#
# Build recipe is in scripts/build_chrono.sh. Two layout notes that will otherwise waste an hour:
# Chrono installs the SWIG modules one level ABOVE the package that imports them, so `_core.so`
# and friends are symlinked into `pychrono/`; and the engine libraries are not on the default
# loader path, hence LD_LIBRARY_PATH.
set -euo pipefail

CHRONO_INSTALL="${CHRONO_INSTALL:-$HOME/.local/opt/chrono-install}"
CHRONO_BUILD_ENV="${CHRONO_BUILD_ENV:-$HOME/.local/opt/chrono-build}"
PYDIR="$CHRONO_INSTALL/share/chrono/python"

if [[ ! -f "$PYDIR/pychrono/_vehicle.so" ]]; then
    echo "Source-built Chrono not found at $CHRONO_INSTALL -- run scripts/build_chrono.sh" >&2
    exit 1
fi

export PYTHONPATH="$PYDIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CHRONO_INSTALL/lib:$CHRONO_BUILD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$CHRONO_BUILD_ENV/bin/python" "$@"
