#!/usr/bin/env bash
# Build Project Chrono 9.0.1 from source WITH the vehicle module (and therefore SCM), which the
# conda `pychrono` package does not ship. User-space, no root. ~30 min on 18 cores.
#
#   scripts/build_chrono.sh          # then run things through scripts/chrono_env.sh
#
# Three things bite, all recorded here because none of them is guessable:
#
#  1. Eigen. conda-forge's default `eigen` is now 5.x, whose version macros moved out of
#     Eigen/src/Core/util/Macros.h -- which is exactly the file Chrono's FindEigen3.cmake parses,
#     so it reports "version .. found" and fails. Pin eigen 3.4.
#  2. SWIG. 4.3+ changed SWIG_Python_AppendOutput to take a third `is_void` argument, and Chrono
#     9.0.1's interface files predate it, so every generated wrapper fails to compile. Pin 4.2.
#     Note that swapping the swig package does NOT update SWIG_DIR in an existing CMake cache --
#     it keeps pointing at the old share/swig/<version> and then cannot find std_pair.i.
#  3. ChVehicleVisualSystem is abstract (ChVisualSystem::Initialize is pure virtual, reached
#     through a VIRTUAL base). SWIG does not follow that and emits a constructor that will not
#     compile. %nodefaultctor does not help -- the class declares its constructor explicitly -- so
#     the class is %ignore'd, which is free in a headless build. Ninja does NOT track the .i file
#     as a dependency of the generated wrapper, so the wrapper must be deleted to regenerate.
set -euo pipefail

SRC="${CHRONO_SRC:-/tmp/chrono-src}"
BUILD="${CHRONO_BUILD:-/tmp/chrono-build}"
ENVDIR="${CHRONO_BUILD_ENV:-$HOME/.local/opt/chrono-build}"
INSTALL="${CHRONO_INSTALL:-$HOME/.local/opt/chrono-install}"
JOBS="${JOBS:-$(( $(nproc) - 4 ))}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/opt/mamba}"
micromamba create -y -p "$ENVDIR" -c conda-forge \
    python=3.12 numpy "cmake<4" ninja "swig=4.2.*" "eigen=3.4" \
    "gxx_linux-64=13" "gcc_linux-64=13" make pkg-config

[[ -d "$SRC" ]] || git clone --depth 1 --branch 9.0.1 https://github.com/projectchrono/chrono.git "$SRC"

# see note 3
IFACE="$SRC/src/chrono_swig/interface/vehicle/ChModuleVehicle.i"
if ! grep -q "%ignore chrono::vehicle::ChVehicleVisualSystem;" "$IFACE"; then
    python - "$IFACE" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = '%include "../../../chrono_vehicle/ChVehicleVisualSystem.h"'
p.write_text(s.replace(old, "%ignore chrono::vehicle::ChVehicleVisualSystem;\n" + old, 1))
PY
fi

export CC="$ENVDIR/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENVDIR/bin/x86_64-conda-linux-gnu-g++"
export PATH="$ENVDIR/bin:$PATH"

"$ENVDIR/bin/cmake" -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTALL" \
    -DENABLE_MODULE_VEHICLE=ON -DENABLE_MODULE_PYTHON=ON \
    -DEIGEN3_INCLUDE_DIR="$ENVDIR/include/eigen3" \
    -DTHRUST_INCLUDE_DIR=/opt/cuda/include/cccl \
    -DPython3_EXECUTABLE="$ENVDIR/bin/python" \
    -DSWIG_EXECUTABLE="$ENVDIR/bin/swig" -DSWIG_DIR="$ENVDIR/share/swig/4.2.1" \
    -DBUILD_DEMOS=OFF -DBUILD_TESTING=OFF -DBUILD_BENCHMARKING=OFF

rm -f "$BUILD"/chrono_python/*_wrap.cxx  # see note 3: ninja does not track the .i files
"$ENVDIR/bin/cmake" --build "$BUILD" -j "$JOBS"
"$ENVDIR/bin/cmake" --install "$BUILD"

# Chrono installs the SWIG modules one level ABOVE the package that does `from . import _core`
PYDIR="$INSTALL/share/chrono/python"
for f in _core _fea _robot _vehicle; do ln -sf "$PYDIR/$f.so" "$PYDIR/pychrono/$f.so"; done

PYTHONPATH="$PYDIR" LD_LIBRARY_PATH="$INSTALL/lib:$ENVDIR/lib" "$ENVDIR/bin/python" -c \
    "import pychrono, pychrono.vehicle as v; print('OK: vehicle + SCM available:', v.SCMTerrain)"
