#!/bin/bash
# Stage B driver (host): the CI wheel path — cibuildwheel 2.21.3 (the version
# pinned in python-wheels.yml) driving the manylinux_2_28 x86_64 image under
# OrbStack/Rosetta, with the workflow's Linux environment verbatim. Runs on
# the committed HEAD export; every wheel is tested inside the container.
#
#   tools/orbstack/run_manylinux.sh                                    cp312 only (~3 min)
#   CIBW_BUILD="cp310-* cp311-* cp312-* cp313-*" tools/orbstack/run_manylinux.sh   the CI matrix (~13 min)
#
# Needs: docker (OrbStack) and `pip install cibuildwheel==2.21.3` in the
# Python env on PATH (or TREX_ORB_CIBW pointing at the executable).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${TREX_ORB_WORKDIR:-${TMPDIR:-/tmp}/trex-orbstack}
CIBW=${TREX_ORB_CIBW:-cibuildwheel}

mkdir -p "$WORK/wheelhouse" "$WORK/logs"
if [ ! -d "$WORK/repo" ]; then
    mkdir -p "$WORK/repo"
    ( cd "$ROOT" && git archive HEAD | tar -x -C "$WORK/repo" )
    echo "exported $(git -C "$ROOT" rev-parse --short HEAD) to $WORK/repo"
fi
cd "$WORK/repo"

export CIBW_BUILD="${CIBW_BUILD:-cp312-manylinux_x86_64}"
export CIBW_SKIP="*-musllinux_*"
export CIBW_TEST_REQUIRES="pytest"
export CIBW_TEST_COMMAND="python -X faulthandler -m pytest {package}/tests -x -q"
export CIBW_BUILD_VERBOSITY="1"
export CIBW_ARCHS_LINUX="x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux_2_28"
# Verbatim from python-wheels.yml (CIBW_BEFORE_ALL_LINUX).
export CIBW_BEFORE_ALL_LINUX='set -e
curl -fsSL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz | tar xz -C /tmp
cmake -S /tmp/eigen-5.0.1 -B /tmp/eigen-build -DEIGEN_BUILD_BLAS=OFF -DEIGEN_BUILD_LAPACK=OFF \
  -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --install /tmp/eigen-build
curl -fsSL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz | tar xz -C /tmp
cmake -S /tmp/cereal-1.3.2 -B /tmp/cereal-build -DJUST_INSTALL_CEREAL=ON -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --install /tmp/cereal-build
curl -fsSL https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz | tar xz -C /tmp
cp -r /tmp/boost_1_85_0/boost /usr/local/include/boost'

echo "Stage B start $(date): CIBW_BUILD=$CIBW_BUILD"
"$CIBW" --platform linux --output-dir "$WORK/wheelhouse" Python/trexselector 2>&1 | tee "$WORK/logs/manylinux.log"
rc=${PIPESTATUS[0]}
echo "cibuildwheel exit $rc"
ls -la "$WORK/wheelhouse"
exit $rc
