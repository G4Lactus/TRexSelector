#!/bin/bash
# =============================================================================
# run_install_smoke.sh — install the C++ SDK into a scratch prefix and build
# a find_package consumer against it (the install / find_package smoke test).
#
#   usage: cpp/tests/install_smoke/run_install_smoke.sh [build-dir]
#          (default build dir: cpp/build/release)
#   TREX_SMOKE_CONSUMER_ARGS: extra CMake arguments for the consumer, e.g.
#          -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON to prove that the config of a
#          single-threaded install asks for no OpenMP.
#
# Checks, in order: the headers land under the NAMESPACED root
# include/TRexSelector/ and nowhere else; the stamp is written; a downstream
# project finds the package and runs a selection; a second install prunes and
# rewrites the tree; a foreign directory at the root is REFUSED, not deleted.
# =============================================================================
set -euo pipefail

CPP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${1:-$CPP_ROOT/build/release}"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/trex_install_smoke.XXXXXX")"
PREFIX="$SCRATCH/prefix"
trap 'rm -rf "$SCRATCH"' EXIT

echo "== install into $PREFIX"
cmake --install "$BUILD_DIR" --prefix "$PREFIX" > "$SCRATCH/install1.log"
test -f "$PREFIX/include/TRexSelector/TRexSelector.install-stamp" || { echo "FAIL: stamp missing"; exit 1; }
for d in ml_methods trex_selector_methods tsolvers utils; do
    test -d "$PREFIX/include/TRexSelector/$d" || { echo "FAIL: include/TRexSelector/$d missing"; exit 1; }
    test ! -e "$PREFIX/include/$d" || { echo "FAIL: generic include/$d was installed"; exit 1; }
done
test -f "$PREFIX/lib/cmake/TRexSelector/TRexSelectorConfig.cmake" || { echo "FAIL: package config missing"; exit 1; }

echo "== build the find_package consumer"
# shellcheck disable=SC2086  # the extra arguments split on purpose
cmake -S "$CPP_ROOT/tests/install_smoke" -B "$SCRATCH/consumer" \
      -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release ${TREX_SMOKE_CONSUMER_ARGS:-} \
      > "$SCRATCH/consumer-configure.log" 2>&1 \
    || { cat "$SCRATCH/consumer-configure.log"; echo "FAIL: consumer configure"; exit 1; }
grep -i -A3 'warning' "$SCRATCH/consumer-configure.log" || true
cmake --build "$SCRATCH/consumer" -j4 > "$SCRATCH/consumer-build.log" 2>&1 \
    || { tail -30 "$SCRATCH/consumer-build.log"; echo "FAIL: consumer build (a header the code includes is missing from an install FILE_SET?)"; exit 1; }
"$SCRATCH/consumer/smoke"

echo "== second install prunes and rewrites the stamped tree"
cmake --install "$BUILD_DIR" --prefix "$PREFIX" > "$SCRATCH/install2.log" 2>&1
grep -q "pruned the package header tree" "$SCRATCH/install2.log" \
    || { cat "$SCRATCH/install2.log"; echo "FAIL: second install did not prune"; exit 1; }
test -f "$PREFIX/include/TRexSelector/TRexSelector.install-stamp" || { echo "FAIL: stamp not rewritten"; exit 1; }

echo "== a foreign include/TRexSelector is refused, not deleted"
FOREIGN="$SCRATCH/foreign"
mkdir -p "$FOREIGN/include/TRexSelector"
echo "not ours" > "$FOREIGN/include/TRexSelector/keep.h"
if cmake --install "$BUILD_DIR" --prefix "$FOREIGN" > "$SCRATCH/install-foreign.log" 2>&1; then
    echo "FAIL: install into a foreign tree should have been refused"; exit 1
fi
test -f "$FOREIGN/include/TRexSelector/keep.h" || { echo "FAIL: foreign file was deleted"; exit 1; }

echo "install smoke: OK"
