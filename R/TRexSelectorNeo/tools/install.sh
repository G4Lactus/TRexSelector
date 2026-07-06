#!/usr/bin/env bash
# install.sh — Re-bundle C++ sources and (re)install the TRexSelectorNeo R package.
#
# Usage (from any directory):
#   bash R/TRexSelectorNeo/tools/install.sh
#   # or, after chmod +x:
#   R/TRexSelectorNeo/tools/install.sh
#
# Content:
#   1. bundle_for_cran.sh  — copies the latest C++ sources from cpp/src/ into
#                             R/TRexSelectorNeo/src/ and regenerates all Makevars
#                             files.  Without this, R CMD INSTALL would compile
#                             stale code.
#   2. autoconf            — regenerates the configure script from configure.ac.
#                             R CMD INSTALL runs ./configure but never calls
#                             autoconf itself, so changes to configure.ac (e.g.
#                             new OpenMP probe paths) would otherwise be ignored.
#   3. Clean *.o files     — removes leftover object files from prior builds so
#                             that every translation unit is recompiled cleanly.
#   4. R CMD INSTALL       — compiles and installs the package into the active
#                             R library.

set -e

# Resolve the repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"      # R/TRexSelectorNeo
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"    # repo root

echo "=== Step 1: Bundle C++ backend into R package ==="
bash "$SCRIPT_DIR/bundle_for_cran.sh"

echo ""
echo "=== Step 2: Regenerate configure from configure.ac ==="
cd "$PKG_DIR"
autoconf
rm -f configure~          # autoconf backup — not needed
rm -rf autom4te.cache     # autoconf working cache — not needed after generation

echo ""
echo "=== Step 3: Clean stale object files ==="
find "$PKG_DIR/src" -name "*.o" -delete
echo "Stale .o files removed."

echo ""
echo "=== Step 4: Install package ==="
cd "$REPO_ROOT"
R CMD INSTALL "$PKG_DIR"

echo ""
echo "=== Done: TRexSelectorNeo installed successfully ==="
