#!/usr/bin/env bash
set -e

# Resolve the package directory regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"      # R/TRexSelectorNeo
R_DIR="$(cd "$PKG_DIR/.." && pwd)"           # R/ — tarball and .Rcheck land here

# Step 1: Bundle C++ backend into the R package and generate CRAN Makevars
bash "$SCRIPT_DIR/bundle_for_cran.sh"

# Step 2: Build the source tarball (R CMD build writes into the CWD)
cd "$R_DIR"
R CMD build "$PKG_DIR"

# Step 3: Full CRAN check on the freshly built tarball
VERSION=$(sed -n 's/^Version: *//p' "$PKG_DIR/DESCRIPTION")
R CMD check --as-cran "TRexSelectorNeo_${VERSION}.tar.gz"
