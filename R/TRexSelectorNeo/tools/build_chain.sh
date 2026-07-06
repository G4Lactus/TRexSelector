#!/usr/bin/env bash
set -e

# Step 1: Bundle C++ backend into the R package and generate CRAN Makevars
cd R/TRexSelector && chmod +x bundle_for_cran.sh && ./bundle_for_cran.sh && cd ../..

# Step 2: Build the source tarball
R CMD build R/TRexSelector

# Step 3: Full CRAN check (adjust version number if needed)
R CMD check --as-cran TRexSelector_2.0.0.tar.gz
