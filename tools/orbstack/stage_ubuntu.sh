#!/bin/bash
# Runs INSIDE the ubuntu:24.04 container (started by run_ubuntu.sh).
# Mirrors the three Ubuntu CI jobs on the exported tree at /work/repo, plus
# the install smoke, R CMD build and R CMD check. Every step logs to
# /work/logs/<step>.log; the summary goes to /work/logs/summary.log.
#
#   bash stage_ubuntu.sh            all steps (apt first)
#   bash stage_ubuntu.sh --no-apt   in a provisioned image (run_ubuntu.sh --reuse)
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
REPO=/work/repo
LOG=/work/logs
SUM=$LOG/summary.log
: > "$SUM"
say() { echo "$*" | tee -a "$SUM"; }
step() { say ""; say "===== $1 ($(date +%T)) ====="; }
# run NAME cmd... : logs to $LOG/NAME.log, records the exit code in the summary
run() {
    local name="$1"; shift
    step "$name"
    ( cd "$REPO" && "$@" ) > "$LOG/$name.log" 2>&1
    local rc=$?
    say "$name: exit $rc"
    tail -n 4 "$LOG/$name.log" | sed 's/^/    /' | tee -a "$SUM"
    return $rc
}

say "Stage A start $(date) on $(uname -m), nproc=$(nproc)"

# ---------------------------------------------------------------- 1. apt
# cpp-tests.yml's list, plus R with the CRAN packages Ubuntu ships as
# binaries, pandoc/qpdf for R CMD check, and Python.
if [ "${1:-}" != "--no-apt" ]; then
    step apt
    { apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        build-essential cmake git pkg-config ca-certificates curl rsync \
        libeigen3-dev libboost-all-dev libcereal-dev libgtest-dev libomp-dev \
        libopenblas-dev liblapack-dev \
        r-base r-base-dev r-cran-rcpp r-cran-rcppeigen r-cran-bh r-cran-r6 \
        r-cran-matrix r-cran-testthat r-cran-knitr r-cran-rmarkdown r-cran-ggplot2 \
        pandoc qpdf locales \
        python3 python3-venv python3-pip python3-dev \
      && locale-gen en_US.UTF-8; } > "$LOG/apt.log" 2>&1
    say "apt: exit $?"
    # Rcereal is LinkingTo and not in apt; patchwork (Suggests) may fail
    # to install from source on the apt ggplot2 and is not required.
    Rscript -e 'install.packages("Rcereal", repos = "https://cloud.r-project.org", quiet = TRUE); stopifnot(requireNamespace("Rcereal"))' > "$LOG/r_deps.log" 2>&1
    say "r_deps: exit $?"
fi
say "    $(cmake --version | head -1); $(g++ --version | head -1); $(R --version | head -1); $(python3 --version)"
eigen_version=$(pkg-config --modversion eigen3 2>/dev/null || echo '?')
boost_version=$(grep 'define BOOST_LIB_VERSION' /usr/include/boost/version.hpp | awk '{print $3}')
say "    eigen: $eigen_version; boost: $boost_version"

# ---------------------------------------------------------------- 2. C++
run cpp_configure cmake -S cpp -B cpp/build/ci -DCMAKE_BUILD_TYPE=Release -DTREX_PORTABLE_BUILD=ON
run cpp_build cmake --build cpp/build/ci --config Release --parallel "$(nproc)"
run cpp_ctest ctest --test-dir cpp/build/ci -C Release --output-on-failure
grep -E "tests passed|tests failed|Total Test time" "$LOG/cpp_ctest.log" | sed 's/^/    /' | tee -a "$SUM"
run cpp_install_smoke bash cpp/tests/install_smoke/run_install_smoke.sh cpp/build/ci

# ---------------------------------------------------------------- 3. R
run r_vendor_check bash R/TRexSelector/tools/bundle_for_cran.sh --check
run r_install R CMD INSTALL --no-docs R/TRexSelector
step r_testthat
( cd "$REPO/R/TRexSelector" && Rscript -e 'library(TRexSelector)
    res <- testthat::test_dir("tests/testthat", stop_on_failure = TRUE)
    df <- as.data.frame(res)
    cat(sprintf("TOTAL: %d passed, %d failed, %d skipped\n",
                sum(df$passed), sum(df$failed), sum(df$skipped)))' ) > "$LOG/r_testthat.log" 2>&1
say "r_testthat: exit $?"
grep -E "^TOTAL:|Error|FAIL" "$LOG/r_testthat.log" | tail -3 | sed 's/^/    /' | tee -a "$SUM"
# The CRAN-shaped check on the built tarball (vignettes skipped: the
# usage-and-simulations vignette is a long run).
rm -rf "$REPO"/TRexSelector_*.tar.gz "$REPO"/TRexSelector.Rcheck
run r_build R CMD build --no-build-vignettes --no-manual R/TRexSelector
TARBALL=$(ls -t "$REPO"/TRexSelector_*.tar.gz 2>/dev/null | head -1)
step r_check
# A UTF-8 session: en_US.UTF-8 (generated in the apt step) or the C.UTF-8 every
# image has (a snapshot taken before locale-gen holds only that one). An ASCII
# session makes the DESCRIPTION meta-information check fail with a bare ERROR.
UTF8_LOCALE=$(locale -a 2>/dev/null | grep -qiE '^en_US\.utf-?8$' && echo en_US.UTF-8 || echo C.UTF-8)
( cd "$REPO" && LANG=$UTF8_LOCALE LC_ALL=$UTF8_LOCALE \
    _R_CHECK_CRAN_INCOMING_=false _R_CHECK_FORCE_SUGGESTS_=false \
    R CMD check --no-manual --ignore-vignettes --no-build-vignettes "$TARBALL" ) > "$LOG/r_check.log" 2>&1
say "r_check: exit $?"
grep -E "^Status:|WARNING|ERROR|NOTE" "$LOG/r_check.log" | sed 's/^/    /' | tee -a "$SUM"

# ---------------------------------------------------------------- 4. Python
step py_venv
python3 -m venv /work/venv > "$LOG/py_venv.log" 2>&1 \
  && /work/venv/bin/pip install -q --upgrade pip numpy pytest >> "$LOG/py_venv.log" 2>&1
say "py_venv: exit $?"
run py_build /work/venv/bin/pip install -v Python/trexselector
step py_pytest
( cd "$REPO/Python/trexselector" \
  && /work/venv/bin/python -X faulthandler -m pytest tests -x -q ) > "$LOG/py_pytest.log" 2>&1
say "py_pytest: exit $?"
tail -n 2 "$LOG/py_pytest.log" | sed 's/^/    /' | tee -a "$SUM"

say ""
say "STAGE A DONE $(date)"
