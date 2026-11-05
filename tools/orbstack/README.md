# Linux verification in OrbStack (Docker)

Reproduces the three Ubuntu CI jobs and the Linux wheel build locally, on
an export of the committed `HEAD` (the worktree is never touched). Its first
run found two macOS-calibrated C++ tests and three unparseable Rd pages that
CI could not see.

## Stage A: `ubuntu:24.04`, native arm64

```sh
tools/orbstack/run_ubuntu.sh            # ~20 min on 4 cores: apt, C++, R, Python
```

Mirrors `cpp-tests.yml` (apt dependencies, `-DTREX_PORTABLE_BUILD=ON`,
ctest), `r-package.yml` (vendor check, `R CMD INSTALL --no-docs`, testthat)
and the cibuildwheel test command on a Python source build, and adds the
install smoke, `R CMD build` and `R CMD check` (vignettes skipped). Every step
logs to `<workdir>/logs/`, the summary to `<workdir>/logs/summary.log`. The
provisioned container is committed as the image `trex-orb-ubuntu:deps`, so a
rerun of one step (`run_ubuntu.sh --reuse`) skips apt.

Expected noise in the check: the installed-size NOTE (the shared object)
and the Ubuntu `BH` NOTE (the apt package strips its headers). The check
runs in `en_US.UTF-8` (generated in the apt step) or in the image's
`C.UTF-8`; an ASCII session turns the `BH` NOTE into a bare ERROR.

## Stage B: manylinux_2_28 x86_64 wheels through cibuildwheel

```sh
tools/orbstack/run_manylinux.sh                       # cp312 only, ~3 min
CIBW_BUILD="cp310-* cp311-* cp312-* cp313-*" tools/orbstack/run_manylinux.sh   # the CI matrix
```

Runs the CI-pinned cibuildwheel (2.21.3) with the workflow's Linux environment
verbatim (Eigen 5.0.1, cereal 1.3.2 and Boost 1.85 built into the image); the
x86_64 image runs under Rosetta. Each wheel is tested inside the container
with `python -X faulthandler -m pytest tests -x -q`. Wheels land in
`<workdir>/wheelhouse/`.

Both drivers take `TREX_ORB_WORKDIR` (default `${TMPDIR:-/tmp}/trex-orbstack`).
