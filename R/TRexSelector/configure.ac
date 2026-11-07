AC_INIT([TRexSelector], [2.0.1])

# Find the R compiler and flags
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`

# OpenMP is optional. The probes run in this order, and the first hit wins:
#   1. R's own flags (SHLIB_OPENMP_CXXFLAGS): Linux, Windows and CRAN.
#   2. macOS without them: R's bundled runtime ${R_HOME}/lib/libomp.dylib (the
#      one data.table and RcppArmadillo load) with an omp.h from Homebrew or
#      from the mac.r-project.org OpenMP package (/usr/local/include), so the
#      process maps exactly one OpenMP runtime.
#   3. macOS: Homebrew's libomp alone.
#   4. None: sequential build (utils_openmp.hpp stubs report one thread).
# TREX_OPENMP=no (environment or --configure-vars) forces the sequential build
# and silences the inactive pragmas (-Wno-unknown-pragmas, a local choice: R CMD
# check lists it as non-portable, so the automatic fallback does not add it).
AC_ARG_VAR([TREX_OPENMP], [set to "no" to compile without OpenMP (single-threaded)])
OPENMP_CXXFLAGS=""
OPENMP_LIBS=""
OPENMP_NOPRAGMA_FLAGS=""

if test "x${TREX_OPENMP}" = "xno"; then
  AC_MSG_NOTICE([TREX_OPENMP=no: OpenMP disabled])
  OPENMP_NOPRAGMA_FLAGS="-Wno-unknown-pragmas"
else
  # Step 1: R's native OpenMP flags. Makeconf defines SHLIB_OPENMP_CXXFLAGS
  # (empty where R has no OpenMP, as CRAN's R on macOS); R CMD config does not
  # serve the variable on R 4.6 and R-devel, so the file is read directly. When
  # the flags exist, Makevars carries the make variable itself, the form
  # Writing R Extensions asks for, and make expands it at build time.
  AC_MSG_CHECKING([for R's OpenMP flags])
  trex_makeconf="${R_HOME}/etc${R_ARCH}/Makeconf"
  trex_r_openmp=`sed -n 's/^SHLIB_OPENMP_CXXFLAGS[[ ]]*=[[ ]]*//p' "${trex_makeconf}" 2>/dev/null`
  if test -n "${trex_r_openmp}"; then
    OPENMP_CXXFLAGS='$(SHLIB_OPENMP_CXXFLAGS)'
    OPENMP_LIBS='$(SHLIB_OPENMP_CXXFLAGS)'
    AC_MSG_RESULT([${trex_r_openmp}])
  else
    OPENMP_CXXFLAGS=""
    OPENMP_LIBS=""
    AC_MSG_RESULT([none])
  fi

  # Step 2: macOS, R's bundled runtime with an omp.h (Homebrew, mac.r-project.org)
  if test -z "${OPENMP_CXXFLAGS}" && test "`uname -s`" = "Darwin"; then
    AC_MSG_CHECKING([for R's bundled OpenMP runtime with an omp.h])
    trex_omp_include=""
    for trex_dir in /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include /usr/local/include; do
      if test -f "${trex_dir}/omp.h"; then
        trex_omp_include="${trex_dir}"
        break
      fi
    done
    if test -f "${R_HOME}/lib/libomp.dylib" && test -n "${trex_omp_include}"; then
      OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp -I${trex_omp_include}"
      OPENMP_LIBS="-L${R_HOME}/lib -lomp"
      AC_MSG_RESULT([${R_HOME}/lib/libomp.dylib, omp.h from ${trex_omp_include}])
    else
      AC_MSG_RESULT([no])
    fi
  fi

  # Step 3: macOS, Homebrew's runtime alone
  if test -z "${OPENMP_CXXFLAGS}"; then
    AC_MSG_CHECKING([for macOS Homebrew OpenMP])
    if test -f "/opt/homebrew/opt/libomp/lib/libomp.dylib"; then
      OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
      OPENMP_LIBS="/opt/homebrew/opt/libomp/lib/libomp.dylib"
      AC_MSG_RESULT([/opt/homebrew/opt/libomp])
    elif test -f "/usr/local/opt/libomp/lib/libomp.dylib"; then
      OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
      OPENMP_LIBS="/usr/local/opt/libomp/lib/libomp.dylib"
      AC_MSG_RESULT([/usr/local/opt/libomp])
    else
      AC_MSG_RESULT([no])
    fi
  fi
fi

# Step 4: no OpenMP
if test -z "${OPENMP_CXXFLAGS}"; then
  AC_MSG_NOTICE([no OpenMP: compiling sequentially (one thread)])
fi

# Substitute into Makevars.in
AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIBS)
AC_SUBST(OPENMP_NOPRAGMA_FLAGS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
