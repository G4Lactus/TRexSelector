AC_INIT([TRexSelectorNeo], [0.2.0])

# Find the R compiler and flags
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`

# Step 1: Check R's native OpenMP flags first (This covers CRAN and Linux/Windows)
OPENMP_CXXFLAGS=`"${R_HOME}/bin/R" CMD config SHLIB_OPENMP_CXXFLAGS 2>/dev/null`
# R's CMD config might output "ERROR" instead of being empty on macOS without OpenMP natively
if echo "${OPENMP_CXXFLAGS}" | grep -q "ERROR"; then
  OPENMP_CXXFLAGS=""
fi
OPENMP_LIBS="${OPENMP_CXXFLAGS}"

# Step 2: If R didn't find OpenMP, probe for macOS Homebrew (local dev builds)
if test -z "${OPENMP_CXXFLAGS}"; then
  AC_MSG_CHECKING([for macOS Homebrew OpenMP])
  if test -f "/opt/homebrew/opt/libomp/lib/libomp.dylib"; then
    OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
    OPENMP_LIBS="/opt/homebrew/opt/libomp/lib/libomp.dylib"
    AC_MSG_RESULT([found loosely in /opt/homebrew])
  elif test -f "/usr/local/opt/libomp/lib/libomp.dylib"; then
    OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
    OPENMP_LIBS="/usr/local/opt/libomp/lib/libomp.dylib"
    AC_MSG_RESULT([found loosely in /usr/local])
  else
    AC_MSG_RESULT([not found. Compiling sequentially.])
  fi
fi

# Substitute into Makevars.in
AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIBS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
