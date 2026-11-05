# ==================================================================================
# TRexCompilerOptions.cmake
# ==================================================================================
# Shared per-target compile options for all trex libraries.
# (Previously duplicated across the four src/*/CMakeLists.txt files.)
#
# NOTE: -march=native (MSVC: /arch:AVX2) is gated behind NOT TREX_PORTABLE_BUILD
# (wheel/PyPI builds).
#
# FLAG POLICY: Release builds are
# IEEE 754 conformant — NO -ffast-math / /fp:fast in any configuration.
# -ffast-math implies -ffinite-math-only, under which the compiler tags every
# floating-point argument and return value as finite and folds away exactly
# the NaN/Inf observability the guards depend on: with it, every
# fp_classify::isfinite / isinf / isnan in the Release library compiled to a
# constant (found through a validator that let an infinite lambda_2 through),
# and a literal infinity materialized into a container arrives as an
# arbitrary finite value. The bit-based checks in utils/fp_classify stay
# mandatory in HEADERS, because downstream consumers may still compile them
# with fast-math.
# ==================================================================================

function(trex_set_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /wd4996  # ~ -Wno-deprecated-declarations
            # /O2 /DNDEBUG are MSVC Release defaults; no -funroll-loops analog needed
            $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TREX_PORTABLE_BUILD}>>>:/arch:AVX2>
        )
    else()
        target_compile_options(${target} PRIVATE
            -Wno-deprecated-declarations
            $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
            $<$<CONFIG:Release>:-O3 -DNDEBUG -funroll-loops>
            $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TREX_PORTABLE_BUILD}>>>:-march=native>
        )
    endif()
endfunction()
