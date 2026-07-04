# ==================================================================================
# TRexCompilerOptions.cmake
# ==================================================================================
# Shared per-target compile options for all trex libraries.
# (Previously duplicated across the four src/*/CMakeLists.txt files.)
#
# NOTE: -ffast-math / -march=native (and their MSVC counterparts /fp:fast /
# /arch:AVX2) are gated behind NOT TREX_PORTABLE_BUILD (wheel/PyPI builds).
# ==================================================================================

function(trex_set_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /wd4996  # ~ -Wno-deprecated-declarations
            # /O2 /DNDEBUG are MSVC Release defaults; no -funroll-loops analog needed
            $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TREX_PORTABLE_BUILD}>>>:/arch:AVX2 /fp:fast>
        )
    else()
        target_compile_options(${target} PRIVATE
            -Wno-deprecated-declarations
            $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
            $<$<CONFIG:Release>:-O3 -DNDEBUG -funroll-loops>
            $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TREX_PORTABLE_BUILD}>>>:-march=native -ffast-math>
        )
    endif()
endfunction()
