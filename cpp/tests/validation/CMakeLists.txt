# =============================================================================
# Opt-in cross-language validations (NOT part of the default build or `ctest`).
#
# These are standalone pass/fail programs (exit 0 = PASS, 1 = FAIL) that check
# the C++ implementations against frozen R reference dumps. They are heavier and
# data-dependent, so they are deliberately kept OUT of the unit-test suite:
#
#   * the executables are EXCLUDE_FROM_ALL (a normal build does not build them),
#   * they are NOT registered with CTest (a bare `ctest` never runs them).
#
# Run them on demand with the aggregate target:
#
#     cmake --build build/debug --target run_validations
#
# which builds both and runs them in sequence, failing if either returns non-zero.
# The reference dumps live under data/rdump_tlars/ (committed; see data/README.md).
# The dump path is injected via TREX_VALIDATION_RDUMP_DIR; each program also
# accepts --dir <path> to override.
# =============================================================================

# Helper: build a standalone validation program (excluded from the default build).
function(add_trex_validation TEST_NAME)
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            trex::utils
            trex::ml_methods
    )
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_20)
    target_compile_definitions(${TEST_NAME} PRIVATE
        TREX_VALIDATION_RDUMP_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/rdump_tlars")
endfunction()

# HAC clustering cross-checks against R hclust() / add_dummies_GVS().
add_trex_validation(validation_mlm_hac_01_rcompare
    hac_clustering/validation_mlm_hac_01_rcompare/validation_mlm_hac_01_rcompare.cpp)

add_trex_validation(validation_mlm_hac_02_gvs_dummy_rcompare
    hac_clustering/validation_mlm_hac_02_gvs_dummy_rcompare/validation_mlm_hac_02_gvs_dummy_rcompare.cpp)

# Aggregate on-demand runner: builds both, runs them, fails on non-zero exit.
add_custom_target(run_validations
    COMMAND $<TARGET_FILE:validation_mlm_hac_01_rcompare>
    COMMAND $<TARGET_FILE:validation_mlm_hac_02_gvs_dummy_rcompare>
    DEPENDS validation_mlm_hac_01_rcompare validation_mlm_hac_02_gvs_dummy_rcompare
    COMMENT "Running opt-in cross-language validations (HAC vs R)..."
    VERBATIM)
