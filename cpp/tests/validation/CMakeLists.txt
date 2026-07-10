# =============================================================================
# Opt-in cross-language validations & diagnostics (NOT part of the default
# build or `ctest`).
#
# Two families of standalone programs live here, both kept OUT of the unit-test
# suite (executables are EXCLUDE_FROM_ALL; none are registered with CTest, so a
# bare `ctest` never builds or runs them):
#
#   1. Gated validations  -> `run_validations`
#      Pass/fail regression checks (exit 0 = PASS, 1 = FAIL) against frozen R
#      reference dumps. `run_validations` fails if ANY returns non-zero.
#
#   2. Diagnostics        -> `build_diagnostics`
#      Investigative probes/dumpers/comparisons migrated from the examples
#      package. They always exit 0 (they print numbers for a human to read, they
#      do not gate). Most are heavy Monte-Carlo studies that run for many minutes
#      each, so they are NOT auto-run: `build_diagnostics` just compiles all of
#      them (catching API drift) and you run whichever one you need by hand from
#      build/debug/bin/. Notes:
#        * self-generating probes (trex_01, spca_02/03/06, ts_01) just run;
#        * da_01 / spca_05 read committed reference CSVs and finish quickly;
#        * spca_04 / spca_07 read the baked R dumps committed beside them
#          (rdump/ + rdump10/, injected via TREX_SPCA_RDUMP{,10}_DIR; --dir
#          overrides; see the trex_spca README);
#        * gvs_01 targets the GVS-EN + TLARS combo the library now rejects (the
#          GVS-EN FDR fix); it compiles but throws at run time by design.
#
# Run on demand:
#     cmake --build build/debug --target run_validations     # the gate (fast)
#     cmake --build build/debug --target build_diagnostics   # compile the probes
#     ./build/debug/bin/<validation_...>                     # run one probe
#
# The gated reference dumps live under data/rdump_tlars/ (committed; see
# data/README.md), injected via TREX_VALIDATION_RDUMP_DIR; each gate also
# accepts --dir <path> to override.
# =============================================================================

# -----------------------------------------------------------------------------
# Helper: gated validation program (excluded from the default build).
# -----------------------------------------------------------------------------
function(add_trex_validation TEST_NAME)
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            trex::utils
            trex::ml_methods
            trex::tsolvers
    )
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_20)
    target_compile_definitions(${TEST_NAME} PRIVATE
        TREX_VALIDATION_RDUMP_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/rdump_tlars")
endfunction()

# -----------------------------------------------------------------------------
# Helper: diagnostic program (excluded from the default build).
#
# Links the full trex stack (diagnostics reach into every subsystem) and routes
# DEMO_OUTPUT_DIR to a validation_results/ folder next to the program's source,
# where its committed reference CSVs live and where its own output is written --
# matching the behavior these programs had in the examples package.
# -----------------------------------------------------------------------------
function(add_trex_diagnostic TEST_NAME)
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${ARGN})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            trex::utils
            trex::ml_methods
            trex::tsolvers
            trex::trex_selector_methods
    )
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_20)
    target_compile_options(${TEST_NAME} PRIVATE -Wno-deprecated-declarations)

    # Derive validation_results/ next to the first source file.
    list(GET ARGN 0 _first_src)
    get_filename_component(_src_subdir "${_first_src}" DIRECTORY)
    if(_src_subdir)
        set(_val_out "${CMAKE_CURRENT_SOURCE_DIR}/${_src_subdir}/validation_results/")
    else()
        set(_val_out "${CMAKE_CURRENT_SOURCE_DIR}/validation_results/")
    endif()
    file(MAKE_DIRECTORY "${_val_out}")

    target_compile_definitions(${TEST_NAME} PRIVATE
        DEMO_OUTPUT_DIR="${_val_out}")
endfunction()

# =============================================================================
# (1) Gated validations
# =============================================================================

# HAC clustering cross-checks against R hclust() / add_dummies_GVS().
add_trex_validation(validation_mlm_hac_01_rcompare
    hac_clustering/validation_mlm_hac_01_rcompare/validation_mlm_hac_01_rcompare.cpp)

add_trex_validation(validation_mlm_hac_02_gvs_dummy_rcompare
    hac_clustering/validation_mlm_hac_02_gvs_dummy_rcompare/validation_mlm_hac_02_gvs_dummy_rcompare.cpp)

# T-solver cross-checks against R lars/lasso/elastic-net reference dumps
# (TLARS / TLASSO / TENET / TENET_AUG).
add_trex_validation(validation_ts_02_tlars_tlasso_rcompare
    tsolvers/validation_ts_02_tlars_tlasso_rcompare/validation_ts_02_tlars_tlasso_rcompare.cpp)

add_custom_target(run_validations
    COMMAND $<TARGET_FILE:validation_mlm_hac_01_rcompare>
    COMMAND $<TARGET_FILE:validation_mlm_hac_02_gvs_dummy_rcompare>
    COMMAND $<TARGET_FILE:validation_ts_02_tlars_tlasso_rcompare>
    DEPENDS validation_mlm_hac_01_rcompare validation_mlm_hac_02_gvs_dummy_rcompare
            validation_ts_02_tlars_tlasso_rcompare
    COMMENT "Running opt-in cross-language validations (HAC + t-solvers vs R)..."
    VERBATIM)

# =============================================================================
# (2) Diagnostics (investigative; always exit 0)
# =============================================================================

# Classical T-Rex selector.
add_trex_diagnostic(validation_trex_01_scaling_comparison
    trex_selector_methods/trex/validation_trex_01_scaling_comparison.cpp)

# GVS T-Rex selector.
add_trex_diagnostic(validation_trex_gvs_01_scaling_solver_comparison
    trex_selector_methods/trex_gvs/validation_trex_gvs_01_scaling_solver_comparison.cpp)

# Dependency-Aware T-Rex selector (writes diag_*.csv compared by the sibling R script).
add_trex_diagnostic(validation_trex_da_01_bt_dendro_diag
    trex_selector_methods/trex_da/validation_trex_da_01_bt_dendro_diag/validation_trex_da_01_bt_dendro_diag.cpp)

# Sparse-PCA T-Rex selector probes (01 dropped: relied on removed CV API).
add_trex_diagnostic(validation_trex_spca_02_solver_comparison
    trex_selector_methods/trex_spca/validation_trex_spca_02_solver_comparison.cpp)
add_trex_diagnostic(validation_trex_spca_03_scaling_comparison
    trex_selector_methods/trex_spca/validation_trex_spca_03_scaling_comparison.cpp)
add_trex_diagnostic(validation_trex_spca_04_rdump_pipeline
    trex_selector_methods/trex_spca/validation_trex_spca_04_rdump_pipeline.cpp)
# Point spca_04 at the baked R dump committed beside the program (from the
# co-located demo_trex_spca_02.R generator); --dir still overrides.
target_compile_definitions(validation_trex_spca_04_rdump_pipeline PRIVATE
    TREX_SPCA_RDUMP_DIR="${CMAKE_CURRENT_SOURCE_DIR}/trex_selector_methods/trex_spca/rdump")
add_trex_diagnostic(validation_trex_spca_05_lambda2_foldmatch
    trex_selector_methods/trex_spca/validation_trex_spca_05_lambda2_foldmatch.cpp)
add_trex_diagnostic(validation_trex_spca_06_handrolled_comparison
    trex_selector_methods/trex_spca/validation_trex_spca_06_handrolled_comparison.cpp)
add_trex_diagnostic(validation_trex_spca_07_matrix_dump
    trex_selector_methods/trex_spca/validation_trex_spca_07_matrix_dump.cpp)
# Point spca_07 at the baked rdump10 committed beside the program; --dir overrides.
target_compile_definitions(validation_trex_spca_07_matrix_dump PRIVATE
    TREX_SPCA_RDUMP10_DIR="${CMAKE_CURRENT_SOURCE_DIR}/trex_selector_methods/trex_spca/rdump10")

# T-solver equivalence probe.
add_trex_diagnostic(validation_ts_01_tenet_aug_comparison
    tsolvers/validation_ts_01_tenet_aug_comparison/validation_ts_01_tenet_aug_comparison.cpp)

# Compile-only aggregate: builds every diagnostic (catching API drift against the
# library) without auto-running them. Run individual probes by hand from bin/.
add_custom_target(build_diagnostics
    DEPENDS validation_trex_01_scaling_comparison
            validation_trex_gvs_01_scaling_solver_comparison
            validation_trex_da_01_bt_dendro_diag
            validation_trex_spca_02_solver_comparison
            validation_trex_spca_03_scaling_comparison
            validation_trex_spca_04_rdump_pipeline
            validation_trex_spca_05_lambda2_foldmatch
            validation_trex_spca_06_handrolled_comparison
            validation_trex_spca_07_matrix_dump
            validation_ts_01_tenet_aug_comparison
    COMMENT "Built all migrated diagnostics -- run individually from build/debug/bin/ (most are multi-minute MC studies)."
    VERBATIM)
