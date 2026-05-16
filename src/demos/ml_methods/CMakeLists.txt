# CMakeLists.txt for ml_methods demos

add_demo_target(demo_mlm_scaler_01
    scaler_methods/demo_mlm_scaler_01.cpp
    DATA_PREPROCESSING_LIBS
)

add_demo_target(demo_mlm_hac_01
    hac/demo_mlm_hac_01.cpp
    DATA_PREPROCESSING_LIBS
)

add_demo_target(demo_mlm_hac_02_mmap
    hac/demo_mlm_hac_02_mmap.cpp
    DATA_PREPROCESSING_LIBS
)

target_compile_definitions(demo_mlm_hac_02_mmap PRIVATE
    "TREX_DEMO_MMAP_PATH=\"${PROJECT_SOURCE_DIR}/trex_mlm_demo_02_hac_mmap.bin\""
)

add_demo_target(demo_mlm_ms_01_ridge_gcv
    model_selection/demo_mlm_ms_01_ridge_gcv.cpp
    DATA_PREPROCESSING_LIBS
)
target_compile_definitions(demo_mlm_ms_01_ridge_gcv PRIVATE
    "TREX_DEMO_MMAP_PATH=\"${PROJECT_SOURCE_DIR}/trex_mlm_demo_ms01_gcv_mmap.bin\""
)

add_demo_target(demo_mlm_ms_02_ridge_cv
    model_selection/demo_mlm_ms_02_ridge_cv.cpp
    DATA_PREPROCESSING_LIBS
)
target_compile_definitions(demo_mlm_ms_02_ridge_cv PRIVATE
    "TREX_DEMO_MMAP_PATH=\"${PROJECT_SOURCE_DIR}/trex_mlm_demo_ms02_cv_mmap.bin\""
)
