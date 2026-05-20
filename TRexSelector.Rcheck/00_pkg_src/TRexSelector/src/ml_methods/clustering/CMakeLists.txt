# src/ml_methods/clustering/CMakeLists.txt
# ==================================================================================
# CMake configuration for the clustering module of ml_methods.
#
# This module provides unsupervised learning algorithms, including hierarchical
# dendrograms and partitional clustering (K-Means) and future extensions,
# for the T-Rex framework.
#
# Organization:
# -------------
# src/ml_methods/
# |-- clustering/
#     |-- hierarchical/
#         |-- agglomerative/
#             |-- cluster_update_policy.hpp
#             |-- Condensed_Cluster_Update_Policy.hpp
#             |-- dendrogram_utils.hpp
#             |-- generic_linkage_core.hpp
#             |-- mst_slink_core.hpp
#             |-- nnchain_core.hpp
#             |-- slink_core.hpp
#             |-- deprecated/ -> deprecated files for reference
#
#     |-- kmeans/
#         |-- (placeholder for future K-Means files or alike)
# ==================================================================================


# ==================================================================================
# 1. Append Source Files
# ==================================================================================
# (Currently empty, as hierarchical clustering is a header-only template library.
# Future .cpp files for clustering should be added here as PRIVATE).
#
# Example:
# --------
# target_sources(trex_ml_methods PRIVATE
#     ${CMAKE_CURRENT_LIST_DIR}/kmeans/KMeans_core.cpp
# )
# ==================================================================================


# ==================================================================================
# 2. Append Header Files
# ==================================================================================
# Append header-only implementations to the parent library target for IDE visibility.
# Marked PUBLIC so that CMake tools and IDEs index them correctly.
target_sources(trex_ml_methods PRIVATE
    # --- Hierarchical Clustering (Agglomerative) ---
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/agglomerative_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/agglomerative_dispatcher.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/block_tiled_matrix_policy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/proj_geom_updates.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/dendrogram_utils.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/distance_policy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/generic_linkage_core.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/nnchain_core.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical/agglomerative/slink_core.hpp

    # --- K-Means ---
    # ${CMAKE_CURRENT_LIST_DIR}/kmeans/KMeans_Init.hpp
)
# ==================================================================================
