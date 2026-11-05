# =============================================================================
# Unit tests for hierarchical agglomerative clustering (clustering.R)
#
# Test groups:
#   1. ml_agglomerative_cluster input validation
#   2. ml_cut_tree input validation
#   3. Output format contracts
#   4. Single linkage correctness  (mirrors SingleLinkageSciPyMatch in C++)
#   5. Average linkage correctness (mirrors AverageLinkageSciPyMatch in C++)
#   6. Centroid linkage correctness (mirrors CentroidLinkageSciPyMatch in C++)
#   7. ml_cut_tree correctness         (mirrors CutTree in C++)
#   8. End-to-end round-trip with hclust cross-check
#
# Index convention:
#   Columns 1-2 of the linkage matrix and ml_cut_tree labels use 1-based R indexing.
#   Original objects are numbered 1..N; merged cluster IDs start at N+1.
#   Distances are squared Euclidean (C++ engine native, = SciPy metric='sqeuclidean').
# =============================================================================

library(TRexSelector)

# Shared 5-point fixture: x = {0,2,5,10,16}, y = 0
# Matches the data used in the C++ dispatcher tests.
n_fix <- 5L
data_fix <- matrix(c(0, 0,
                     2, 0,
                     5, 0,
                     10, 0,
                     16, 0),
                   nrow = 5L, ncol = 2L, byrow = TRUE)


# =============================================================================
# Group 1: ml_agglomerative_cluster input validation
# =============================================================================

test_that("ml_agglomerative_cluster rejects non-matrix input", {
  expect_error(
    ml_agglomerative_cluster(list(1, 2, 3), ml_linkage_method$Single),
    "must be a numeric matrix"
  )
})

test_that("ml_agglomerative_cluster rejects non-numeric matrix", {
  expect_error(
    ml_agglomerative_cluster(matrix(c("a", "b", "c", "d"), 2, 2), ml_linkage_method$Single),
    "must be a numeric matrix"
  )
})


# =============================================================================
# Group 2: ml_cut_tree input validation
# =============================================================================

test_that("ml_cut_tree rejects non-matrix linkage", {
  expect_error(
    ml_cut_tree(list(1, 2, 3), 4L, 2L),
    "must be a numeric matrix"
  )
})

test_that("ml_cut_tree rejects num_clusters < 1", {
  lnk <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single)
  expect_error(
    ml_cut_tree(lnk, n_fix, 0L),
    "num_clusters must be > 0"
  )
})

test_that("ml_cut_tree rejects num_clusters > num_orig_objs", {
  lnk <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single)
  expect_error(
    ml_cut_tree(lnk, n_fix, n_fix + 1L),
    "num_clusters must be > 0"
  )
})


# =============================================================================
# Group 3: Output format contracts
# =============================================================================

test_that("ml_agglomerative_cluster returns (n-1) x 4 matrix", {
  lnk <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single)
  expect_true(is.matrix(lnk))
  expect_equal(nrow(lnk), n_fix - 1L)
  expect_equal(ncol(lnk), 4L)
})

test_that("distance column is non-decreasing", {
  lnk <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single)
  expect_true(all(diff(lnk[, 3]) >= 0))
})

test_that("new_size column is in [2, n]", {
  lnk <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single)
  expect_true(all(lnk[, 4] >= 2))
  expect_true(all(lnk[, 4] <= n_fix))
})

test_that("Ward rejects non-Euclidean metrics except Correlation_LSH_Approx", {
  expect_error(
    ml_agglomerative_cluster(data_fix, ml_linkage_method$Ward,
                          metric = ml_distance_metric$Correlation),
    "Euclidean"
  )
})

test_that("Ward + Correlation_LSH_Approx is permitted (SimHash-projected Ward)", {
  # Needs variation in both coordinates: rows with zero variance make the
  # correlation metric degenerate.
  set.seed(42)
  X <- matrix(rnorm(5L * 8L), nrow = 5L, ncol = 8L)
  lnk <- ml_agglomerative_cluster(X, ml_linkage_method$Ward,
                               metric = ml_distance_metric$Correlation_LSH_Approx)
  expect_true(is.matrix(lnk))
  expect_equal(nrow(lnk), 4L)
  expect_equal(ncol(lnk), 4L)
})


# =============================================================================
# Group 4: Single linkage correctness
# Mirrors C++ test SingleLinkageSciPyMatch (1-based IDs: +1 to each C++ cluster ID)
#
# C++ expected (0-based):            R expected (1-based):
#   {0,1}  dist=4   size=2  → ID 5    {1,2}  dist=4   size=2  → ID 6
#   {2,5}  dist=9   size=3  → ID 6    {3,6}  dist=9   size=3  → ID 7
#   {3,6}  dist=25  size=4  → ID 7    {4,7}  dist=25  size=4  → ID 8
#   {4,7}  dist=36  size=5  → ID 8    {5,8}  dist=36  size=5  → ID 9
# =============================================================================

test_that("single linkage produces correct merge sequence (1-based)", {
  lnk <- ml_agglomerative_cluster(data_fix,
                               method = ml_linkage_method$Single,
                               metric = ml_distance_metric$Euclidean)

  expect_equal(nrow(lnk), n_fix - 1L)

  expect_equal(lnk[1, 1], 1)
  expect_equal(lnk[1, 2], 2)
  expect_equal(lnk[1, 3], 4.0, tolerance = 1e-6)
  expect_equal(lnk[1, 4], 2)

  expect_equal(lnk[2, 1], 3)
  expect_equal(lnk[2, 2], 6)
  expect_equal(lnk[2, 3], 9.0, tolerance = 1e-6)
  expect_equal(lnk[2, 4], 3)

  expect_equal(lnk[3, 1], 4)
  expect_equal(lnk[3, 2], 7)
  expect_equal(lnk[3, 3], 25.0, tolerance = 1e-6)
  expect_equal(lnk[3, 4], 4)

  expect_equal(lnk[4, 1], 5)
  expect_equal(lnk[4, 2], 8)
  expect_equal(lnk[4, 3], 36.0, tolerance = 1e-6)
  expect_equal(lnk[4, 4], 5)
})


# =============================================================================
# Group 5: Average linkage correctness
# Mirrors C++ test AverageLinkageSciPyMatch (1-based IDs)
#
# C++ expected (0-based):              R expected (1-based):
#   {0,1}  dist=4    size=2  → ID 5     {1,2}  dist=4    size=2  → ID 6
#   {2,5}  dist=17   size=3  → ID 6     {3,6}  dist=17   size=3  → ID 7
#   {3,4}  dist=36   size=2  → ID 7     {4,5}  dist=36   size=2  → ID 8
#   {6,7}  dist=127  size=5  → ID 8     {7,8}  dist=127  size=5  → ID 9
# =============================================================================

test_that("average linkage produces correct merge sequence (1-based)", {
  lnk <- ml_agglomerative_cluster(data_fix,
                               method = ml_linkage_method$Average,
                               metric = ml_distance_metric$Euclidean)

  expect_equal(nrow(lnk), n_fix - 1L)

  expect_equal(lnk[1, 1], 1)
  expect_equal(lnk[1, 2], 2)
  expect_equal(lnk[1, 3], 4.0, tolerance = 1e-6)
  expect_equal(lnk[1, 4], 2)

  expect_equal(lnk[2, 1], 3)
  expect_equal(lnk[2, 2], 6)
  expect_equal(lnk[2, 3], 17.0, tolerance = 1e-6)
  expect_equal(lnk[2, 4], 3)

  expect_equal(lnk[3, 1], 4)
  expect_equal(lnk[3, 2], 5)
  expect_equal(lnk[3, 3], 36.0, tolerance = 1e-6)
  expect_equal(lnk[3, 4], 2)

  expect_equal(lnk[4, 1], 7)
  expect_equal(lnk[4, 2], 8)
  expect_equal(lnk[4, 3], 127.0, tolerance = 1e-6)
  expect_equal(lnk[4, 4], 5)
})


# =============================================================================
# Group 6: Centroid linkage correctness
# Mirrors C++ test CentroidLinkageSciPyMatch (1-based IDs)
#
# C++ expected (0-based):                     R expected (1-based):
#   {0,1}  dist=4          size=2  → ID 5      {1,2}  dist=4          size=2  → ID 6
#   {2,5}  dist=16         size=3  → ID 6      {3,6}  dist=16         size=3  → ID 7
#   {3,4}  dist=36         size=2  → ID 7      {4,5}  dist=36         size=2  → ID 8
#   {6,7}  dist≈113.7778   size=5  → ID 8      {7,8}  dist≈113.7778   size=5  → ID 9
# =============================================================================

test_that("centroid linkage produces correct merge sequence (1-based)", {
  lnk <- ml_agglomerative_cluster(data_fix,
                               method = ml_linkage_method$Centroid,
                               metric = ml_distance_metric$Euclidean)

  expect_equal(nrow(lnk), n_fix - 1L)

  expect_equal(lnk[1, 1], 1)
  expect_equal(lnk[1, 2], 2)
  expect_equal(lnk[1, 3], 4.0, tolerance = 1e-6)
  expect_equal(lnk[1, 4], 2)

  expect_equal(lnk[2, 1], 3)
  expect_equal(lnk[2, 2], 6)
  expect_equal(lnk[2, 3], 16.0, tolerance = 1e-6)
  expect_equal(lnk[2, 4], 3)

  expect_equal(lnk[3, 1], 4)
  expect_equal(lnk[3, 2], 5)
  expect_equal(lnk[3, 3], 36.0, tolerance = 1e-6)
  expect_equal(lnk[3, 4], 2)

  expect_equal(lnk[4, 1], 7)
  expect_equal(lnk[4, 2], 8)
  expect_equal(lnk[4, 3], 113.77777777777777, tolerance = 1e-4)
  expect_equal(lnk[4, 4], 5)
})


# =============================================================================
# Group 7: ml_cut_tree correctness
# Mirrors C++ test CutTree (DendrogramUtilsTest).
# Uses a manually constructed 4-object linkage with 1-based cluster IDs.
#
# Merge steps (1-based):
#   {1,2} at dist=2.0 → new cluster 5
#   {3,4} at dist=5.0 → new cluster 6
#   {5,6} at dist=8.0 → root (cluster of all 4)
#
# Cutting at k=2 keeps merges 1 and 2 (dist ≤ 5.0), so:
#   objects 1 and 2 share one label, objects 3 and 4 share another label.
# =============================================================================

test_that("ml_cut_tree partitions 4 objects into 2 correct groups (1-based)", {
  lnk <- matrix(c(1, 2, 2.0, 2,
                  3, 4, 5.0, 2,
                  5, 6, 8.0, 4),
                nrow = 3L, ncol = 4L, byrow = TRUE)
  labels <- ml_cut_tree(lnk, num_orig_objs = 4L, num_clusters = 2L)

  expect_equal(length(labels), 4L)
  expect_equal(labels[1], labels[2])
  expect_equal(labels[3], labels[4])
  expect_false(labels[1] == labels[3])
  expect_true(all(labels %in% c(1L, 2L)))
})

test_that("ml_cut_tree with k=1 assigns all objects the same label", {
  lnk <- matrix(c(1, 2, 2.0, 2,
                  3, 4, 5.0, 2,
                  5, 6, 8.0, 4),
                nrow = 3L, ncol = 4L, byrow = TRUE)
  labels <- ml_cut_tree(lnk, num_orig_objs = 4L, num_clusters = 1L)

  expect_equal(length(labels), 4L)
  expect_true(all(labels == labels[1]))
})

test_that("ml_cut_tree with k=n assigns each object a unique label", {
  lnk <- matrix(c(1, 2, 2.0, 2,
                  3, 4, 5.0, 2,
                  5, 6, 8.0, 4),
                nrow = 3L, ncol = 4L, byrow = TRUE)
  labels <- ml_cut_tree(lnk, num_orig_objs = 4L, num_clusters = 4L)

  expect_equal(length(labels), 4L)
  expect_equal(length(unique(labels)), 4L)
})


# =============================================================================
# Group 8: End-to-end round-trip + independent hclust cross-check
#
# Runs single linkage on the 5-point fixture, cuts to 2 clusters, and checks:
#   (a) The partition geometry: objects 1-3 together, objects 4-5 together.
#   (b) An independent reference from R's hclust using squared Euclidean distances.
#       hclust(d_sq, "single") is topologically equivalent to our engine
#       (monotone distance transform preserves single-linkage partition).
# =============================================================================

test_that("single linkage cut to 2 clusters gives correct partition (1-based)", {
  lnk <- ml_agglomerative_cluster(data_fix,
                               method = ml_linkage_method$Single,
                               metric = ml_distance_metric$Euclidean)
  labels <- ml_cut_tree(lnk, num_orig_objs = n_fix, num_clusters = 2L)

  expect_equal(length(labels), n_fix)
  expect_true(all(labels %in% c(1L, 2L)))

  # From the single-linkage merge sequence (1-based, squared Euclidean):
  #   Step 1: {1,2} dist=4  → ID 6
  #   Step 2: {3,6} dist=9  → ID 7   (object 3 joins {1,2})
  #   Step 3: {4,7} dist=25 → ID 8   (object 4 joins {1,2,3})
  #   Step 4: {5,8} dist=36 → ID 9   (object 5 joins {1,2,3,4})
  # Cutting at k=2 removes the last merge (step 4), yielding {1,2,3,4} and {5}.
  expect_equal(labels[1], labels[2])
  expect_equal(labels[2], labels[3])
  expect_equal(labels[3], labels[4])
  expect_false(labels[1] == labels[5])
})


test_that("single linkage partition matches hclust reference (squared Euclidean)", {
  lnk <- ml_agglomerative_cluster(data_fix,
                               method = ml_linkage_method$Single,
                               metric = ml_distance_metric$Euclidean)
  ours <- ml_cut_tree(lnk, num_orig_objs = n_fix, num_clusters = 2L)

  # Build squared Euclidean distance matrix matching the C++ engine
  d_sq <- as.dist(outer(data_fix[, 1], data_fix[, 1], function(a, b) (a - b)^2 + (0 - 0)^2))
  ref <- cutree(hclust(d_sq, method = "single"), k = 2L)

  # Both are 1-based; check partition topology (same-group iff same label)
  for (i in seq_len(n_fix)) {
    for (j in seq_len(n_fix)) {
      expect_equal(
        ours[i] == ours[j],
        ref[i]  == ref[j],
        label = paste0("partition agreement for objects ", i, " and ", j)
      )
    }
  }
})


# =============================================================================
# Group 9: use_mmap=TRUE produces numerically identical results to use_mmap=FALSE
# =============================================================================

test_that("ml_agglomerative_cluster use_mmap=TRUE matches use_mmap=FALSE (Single linkage)", {
  lnk_ram  <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single,  use_mmap = FALSE)
  lnk_mmap <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single,  use_mmap = TRUE)
  expect_equal(lnk_ram, lnk_mmap)
})

test_that("ml_agglomerative_cluster use_mmap=TRUE matches use_mmap=FALSE (Average linkage)", {
  lnk_ram  <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Average, use_mmap = FALSE)
  lnk_mmap <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Average, use_mmap = TRUE)
  expect_equal(lnk_ram, lnk_mmap)
})

test_that("ml_agglomerative_cluster use_mmap=TRUE matches use_mmap=FALSE (Centroid linkage)", {
  lnk_ram  <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Centroid, use_mmap = FALSE)
  lnk_mmap <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Centroid, use_mmap = TRUE)
  expect_equal(lnk_ram, lnk_mmap)
})

test_that("ml_cut_tree round-trip works on mmap-derived linkage", {
  lnk <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single, use_mmap = TRUE)
  clusters <- ml_cut_tree(lnk, num_orig_objs = n_fix, num_clusters = 2L)
  expect_length(clusters, n_fix)
  expect_equal(length(unique(clusters)), 2L)
})


# =============================================================================
# Group 8: Input guards — the C++ core rejects non-finite
# data and malformed linkage matrices with a translatable error instead of
# indexing out of range.
# =============================================================================

test_that("ml_agglomerative_cluster rejects non-finite input loudly", {
  bad <- data_fix
  bad[2, 1] <- NA_real_
  expect_error(ml_agglomerative_cluster(bad, ml_linkage_method$Single), "non-finite")
  bad[2, 1] <- Inf
  expect_error(ml_agglomerative_cluster(bad, ml_linkage_method$Average,
                                     ml_distance_metric$Correlation),
               "non-finite")
})

test_that("ml_cut_tree rejects malformed linkage matrices", {
  lnk <- ml_agglomerative_cluster(data_fix, ml_linkage_method$Single)

  # Too few rows for the requested cut (K = 1 needs n - 1 = 4 merges).
  expect_error(ml_cut_tree(lnk[1:2, , drop = FALSE], n_fix, 1L), "rows")
  # ... while the same prefix still cuts into 3 clusters (2 merges).
  expect_length(ml_cut_tree(lnk[1:2, , drop = FALSE], n_fix, 3L), n_fix)

  # A cluster id that does not exist yet at its step (1-based n + 1 in row 1).
  bad <- lnk
  bad[1, 2] <- n_fix + 1
  expect_error(ml_cut_tree(bad, n_fix, 2L), "does not exist")

  # A self-merge.
  bad <- lnk
  bad[1, 2] <- bad[1, 1]
  expect_error(ml_cut_tree(bad, n_fix, 2L), "itself")

  # An id consumed twice (object 1 was merged in row 1).
  bad <- lnk
  bad[2, 1] <- lnk[1, 1]
  expect_error(ml_cut_tree(bad, n_fix, 2L), "already consumed")
})

test_that("ml_agglomerative_cluster is silent by default and reports its backend on request", {
  expect_silent(ml_agglomerative_cluster(data_fix, ml_linkage_method$Ward))
  expect_output(ml_agglomerative_cluster(data_fix, ml_linkage_method$Ward, verbose = TRUE),
                "allocated")
})
