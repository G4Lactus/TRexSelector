# ==============================================================================
# diag_bt_dendro_compare.R
# ==============================================================================
#
# R side of the single-dataset DA-BT diagnostic (pairs with the C++ validation
# validation_trex_da_01_bt_dendro_diag.cpp).
#
# Reads the C++-exported design matrix (diag_X.csv), response (diag_y.csv) and
# true support (diag_support0.csv, 0-based) from this validation's
# validation_results/ folder, then reproduces the R DA-BT pipeline on the
# IDENTICAL X:
#
#   1. cor(X) -> as.dist(1 - |cor|) -> hclust(method = "single")  [heights]
#   2. rho_grid = c(1 - rev(height), 1)[round(seq(1, p, length.out = L))]
#   3. cutree(dendro, h = 1 - rho_grid)                           [group sizes]
#   4. trex(method = "trex+DA+BT", hc_dist = "single")            [selection]
#
# It then diffs the R dendrogram heights and per-level support group sizes
# against the C++ dumps (diag_cpp_heights.csv / diag_cpp_groups.csv) and prints
# a PASS/FAIL summary that tells us whether the divergence is in the CLUSTERING
# or DOWNSTREAM.
#
# Run:  Rscript diag_bt_dendro_compare.R
# ==============================================================================

suppressPackageStartupMessages(library(TRexSelector))

# ------------------------------------------------------------------------------
# Locate this script's directory, then its validation_results/ folder.
# ------------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  # Fallback for source()/Rscript-less invocation.
  d <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                error = function(e) NA_character_)
  if (!is.na(d)) return(d)
  getwd()
}

this_dir <- get_script_dir()
res_dir  <- file.path(this_dir, "validation_results")
stopifnot(dir.exists(res_dir))

fp <- function(name) file.path(res_dir, name)

# ------------------------------------------------------------------------------
# Load the shared dataset exported by the C++ demo.
# ------------------------------------------------------------------------------
X        <- as.matrix(utils::read.csv(fp("diag_X.csv"), header = FALSE))
y        <- scan(fp("diag_y.csv"), quiet = TRUE)
support0 <- scan(fp("diag_support0.csv"), quiet = TRUE)   # 0-based
support  <- as.integer(support0) + 1L                     # 1-based for R
p        <- ncol(X)
n        <- nrow(X)
storage.mode(X) <- "double"
dimnames(X) <- NULL

cat(sprintf("Loaded X: %d x %d ; support (1-based): %s\n",
            n, p, paste(support, collapse = " ")))

# ------------------------------------------------------------------------------
# 1. Clustering: reproduce the R DA-BT dendrogram.
# ------------------------------------------------------------------------------
cor_mat  <- stats::cor(X)
dmat     <- stats::as.dist(1 - abs(cor_mat))
dendro   <- stats::hclust(dmat, method = "single")

r_heights <- dendro$height  # length p-1, ascending
utils::write.csv(
  data.frame(merge = seq_along(r_heights), height = r_heights),
  fp("diag_r_heights.csv"), row.names = FALSE)

# ------------------------------------------------------------------------------
# 2. rho_grid (exactly as TRexSelector::trex for method "trex+DA+BT").
# ------------------------------------------------------------------------------
L        <- min(20L, p)
sub      <- round(seq(1, p, length.out = L))
rho_grid <- c(1 - rev(dendro$height), 1)[sub]

# ------------------------------------------------------------------------------
# 3. Cut the tree at h = 1 - rho for every level; record group sizes/members.
# ------------------------------------------------------------------------------
clusters <- stats::cutree(dendro, h = 1 - rho_grid)  # p x L
rows <- list()
for (r in seq_len(L)) {
  labs <- clusters[, r]
  for (j in seq_len(p)) {
    members0 <- which(labs == labs[j]) - 1L            # 0-based to match C++
    rows[[length(rows) + 1L]] <- data.frame(
      rho_level     = r - 1L,                           # 0-based level
      rho           = rho_grid[r],
      cut_height    = 1 - rho_grid[r],
      var           = j - 1L,                           # 0-based var
      group_size    = length(members0),
      group_members = paste(members0, collapse = "|"),
      stringsAsFactors = FALSE)
  }
}
r_groups <- do.call(rbind, rows)
utils::write.csv(r_groups, fp("diag_r_groups.csv"), row.names = FALSE)

# ------------------------------------------------------------------------------
# 4. Full R trex run on the same X, y (sanity + downstream reference).
# ------------------------------------------------------------------------------
r_sel <- tryCatch({
  fit <- TRexSelector::trex(
    X = X, y = y, tFDR = 0.2, K = 20L,
    method = "trex+DA+BT", hc_dist = "single",
    verbose = FALSE, seed = 2026L)
  beta <- numeric(p); beta[support] <- 1
  list(
    selected = which(fit$selected_var > 0) - 1L,        # 0-based
    FDP = TRexSelector::FDP(fit$selected_var, beta),
    TPP = TRexSelector::TPP(fit$selected_var, beta),
    T_stop = fit$T_stop,
    num_dummies = fit$num_dummies)
}, error = function(e) { cat("trex() failed:", conditionMessage(e), "\n"); NULL })

if (!is.null(r_sel)) {
  cat(sprintf("R trex+DA+BT: T_stop=%d  num_dummies=%d (L=%g)  #selected=%d  FDP=%.4f  TPP=%.4f  selected(0-based): %s\n",
              r_sel$T_stop, r_sel$num_dummies, r_sel$num_dummies / p,
              length(r_sel$selected), r_sel$FDP, r_sel$TPP,
              paste(r_sel$selected, collapse = " ")))
  utils::write.csv(
    data.frame(
      key   = c("T_stop", "num_dummies", "L", "n_selected", "FDP", "TPP",
                "selected_0based"),
      value = c(r_sel$T_stop, r_sel$num_dummies, r_sel$num_dummies / p,
                length(r_sel$selected), r_sel$FDP, r_sel$TPP,
                paste(r_sel$selected, collapse = "|")),
      stringsAsFactors = FALSE),
    fp("diag_r_selector.csv"), row.names = FALSE)
}

# ==============================================================================
# COMPARISON vs C++ dumps
# ==============================================================================
cat("\n", strrep("=", 70), "\n  C++ vs R comparison\n", strrep("=", 70), "\n",
    sep = "")

# --- Heights ---
cpp_h_path <- fp("diag_cpp_heights.csv")
if (file.exists(cpp_h_path)) {
  cpp_h <- utils::read.csv(cpp_h_path)
  cpp_heights <- sort(cpp_h$distance)
  r_sorted    <- sort(r_heights)
  if (length(cpp_heights) == length(r_sorted)) {
    max_abs <- max(abs(cpp_heights - r_sorted))
    cat(sprintf("\n[Heights] %d merges. max|C++ - R| (sorted) = %.3e -> %s\n",
                length(r_sorted), max_abs,
                if (max_abs < 1e-6) "MATCH" else "DIFFER"))
  } else {
    cat(sprintf("\n[Heights] length mismatch: C++ %d vs R %d\n",
                length(cpp_heights), length(r_sorted)))
  }
} else {
  cat("\n[Heights] diag_cpp_heights.csv not found — run the C++ demo first.\n")
}

# --- Per-level support group sizes ---
cpp_g_path <- fp("diag_cpp_groups.csv")
if (file.exists(cpp_g_path)) {
  cpp_g <- utils::read.csv(cpp_g_path)
  cat("\n[Group sizes @ support vars]  level : rho :  C++ sizes  |  R sizes\n")
  n_mismatch <- 0L
  for (r in seq_len(L)) {
    lvl <- r - 1L
    cpp_sizes <- vapply(support0, function(s) {
      v <- cpp_g$group_size[cpp_g$rho_level == lvl & cpp_g$var == s]
      if (length(v)) v[1] else NA_integer_
    }, numeric(1))
    r_sizes <- vapply(support0, function(s) {
      v <- r_groups$group_size[r_groups$rho_level == lvl & r_groups$var == s]
      if (length(v)) v[1] else NA_integer_
    }, numeric(1))
    flag <- if (!isTRUE(all.equal(cpp_sizes, r_sizes))) { n_mismatch <- n_mismatch + 1L; " <== DIFF" } else ""
    cat(sprintf("  %2d : %.4f :  %s  |  %s%s\n", lvl, rho_grid[r],
                paste(sprintf("%2d", cpp_sizes), collapse = " "),
                paste(sprintf("%2d", r_sizes),   collapse = " "), flag))
  }
  cat(sprintf("\n[Group sizes] %d / %d rho-levels differ at support vars -> %s\n",
              n_mismatch, L, if (n_mismatch == 0L) "MATCH" else "DIFFER"))
} else {
  cat("\n[Group sizes] diag_cpp_groups.csv not found — run the C++ demo first.\n")
}

# --- Selector summary side-by-side ---
cpp_s_path <- fp("diag_cpp_selector.csv")
if (file.exists(cpp_s_path)) {
  cpp_s <- utils::read.csv(cpp_s_path, stringsAsFactors = FALSE)
  getv <- function(k) cpp_s$value[cpp_s$key == k]
  cat("\n[Selector]\n")
  cat(sprintf("  C++ : T_stop=%s  L=%s  rho_thresh=%s  #sel=%s  FDP=%s  TPP=%s\n",
              getv("T_stop"), getv("L"), getv("rho_thresh"),
              getv("n_selected"), getv("FDP"), getv("TPP")))
  if (!is.null(r_sel)) {
    cat(sprintf("  R   : T_stop=%d  L=%g  (rho_thresh not exposed)  #sel=%d  FDP=%.4f  TPP=%.4f\n",
                r_sel$T_stop, r_sel$num_dummies / p,
                length(r_sel$selected), r_sel$FDP, r_sel$TPP))
  }
}

cat("\nDone. Artifacts in:", res_dir, "\n")
