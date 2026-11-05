# =============================================================================
# utils.R - Utility functions for TRexSelector
# =============================================================================

#' @importFrom Rcpp evalCpp
NULL


# ---------------------------------------------------------------------------
# Internal: enforce double (REALSXP) storage before passing to the C++ glue.
#
# The zero-copy selector/PCA endpoints take the R buffer directly and, for
# selectors, write the de-normalized matrix back into it on destruction. They
# therefore require a REALSXP buffer: an integer matrix would otherwise be
# silently coerced to a throwaway double copy inside Rcpp, so the write-back
# (and any in-place normalization) would land on the copy instead of the
# caller's matrix. We coerce here, visibly and once, rather than let that
# happen invisibly in C++. A matrix already stored as double is passed through
# untouched (no copy), so large double inputs stay zero-copy.
# ---------------------------------------------------------------------------

.as_double_matrix <- function(X, arg = "X") {
  if (!is.matrix(X)) X <- as.matrix(X)
  if (storage.mode(X) != "double") storage.mode(X) <- "double"
  X
}

.as_double_vector <- function(y, arg = "y") {
  if (storage.mode(y) != "double") storage.mode(y) <- "double"
  y
}


#' @title Apply the T-Rex data normalization to a design
#'
#' @description The selectors' own preprocessing as a user-callable helper:
#'   the same code path a selector runs at construction for the given
#'   \code{scaling_mode}. Use it to pre-normalize a design and then run a
#'   selector with \code{trex_control(scaling_mode = "NONE")} (the selection
#'   is identical to running the raw design under \code{"L2"}), to inspect
#'   the centering and scaling parameters, or to validate a design before a
#'   long run. The inputs are copied and left untouched.
#'
#' @param X Numeric design matrix (n x p).
#' @param y Numeric response vector (length n).
#' @param scaling_mode One of \code{"L2"} (default; center, divide by the
#'   column L2 norm), \code{"ZSCORE"} (center, divide by the sample SD) or
#'   \code{"NONE"} (copies equal the inputs; identity parameters).
#' @param eps Dead-column threshold on the working L2 norm (default 1e-6,
#'   the selectors' default solver tolerance).
#' @param n_threads Integer, the thread budget of the column passes
#'   (\code{NULL}, the default: the library's default budget,
#'   \code{\link{trex_get_num_threads}()}, initially 1).
#'
#' @return A list with \code{X} and \code{y} (the normalized copies),
#'   \code{X_means}, \code{X_scales} (the per-column divisors: L2 norms for
#'   \code{"L2"}, sample SDs for \code{"ZSCORE"}, ones for \code{"NONE"}) and
#'   \code{y_mean}.
#'
#' @details Errors on a non-finite entry or a dead column (constant under
#'   \code{"L2"} / \code{"ZSCORE"}, all-zero under \code{"NONE"}), naming
#'   the column (0-based, as reported by the C++ core).
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(200), 40, 5)
#' y <- X[, 1] + rnorm(40)
#' d <- trex_normalize_design(X, y)
#' colMeans(d$X)            # ~ 0
#' sqrt(colSums(d$X^2))     # 1
#'
#' @family T-Rex selectors and controls
#' @export
trex_normalize_design <- function(X, y, scaling_mode = c("L2", "ZSCORE", "NONE"),
                                  eps = 1e-6, n_threads = NULL) {
  scaling_mode <- match.arg(scaling_mode)
  X <- .as_double_matrix(X)
  y <- .as_double_vector(y)
  rcpp_normalize_design(X, y, scaling_mode, eps, resolve_n_threads(n_threads))
}


#' @title Convert R Matrix to System Memory Mapped Matrix
#'
#' @description Writes a numeric matrix to a binary memory-mapped file on disk and returns
#'              a \code{MemoryMappedMatrix} handle pointing to it.
#'
#' @param mat A \code{numeric} \code{matrix}.
#' @param filename String destination file path.
#'
#' @return A \code{MemoryMappedMatrix} / \code{mmap_matrix} object backed by \code{filename}.
#'
#' @references
#' F. Scheidt, J. Machkour, and M. Muma, "Solving FDR-Controlled Sparse Regression Problems with Five Million
#'   Variables on a Laptop," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt. Process.
#'   (CAMSAP), 2023, pp. 116-120.
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mm <- mem_matrix_from(mat, f)
#' print(mm)
#' }
#'
#' @family memory-mapped matrices
#' @export
mem_matrix_from <- function(mat, filename) {
  if (!is.matrix(mat) || !is.numeric(mat)) {
    stop("Input must be a numeric matrix.")
  }
  # An integer matrix is written as double: coerce explicitly (one copy in R)
  # rather than letting the binding make a hidden one.
  if (storage.mode(mat) != "double") storage.mode(mat) <- "double"
  if (!is.character(filename) || length(filename) != 1) {
    stop("Filename must be a single string path.")
  }

  convert_to_memory_mapped_matrix(mat, filename)
  mem_matrix(filename, nrow(mat), ncol(mat))
}


#' @title Memory-Mapped Matrix Object
#'
#' @description S3 object wrapping an external pointer to a Cpp MemoryMappedMatrix.
#'
#' @param filename Path to binary mmap file
#' @param rows Number of rows
#' @param cols Number of columns
#' @param mode Access mode: 'readonly' or 'readwrite' (default)
#'
#' @return Object of classes \code{MemoryMappedMatrix} and \code{mmap_matrix} (S3 XPtr wrapper)
#'
#' @references
#' F. Scheidt, J. Machkour, and M. Muma, "Solving FDR-Controlled Sparse Regression Problems with Five Million
#'   Variables on a Laptop," in Proc. IEEE 9th Int. Workshop Comput. Adv. Multi-Sensor Adapt. Process.
#'   (CAMSAP), 2023, pp. 116-120.
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mem_matrix_from(mat, f)
#' mm <- mem_matrix(f, rows = 4, cols = 5)
#' }
#'
#' @family memory-mapped matrices
#' @export
mem_matrix <- function(filename, rows, cols, mode = "readwrite") {
  if (!file.exists(filename) && mode == "readonly") {
    stop("Cannot open a non-existent file in readonly mode.")
  }

  mode_idx <- if (mode == "readonly") 0 else 1
  ptr <- mmap_matrix_create(filename, rows, cols, mode_idx)

  # Attach metadata and class
  attr(ptr, "filename") <- filename
  attr(ptr, "mode") <- mode
  class(ptr) <- c("MemoryMappedMatrix", "mmap_matrix", "externalptr")

  return(ptr)
}


#' @title Get the External Pointer of a MemoryMappedMatrix
#'
#' @param x A \code{MemoryMappedMatrix} / \code{mmap_matrix} object.
#'
#' @param ... unused arguments
#'
#' @return The underlying external pointer (the object itself).
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mem_matrix_from(mat, f)
#' mm <- mem_matrix(f, rows = 4, cols = 5)
#' get_ptr(mm)
#' }
#'
#' @noRd
get_ptr <- function(x, ...) UseMethod("get_ptr")

#' @export
get_ptr.mmap_matrix <- function(x, ...) x


#' @title Print mmap_matrix
#'
#' @param x mmap_matrix object
#' @param ... arguments passed to other methods
#'
#' @return invisibly returns x
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mem_matrix_from(mat, f)
#' mm <- mem_matrix(f, rows = 4, cols = 5)
#' print(mm)
#' }
#'
#' @family memory-mapped matrices
#' @export
print.mmap_matrix <- function(x, ...) {
  cat(sprintf("MemoryMappedMatrix [%d x %d]\n",
              mmap_matrix_rows(x), mmap_matrix_cols(x)))
  cat(sprintf("  File: %s\n", attr(x, "filename")))
  cat(sprintf("  Mode: %s\n", attr(x, "mode")))
  invisible(x)
}


#' @title Get mmap_matrix dimensions
#'
#' @param x mmap_matrix object
#'
#' @return an integer vector of length 2
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mem_matrix_from(mat, f)
#' mm <- mem_matrix(f, rows = 4, cols = 5)
#' dim(mm)
#' }
#' @family memory-mapped matrices
#' @export
dim.mmap_matrix <- function(x) {
  c(mmap_matrix_rows(x), mmap_matrix_cols(x))
}


#' @title Extract or assign blocks in an mmap_matrix
#'
#' @aliases [<-.mmap_matrix
#'
#' @param x mmap_matrix object
#' @param i row indices
#' @param j column indices
#' @param ... unused arguments
#' @param drop logical. If TRUE the result is coerced to the lowest possible dimension
#' (not supported).
#'
#' @return For extraction: a matrix block. For assignment: \code{x} invisibly.
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mem_matrix_from(mat, f)
#' mm <- mem_matrix(f, rows = 4, cols = 5)
#' mm[1:2, 1:3]
#' mm[2, 3] <- 99.0
#' mm[1:2, 1:3] <- matrix(as.double(1:6), nrow = 2, ncol = 3)
#' }
#' @usage \method{[}{mmap_matrix}(x, i, j, ..., drop = FALSE)
#' @family memory-mapped matrices
#' @export
`[.mmap_matrix` <- function(x, i, j, ..., drop = FALSE) {
  rows <- mmap_matrix_rows(x)
  cols <- mmap_matrix_cols(x)

  # Parse row index range
  if (missing(i)) {
    i_start <- 0
    i_count <- rows
  } else {
    i <- as.integer(i)
    if (any(i <= 0 | i > rows)) stop("Row indices out of bounds")
    # For now, we only support contiguous sequences to match Cpp block API
    if (length(i) > 1 && !all(diff(i) == 1)) {
      stop("Non-contiguous row slices are not currently supported by mmap_matrix")
    }
    i_start <- i[1] - 1
    i_count <- length(i)
  }

  # Parse col index range
  if (missing(j)) {
    j_start <- 0
    j_count <- cols
  } else {
    j <- as.integer(j)
    if (any(j <= 0 | j > cols)) stop("Col indices out of bounds")
    if (length(j) > 1 && !all(diff(j) == 1)) {
      stop("Non-contiguous col slices are not currently supported by mmap_matrix")
    }
    j_start <- j[1] - 1
    j_count <- length(j)
  }

  # Scalar path: return a plain double, not a 1x1 matrix
  if (i_count == 1L && j_count == 1L) {
    return(mmap_matrix_get_element(x, i_start, j_start))
  }

  mmap_matrix_read_range(x, i_start, i_count, j_start, j_count)
}


#' @title Assign elements or blocks in an mmap_matrix
#'
#' @description Writes a scalar or a numeric matrix block into the memory-mapped file.
#'   The object must be open in \code{"readwrite"} mode.
#'
#' @param x mmap_matrix object (must be in \code{"readwrite"} mode)
#' @param i Row index or contiguous range (1-based)
#' @param j Column index or contiguous range (1-based)
#' @param value A scalar or numeric matrix whose dimensions match the selected block
#'
#' @return \code{x} invisibly (required by R's replacement function protocol)
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mm <- mem_matrix_from(mat, f)
#' mm[2, 3] <- 99.0
#' mm[1:2, 1:3] <- matrix(as.double(1:6), nrow = 2, ncol = 3)
#' }
#'
#' @noRd
#' @method "[<-" mmap_matrix
#' @export
`[<-.mmap_matrix` <- function(x, i, j, value) {
  if (attr(x, "mode") != "readwrite") {
    stop("Cannot write to a read-only mmap_matrix.")
  }

  rows <- mmap_matrix_rows(x)
  cols <- mmap_matrix_cols(x)

  # Normalize row indices
  if (missing(i)) {
    i_start <- 0L
    i_count <- rows
  } else {
    i <- as.integer(i)
    if (any(i <= 0L | i > rows)) stop("Row indices out of bounds")
    if (length(i) > 1L && !all(diff(i) == 1L)) {
      stop("Non-contiguous row slices are not currently supported by mmap_matrix")
    }
    i_start <- i[1L] - 1L
    i_count <- length(i)
  }

  # Normalize column indices
  if (missing(j)) {
    j_start <- 0L
    j_count <- cols
  } else {
    j <- as.integer(j)
    if (any(j <= 0L | j > cols)) stop("Column indices out of bounds")
    if (length(j) > 1L && !all(diff(j) == 1L)) {
      stop("Non-contiguous col slices are not currently supported by mmap_matrix")
    }
    j_start <- j[1L] - 1L
    j_count <- length(j)
  }

  # Dispatch: single element or block
  if (i_count == 1L && j_count == 1L) {
    mmap_matrix_set_element(x, i_start, j_start, as.double(value))
  } else {
    value <- matrix(as.double(value), nrow = i_count, ncol = j_count)
    mmap_matrix_write_range(x, i_start, i_count, j_start, j_count, value)
  }

  invisible(x)
}


#' @title Get Zero-Copy View as Numeric Vector (Block)
#'
#' @param x mmap_matrix object
#' @param ... extra arguments including row_range and col_range
#'
#' @return Numeric matrix (copied)
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mem_matrix_from(mat, f)
#' mm <- mem_matrix(f, rows = 4, cols = 5)
#' as.matrix(mm)
#' }
#'
#' @family memory-mapped matrices
#' @export
as.matrix.mmap_matrix <- function(x, ...) {
  args <- list(...)
  row_range <- if ("row_range" %in% names(args)) args[["row_range"]] else NULL
  col_range <- if ("col_range" %in% names(args)) args[["col_range"]] else NULL
  rows <- mmap_matrix_rows(x)
  cols <- mmap_matrix_cols(x)

  if (is.null(row_range)) {
    r_s <- 0
    r_c <- rows
  } else {
    r_s <- row_range[1] - 1
    r_c <- row_range[2] - r_s
  }

  if (is.null(col_range)) {
    c_s <- 0
    c_c <- cols
  } else {
    c_s <- col_range[1] - 1
    c_c <- col_range[2] - c_s
  }

  mmap_matrix_read_range(x, r_s, r_c, c_s, c_c)
}


# ========================================================================================
# Evaluation Metrics
# ========================================================================================

#' @title Compute False Discovery Proportion (FDP) using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#'
#' @return Numeric FDP value.
#'
#' @examples
#' trex_fdp(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @family T-Rex selectors and controls
#' @export
trex_fdp <- function(selected_indices, true_support) {
  if (length(selected_indices) == 0 || length(true_support) == 0) return(0.0)
  rcpp_compute_fdp(as.integer(selected_indices), as.integer(true_support))
}


#' @title Compute True Positive Proportion (TPP) using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#'
#' @return Numeric TPP value.
#'
#' @examples
#' trex_tpp(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @family T-Rex selectors and controls
#' @export
trex_tpp <- function(selected_indices, true_support) {
  if (length(selected_indices) == 0 || length(true_support) == 0) return(0.0)
  rcpp_compute_tpp(as.integer(selected_indices), as.integer(true_support))
}


#' @title Compute Precision using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#'
#' @return Numeric precision value.
#'
#' @examples
#' trex_precision(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @family T-Rex selectors and controls
#' @export
trex_precision <- function(selected_indices, true_support) {
  if (length(selected_indices) == 0) return(0.0)
  if (length(true_support) == 0) return(0.0)
  rcpp_compute_precision(as.integer(selected_indices), as.integer(true_support))
}


#' @title Compute Recall using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#'
#' @return Numeric recall value.
#'
#' @examples
#' trex_recall(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @family T-Rex selectors and controls
#' @export
trex_recall <- function(selected_indices, true_support) {
  if (length(true_support) == 0) return(0.0)
  if (length(selected_indices) == 0) return(0.0)
  rcpp_compute_recall(as.integer(selected_indices), as.integer(true_support))
}


#' Compute False Discovery Proportion (FDP) using dense vectors
#'
#' @param beta_hat Numeric vector of estimated coefficients.
#' @param beta Numeric vector of true coefficients.
#' @param eps Numeric threshold for counting non-zeros (default: 1e-15).
#' @return Numeric FDP value.
#' @examples
#' trex_fdp_dense(c(1.5, 0, 0.8, 0, 0), c(2.0, 0, 1.0, 0, 0))
#' @family T-Rex selectors and controls
#' @export
trex_fdp_dense <- function(beta_hat, beta, eps = 1e-15) {
  rcpp_compute_fdp_dense(as.numeric(beta_hat), as.numeric(beta), eps)
}


#' @title Compute True Positive Proportion (TPP) using dense vectors
#'
#' @param beta_hat Numeric vector of estimated coefficients.
#' @param beta Numeric vector of true coefficients.
#' @param eps Numeric threshold for counting non-zeros (default: 1e-15).
#'
#' @return Numeric TPP value.
#'
#' @examples
#' trex_tpp_dense(c(1.5, 0, 0.8, 0, 0), c(2.0, 0, 1.0, 0, 0))
#'
#' @family T-Rex selectors and controls
#' @export
trex_tpp_dense <- function(beta_hat, beta, eps = 1e-15) {
  rcpp_compute_tpp_dense(as.numeric(beta_hat), as.numeric(beta), eps)
}


#' The library's thread budget
#'
#' @description The library is single-threaded by default. Every parallel
#'   region of the C++ backend runs on an explicit budget: a selector's
#'   random experiments and its inner kernels on \code{trex_control(
#'   max_outer_threads, max_inner_threads)}, a T-solver on
#'   \code{tsolver_options(n_threads)}, and every method without an option
#'   of its own (the scalers, the clustering, the ML regressions, the
#'   dummy and data generators, \code{\link{trex_normalize_design}}) on the
#'   library's default budget, which starts at 1.
#'   \code{trex_get_num_threads()} reads that default budget and
#'   \code{trex_set_num_threads()} sets it for this session; the
#'   \code{n_threads} argument of those methods overrides it per object.
#'   \code{trex_get_max_threads()} reports the threads OpenMP can run on this
#'   machine (1 in a build without OpenMP), an upper bound for a budget.
#'
#' @param n Positive integer, the default thread budget.
#' @return \code{trex_get_num_threads()}, \code{trex_get_max_threads()}: an
#'   integer. \code{trex_set_num_threads()}: invisibly, the previous budget.
#' @examples
#' trex_get_num_threads()   # 1
#' old <- trex_set_num_threads(min(2L, trex_get_max_threads()))
#' trex_get_num_threads()
#' trex_set_num_threads(old)
#' @family T-Rex selectors and controls
#' @export
trex_get_num_threads <- function() {
  rcpp_get_num_threads()
}

#' @rdname trex_get_num_threads
#' @export
trex_get_max_threads <- function() {
  rcpp_get_max_threads()
}

#' @rdname trex_get_num_threads
#' @export
trex_set_num_threads <- function(n) {
  stopifnot(is.numeric(n), length(n) == 1L, n >= 1)
  invisible(rcpp_set_num_threads(as.integer(n)))
}

# The budget an n_threads argument resolves to: the library's default budget
# for NULL, the positive integer otherwise.
resolve_n_threads <- function(n_threads) {
  if (is.null(n_threads)) {
    return(rcpp_get_num_threads())
  }
  stopifnot(is.numeric(n_threads), length(n_threads) == 1L, n_threads >= 1)
  as.integer(n_threads)
}
