# =============================================================================
# utils.R - Utility functions for TRexSelector
# =============================================================================

#' @importFrom Rcpp evalCpp
NULL


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
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' mm <- convert_to_memory_mapped(mat, f)
#' print(mm)
#' }
#'
#' @export
convert_to_memory_mapped <- function(mat, filename) {
  if (!is.matrix(mat) || !is.numeric(mat)) {
    stop("Input must be a numeric matrix.")
  }
  if (!is.character(filename) || length(filename) != 1) {
    stop("Filename must be a single string path.")
  }

  convert_to_memory_mapped_matrix(mat, filename)
  mmap_matrix(filename, nrow(mat), ncol(mat))
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
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' convert_to_memory_mapped(mat, f)
#' mm <- mmap_matrix(f, rows = 4, cols = 5)
#' }
#'
#' @export
mmap_matrix <- function(filename, rows, cols, mode = "readwrite") {
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
#' convert_to_memory_mapped(mat, f)
#' mm <- mmap_matrix(f, rows = 4, cols = 5)
#' get_ptr(mm)
#' }
#'
#' @export
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
#' convert_to_memory_mapped(mat, f)
#' mm <- mmap_matrix(f, rows = 4, cols = 5)
#' print(mm)
#' }
#'
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
#' convert_to_memory_mapped(mat, f)
#' mm <- mmap_matrix(f, rows = 4, cols = 5)
#' dim(mm)
#' }
#' @export
dim.mmap_matrix <- function(x) {
  c(mmap_matrix_rows(x), mmap_matrix_cols(x))
}


#' @title Extract block from mmap_matrix
#'
#' @param x mmap_matrix object
#' @param i row indices
#' @param j column indices
#' @param ... unused arguments
#' @param drop logical. If TRUE the result is coerced to the lowest possible dimension
#' (not supported).
#'
#' @return a matrix block
#'
#' @examples
#' \donttest{
#' mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
#' f <- tempfile(fileext = ".bin")
#' on.exit(unlink(f), add = TRUE)
#' convert_to_memory_mapped(mat, f)
#' mm <- mmap_matrix(f, rows = 4, cols = 5)
#' mm[1:2, 1:3]
#' }
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
#' mm <- convert_to_memory_mapped(mat, f)
#' mm[2, 3] <- 99.0
#' mm[1:2, 1:3] <- matrix(as.double(1:6), nrow = 2, ncol = 3)
#' }
#'
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
#' convert_to_memory_mapped(mat, f)
#' mm <- mmap_matrix(f, rows = 4, cols = 5)
#' as.matrix(mm)
#' }
#'
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
#' compute_fdp(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @export
compute_fdp <- function(selected_indices, true_support) {
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
#' compute_tpp(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @export
compute_tpp <- function(selected_indices, true_support) {
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
#' compute_precision(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @export
compute_precision <- function(selected_indices, true_support) {
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
#' compute_recall(c(1L, 3L, 5L), c(1L, 2L, 3L))
#'
#' @export
compute_recall <- function(selected_indices, true_support) {
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
#' compute_fdp_dense(c(1.5, 0, 0.8, 0, 0), c(2.0, 0, 1.0, 0, 0))
#' @export
compute_fdp_dense <- function(beta_hat, beta, eps = 1e-15) {
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
#' compute_tpp_dense(c(1.5, 0, 0.8, 0, 0), c(2.0, 0, 1.0, 0, 0))
#'
#' @export
compute_tpp_dense <- function(beta_hat, beta, eps = 1e-15) {
  rcpp_compute_tpp_dense(as.numeric(beta_hat), as.numeric(beta), eps)
}
