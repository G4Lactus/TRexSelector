#' @useDynLib TRexSelector
#' @importFrom Rcpp evalCpp
NULL


#' @title Convert R Matrix to System Memory Mapped Matrix
#'
#' @description Maps a given R numeric matrix into a binary formatted memory-mapped file on disk
#'              efficiently.
#'
#' @param mat A \code{numeric} \code{matrix}.
#' @param filename String destination file path.
#'
#' @return NULL invisibly mapped.
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
  invisible(NULL)
}


#' Memory-Mapped Matrix Object
#'
#' @description S3 object wrapping an external pointer to a C++ MemoryMappedMatrix.
#'
#' @param filename Path to binary mmap file
#' @param rows Number of rows
#' @param cols Number of columns
#' @param mode Access mode: 'readonly' or 'readwrite' (default)
#'
#' @return Object of class 'mmap_matrix' (S3 XPtr wrapper)
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
  class(ptr) <- c("mmap_matrix", "externalptr")

  return(ptr)
}


#' @title Print mmap_matrix
#' @param x mmap_matrix object
#' @param ... arguments passed to other methods
#' @return invisibly returns x
#' @export
print.mmap_matrix <- function(x, ...) {
  cat(sprintf("MemoryMappedMatrix [%d x %d]\n",
              mmap_matrix_rows(x), mmap_matrix_cols(x)))
  cat(sprintf("  File: %s\n", attr(x, "filename")))
  cat(sprintf("  Mode: %s\n", attr(x, "mode")))
  invisible(x)
}


#' @title Get mmap_matrix dimensions
#' @param x mmap_matrix object
#' @return an integer vector of length 2
#' @export
dim.mmap_matrix <- function(x) {
  c(mmap_matrix_rows(x), mmap_matrix_cols(x))
}


#' @title Extract block from mmap_matrix
#' @param x mmap_matrix object
#' @param i row indices
#' @param j column indices
#' @param ... unused arguments
#' @param drop logical. If TRUE the result is coerced to the lowest possible dimension (not supported).
#' @return a matrix block
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
    # For now, we only support contiguous sequences to match C++ block API
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

  mmap_matrix_read_range(x, i_start, i_count, j_start, j_count)
}


#' Get Zero-Copy View as Numeric Vector (Block)
#'
#' @param x mmap_matrix object
#' @param ... extra arguments including row_range and col_range
#'
#' @return Numeric matrix (copied)
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

#' Compute False Discovery Proportion (FDP) using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#' @return Numeric FDP value.
#' @export
compute_fdp <- function(selected_indices, true_support) {
  if (length(selected_indices) == 0 || length(true_support) == 0) return(0.0)
  rcpp_compute_fdp(as.integer(selected_indices), as.integer(true_support))
}

#' Compute True Positive Proportion (TPP) using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#' @return Numeric TPP value.
#' @export
compute_tpp <- function(selected_indices, true_support) {
  if (length(selected_indices) == 0 || length(true_support) == 0) return(0.0)
  rcpp_compute_tpp(as.integer(selected_indices), as.integer(true_support))
}

#' Compute Precision using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#' @return Numeric precision value.
#' @export
compute_precision <- function(selected_indices, true_support) {
  if (length(selected_indices) == 0) return(0.0)
  if (length(true_support) == 0) return(0.0)
  rcpp_compute_precision(as.integer(selected_indices), as.integer(true_support))
}

#' Compute Recall using index sets
#'
#' @param selected_indices Integer vector of selected indices.
#' @param true_support Integer vector of true active indices.
#' @return Numeric recall value.
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
#' @export
compute_fdp_dense <- function(beta_hat, beta, eps = 1e-15) {
  rcpp_compute_fdp_dense(as.numeric(beta_hat), as.numeric(beta), eps)
}

#' Compute True Positive Proportion (TPP) using dense vectors
#'
#' @param beta_hat Numeric vector of estimated coefficients.
#' @param beta Numeric vector of true coefficients.
#' @param eps Numeric threshold for counting non-zeros (default: 1e-15).
#' @return Numeric TPP value.
#' @export
compute_tpp_dense <- function(beta_hat, beta, eps = 1e-15) {
  rcpp_compute_tpp_dense(as.numeric(beta_hat), as.numeric(beta), eps)
}
