// =========================================================================================
// rcpp_utils.cpp - Rcpp utility functions for TRexSelector.
// =========================================================================================
/*
 * Note on Roxygen Documentation:
 * All functions exported via Rcpp in this file are documented using standard Roxygen tags
 * but strictly include the `@noRd` tag. These are internal C++ bindings that are wrapped
 * by user-friendly R6 classes and R functions in the package namespace. Generating `.Rd`
 * manuals for these endpoints would clutter the public API.
 */
// =========================================================================================

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <string>

// Include the memmap utility
#include <utils/memmap/memory_mapped_matrix.hpp>

// Include the evaluation metrics
#include <utils/eval_metrics/utils_eval_counts.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Include the TempDirManager
#include <trex_selector_methods/utils/trex_temp_dir_manager.hpp>

// ========================================================================================

// Namespace convenience
using namespace Rcpp;
using namespace trex::utils::memmap;

// ========================================================================================
// Utils: Global Configuration
// ========================================================================================

//' @title Set custom temp directory for the C++ backend
//'
//' @description Injects the R session's tempdir() into the C++ backend to obey CRAN memory rules.
//'
//' @param temp_dir The absolute path to the directory (e.g. from `tempdir()`).
//'
//' @noRd
// [[Rcpp::export]]
void rcpp_set_custom_temp_dir(const std::string& temp_dir) {
    trex::trex_selector_methods::utils::filesystem::TempDirManager::setCustomTempDir(temp_dir);
}

// ========================================================================================
// Utils: MemoryMappedMatrix Bindings
// ========================================================================================

//' @title Create Memory Mapped Matrix
//'
//' @param filename Path to the file.
//' @param rows Number of rows
//' @param cols Number of columns.
//' @param mode_idx Mode index with 0: ReadOnly, 1: ReadWrite.
//'
//' @return XPtr to MemoryMappedMatrix
//'
//' @noRd
// [[Rcpp::export]]
XPtr<MemoryMappedMatrix<double>> mmap_matrix_create(
    const std::string& filename,
    int rows,
    int cols,
    int mode_idx = 1
) {

    AccessMode mode = (mode_idx == 0) ? AccessMode::ReadOnly : AccessMode::ReadWrite;
    return XPtr<MemoryMappedMatrix<double>>(
        new MemoryMappedMatrix<double>(filename,
                                         static_cast<std::size_t>(rows),
                                         static_cast<std::size_t>(cols), mode)
    );
}


//' @title Convert Memory Mapped Matrix to R Matrix
//'
//' @param ptr XPtr to MemoryMappedMatrix
//'
//' @return NumericMatrix
//'
//' @noRd
// [[Rcpp::export]]
NumericMatrix mmap_matrix_to_r_matrix(
    XPtr<MemoryMappedMatrix<double>> ptr  // NOLINT(performance-unnecessary-value-param)
) {
    const MemoryMappedMatrix<double>& cref = *ptr;
    auto map = cref.getMap();

    // Copy the mapped memory safely to a standard R matrix buffer
    NumericMatrix result(static_cast<int>(map.rows()), static_cast<int>(map.cols()));
    Eigen::Map<Eigen::MatrixXd> res_map(REAL(result),
                                        static_cast<Eigen::Index>(map.rows()),
                                        static_cast<Eigen::Index>(map.cols()));
    res_map = map;
    return result;
}


//' @title Convert R Matrix to Memory Mapped Matrix
//'
//' @param mat R NumericMatrix
//' @param filename Output filename
//'
//' @noRd
// [[Rcpp::export]]
void convert_to_memory_mapped_matrix(
    NumericMatrix mat, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    std::size_t rows = mat.nrow();
    std::size_t cols = mat.ncol();

    MemoryMappedMatrix<double> mmap(filename, rows, cols, AccessMode::ReadWrite);
    auto map = mmap.getMap();

    // R matrices are natively column-major. Stream directly into the mapped region.
    Eigen::Map<Eigen::MatrixXd> r_map(REAL(mat),
                                      static_cast<Eigen::Index>(rows),
                                      static_cast<Eigen::Index>(cols));
    map = r_map;
}


//' @title Get Number of Rows in Memory Mapped Matrix
//'
//' @param ptr XPtr to MemoryMappedMatrix
//'
//' @return Integer number of rows
//'
//' @noRd
// [[Rcpp::export]]
int mmap_matrix_rows(
    XPtr<MemoryMappedMatrix<double>> ptr  // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->rows());
}


//' @title Get Number of Columns in Memory Mapped Matrix
//'
//' @param ptr XPtr to MemoryMappedMatrix
//' @return Integer number of columns
//'
//' @noRd
// [[Rcpp::export]]
int mmap_matrix_cols(
    XPtr<MemoryMappedMatrix<double>> ptr  // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->cols());
}


//' @title Get Total Size of Memory Mapped Matrix
//'
//' @param ptr XPtr to MemoryMappedMatrix
//'
//' @return Integer total size
//'
//' @noRd
// [[Rcpp::export]]
int mmap_matrix_size(
    XPtr<MemoryMappedMatrix<double>> ptr  // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->size());
}


//' @title Read Block from Memory Mapped Matrix
//'
//' @param ptr XPtr to MemoryMappedMatrix
//' @param row_start Start row index (0-based)
//' @param row_count Number of rows to read
//' @param col_start Start column index (0-based)
//' @param col_count Number of columns to read
//'
//' @return NumericMatrix block
//'
//' @noRd
// [[Rcpp::export]]
NumericMatrix mmap_matrix_read_range(
    XPtr<MemoryMappedMatrix<double>> ptr, // NOLINT(performance-unnecessary-value-param)
    int row_start, int row_count,
    int col_start, int col_count) {

    const MemoryMappedMatrix<double>& cref = *ptr;
    auto map = cref.getMap();

    // Bounds checking
    if (row_start < 0 || col_start < 0 || row_count < 0 || col_count < 0 ||
        row_start + row_count > map.rows() || col_start + col_count > map.cols()) {
        stop("MemoryMappedMatrix: slice out of bounds");
    }

    NumericMatrix result(row_count, col_count);
    Eigen::Map<Eigen::MatrixXd> res_map(REAL(result),
                                           row_count,
                                           col_count);
    res_map = map.block(row_start, col_start, row_count,
                       col_count);
    return result;
}


// ========================================================================================
// Utils: Evaluation Metrics
// ========================================================================================

//' @title Compute FDP from sets
//'
//' @param selected_indices Selected indices
//' @param true_support True support indices
//'
//' @return FDP value
//'
//' @noRd
// [[Rcpp::export]]
double rcpp_compute_fdp(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    return trex::utils::eval::rates::compute_fdp(selected_indices, true_support);
}


//' @title Compute TPP from sets
//'
//' @param selected_indices Selected indices
//' @param true_support True support indices
//'
//' @return TPP value
//'
//' @noRd
// [[Rcpp::export]]
double rcpp_compute_tpp(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    return trex::utils::eval::rates::compute_tpp(selected_indices, true_support);
}


//' @title Compute Precision from sets
//'
//' @param selected_indices Selected indices
//' @param true_support True support indices
//'
//' @return Precision value
//'
//' @noRd
// [[Rcpp::export]]
double rcpp_compute_precision(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    return trex::utils::eval::rates::compute_precision(selected_indices, true_support);
}


//' @title Compute Recall from sets
//'
//' @param selected_indices Selected indices
//' @param true_support True support indices
//'
//' @return Recall value
//'
//' @noRd
// [[Rcpp::export]]
double rcpp_compute_recall(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support
) {
    return trex::utils::eval::rates::compute_recall(selected_indices, true_support);
}


//' @title Compute FDP from dense vectors
//'
//' @param beta_hat Estimated coefficients.
//' @param beta True coefficients.
//' @param eps Tolerance.
//'
//' @return FDP value
//'
//' @noRd
// [[Rcpp::export]]
double rcpp_compute_fdp_dense(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& beta,
    double eps = 1e-15
) {
    return trex::utils::eval::rates::compute_fdp(beta_hat, beta, eps);
}


//' @title Compute TPP from dense vectors
//'
//' @param beta_hat Estimated coefficients.
//' @param beta True coefficients.
//' @param eps Tolerance.
//'
//' @return TPP value.
//'
//' @noRd
// [[Rcpp::export]]
double rcpp_compute_tpp_dense(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& beta,
    double eps = 1e-15
) {
    return trex::utils::eval::rates::compute_tpp(beta_hat, beta, eps);
}
