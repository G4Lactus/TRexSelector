#' @importFrom Rcpp sourceCpp
#' @useDynLib TRexSelector, .registration = TRUE
NULL

.onLoad <- function(libname, pkgname) {
  # Set the C++ custom temporary directory to R's session temp directory
  # This ensures memory-mapped matrices and temporary files fall under the CRAN policy
  rcpp_set_custom_temp_dir(tempdir())
}