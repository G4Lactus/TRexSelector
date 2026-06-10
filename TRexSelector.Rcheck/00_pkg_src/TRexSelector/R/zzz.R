# =============================================================================
# zzz.R - Package initialization and registration
# =============================================================================

#' @useDynLib TRexSelector, .registration = TRUE
NULL

.onLoad <- function(libname, pkgname) {
  # Set the Cpp custom temporary directory to R's session temp directory
  # This ensures memory-mapped matrices and temporary files comply with the CRAN policy
  rcpp_set_custom_temp_dir(tempdir())
}
