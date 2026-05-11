# ==============================================================================
# add_dummies.R
#
# Appends num_dummies independent N(0,1) dummy predictors to X.
# ==============================================================================

#' Add dummy predictors to the original predictor matrix
#'
#' Samples \code{num_dummies} dummy predictors from the univariate standard
#' normal distribution and appends them to the predictor matrix X.
#'
#' @param X           Real valued predictor matrix (n x p).
#' @param num_dummies Number of dummy predictors to append (integer >= 1).
#'
#' @return Enlarged predictor matrix of dimension n x (p + num_dummies).
#'
#' @importFrom stats rnorm
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- matrix(stats::rnorm(50 * 100), nrow = 50, ncol = 100)
#' add_dummies(X = X, num_dummies = 100)
add_dummies <- function(X, num_dummies) {

  if (!is.matrix(X))    stop("'X' must be a matrix.")
  if (!is.numeric(X))   stop("'X' only allows numerical values.")
  if (anyNA(X))         stop("'X' contains NAs.")
  if (length(num_dummies) != 1 || num_dummies %% 1 != 0 || num_dummies < 1)
    stop("'num_dummies' must be a positive integer.")

  dummies <- matrix(stats::rnorm(nrow(X) * num_dummies),
                    nrow  = nrow(X),
                    ncol  = num_dummies,
                    byrow = FALSE)

  cbind(X, dummies)
}
