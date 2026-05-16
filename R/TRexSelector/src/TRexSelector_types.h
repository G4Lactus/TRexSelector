#ifndef TREXSELECTOR_TYPES_H
#define TREXSELECTOR_TYPES_H

#include <RcppEigen.h>
#include <Rcpp.h>

class RTRexSelector;
#include "rcpp_trex_wrappers.h"

// Forward declare or include all classes used in XPtr by RcppExports
#include <ml_methods/standardization/z_score_scaler.hpp>
#include <ml_methods/standardization/lp_norm_scaler.hpp>
#include <ml_methods/model_selection/ridge_cv.hpp>
#include <ml_methods/model_selection/ridge_gcv.hpp>
#include <tsolvers/tsolver_base.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>

#endif

using namespace trex::ml_methods::standardization;
using namespace trex::ml_methods::model_selection;
using namespace trex::tsolvers;
using namespace trex::utils::memmap;

