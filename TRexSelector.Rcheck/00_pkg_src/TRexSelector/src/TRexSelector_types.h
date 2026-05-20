#ifndef RCPP_TREX_SELECTOR_TYPES_H
#define RCPP_TREX_SELECTOR_TYPES_H

#include <RcppEigen.h>
#include <Rcpp.h>

class RTRexSelector;
class RTRexDASelector;
class RTRexGVSSelector;
class RTRexScreeningSelector;
class RTRexBiobankScreeningSelector;
#include "rcpp_trex_wrappers.h"
#include "rcpp_trex_da_wrapper.h"
#include "rcpp_trex_gvs_wrapper.h"
#include "rcpp_trex_screening_wrapper.h"
#include "rcpp_trex_biobank_screening_wrapper.h"

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

#ifdef FALSE
#undef FALSE
#endif
#define FALSE ((Rboolean)0)
