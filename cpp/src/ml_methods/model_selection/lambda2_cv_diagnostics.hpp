// ===================================================================================
// lambda2_cv_diagnostics.hpp
// ===================================================================================
/**
 * @file lambda2_cv_diagnostics.hpp
 *
 * @brief Health record of one lambda_2 CV tuner run, reported alongside the
 *        value the tuner returned, and the boundary-pick guard built on it.
 *
 * @details The selectors resolve lambda_2 internally and used to expose only
 *  the pick, so a boundary solution or a binding path truncation was
 *  invisible to the caller. This struct is what `computeLambda2()` fills in
 *  addition to the value; see `TRexGVSSelector::getLambda2CVDiagnostics()`
 *  and `TRexTikhonovSelector::getLambda2CVDiagnostics()`.
 *
 *  Boundary-pick guard (warning-only by design): a pick is boundary-dominated
 *  when the CHOSEN lambda_2 sits on the first or last point of its grid
 *  (`at_grid_edge`), or when, in the 2D IEN tuner, the profiled lambda_1
 *  optimum of that chosen lambda_2 sits on the truncation edge of the
 *  lambda_1 path (`profile_at_edge`). Either
 *  makes `boundary_pick` true; the selectors then emit one warning built by
 *  `boundary_pick_message()` and return the pick unchanged. When that edge
 *  is the support cap (`profile_at_support_cap`), the warning also names
 *  the selectors' `cv_dfmax_ratio` as a remedy. The two modes
 *  were established on the demo designs in August 2026: a truncation edge
 *  is counted by `n_profile_at_edge`, while a flat CV curve at low SNR
 *  lands on the lambda_2 grid ceiling with that counter at zero, so a
 *  guard must read both.
 */
// ===================================================================================
#ifndef TREX_ML_METHODS_MODEL_SELECTION_LAMBDA2_CV_DIAGNOSTICS_HPP
#define TREX_ML_METHODS_MODEL_SELECTION_LAMBDA2_CV_DIAGNOSTICS_HPP

// std includes
#include <sstream>
#include <string>
#include <string_view>

// Eigen includes
#include <Eigen/Core>

namespace trex {
namespace ml_methods {
namespace model_selection {

/**
 * @brief What a lambda_2 CV tuner did.
 *
 * `ran == false` means lambda_2 was user-supplied and no tuner executed.
 * The structural fields `n_profile_at_edge`, `support_cap`, `dfmax_ratio`,
 * `low_dimensional`, `profile_at_edge` and `profile_at_support_cap` are
 * specific to the 2D IEN-geometry tuner (`ienet_cv_ccd`); every CV method
 * fills `ran`, `seconds` and the grid-pick fields.
 */
struct lambda2_cv_diagnostics {

    /** @brief A CV tuner ran (false: lambda_2 came from the caller). */
    bool ran = false;

    /** @brief Wall-clock seconds spent resolving lambda_2. */
    double seconds = 0.0;

    /** @brief Number of lambda_2 values whose profiled lambda_1 optimum
     *  landed on a truncation edge. Zero is the healthy case: the path
     *  truncation never touched the answer. Nonzero means boundary
     *  solutions — raise `dfmax_ratio` or lower `lambda1_min_ratio`. */
    Eigen::Index n_profile_at_edge = 0;

    /** @brief Support cap (glmnet `dfmax`) applied, or -1 when disabled. */
    Eigen::Index support_cap = -1;

    /** @brief Support-cap ratio the IEN tuner was called with (the
     *  selectors' `cv_dfmax_ratio`), recorded as passed: the cap is
     *  floor(dfmax_ratio * n_train_min), and a value <= 0 disabled it
     *  (`support_cap == -1`). 0 when no IEN-geometry tuner ran. */
    double dfmax_ratio = 0.0;

    /** @brief p <= `support_cap`: the cap cannot bind, so the run pays the
     *  full cost of the deviance rules alone. Explains an expensive tuner
     *  call on a small-p design. */
    bool low_dimensional = false;

    // ----- the chosen pick on its grid (every CV method) ------------------

    /** @brief Index of the returned lambda_2 on the tuner's grid (the
     *  MIN or 1SE index, whichever rule was requested); -1 when no tuner
     *  ran. */
    Eigen::Index grid_index = -1;

    /** @brief Number of lambda_2 grid points searched; 0 when no tuner
     *  ran. */
    Eigen::Index grid_size = 0;

    /** @brief The returned lambda_2 is the first or the last grid point:
     *  the pick is the grid boundary, not a CV optimum (flat or still
     *  improving curve — low SNR, or a range that does not bracket the
     *  optimum). For the EN CCD tuner a last-point pick on a path that
     *  glmnet's deviance rules truncated is benign and is NOT flagged. */
    bool at_grid_edge = false;

    /** @brief IEN tuner only: the profiled lambda_1 optimum of the CHOSEN
     *  lambda_2 sits on the truncation edge of the lambda_1 path (support
     *  cap, deviance rule or end of the path) — a boundary solution in the
     *  nuisance dimension. */
    bool profile_at_edge = false;

    /** @brief IEN tuner only: `profile_at_edge`, and that edge is the
     *  support cap (the reference lambda_1 path of the chosen lambda_2
     *  stopped on `support_cap`, not on a deviance rule or the end of the
     *  path). Raising `dfmax_ratio` moves this edge. */
    bool profile_at_support_cap = false;

    /** @brief `at_grid_edge || profile_at_edge`: the guard condition. The
     *  selectors warn once when it holds and return the pick unchanged. */
    bool boundary_pick = false;
};


/**
 * @brief Record the chosen grid position in @p d and derive the grid-edge
 *        and boundary flags.
 *
 * @param d         Record to fill (`profile_at_edge` is read, not written).
 * @param idx       Index of the returned lambda_2 on the grid.
 * @param grid_size Number of grid points.
 * @param upper_edge_benign  True when a last-point pick must not count as
 *        a grid edge (EN CCD tuner: the deviance rules truncated the path,
 *        so the CV was indifferent over a tail it never needed).
 */
inline void record_grid_pick(lambda2_cv_diagnostics& d,
                             Eigen::Index idx,
                             Eigen::Index grid_size,
                             bool upper_edge_benign = false) {
    d.grid_index = idx;
    d.grid_size  = grid_size;
    const bool lower = (grid_size > 0 && idx == 0);
    const bool upper = (grid_size > 0 && idx == grid_size - 1 && !upper_edge_benign);
    d.at_grid_edge  = lower || upper;
    d.boundary_pick = d.at_grid_edge || d.profile_at_edge;
}


/**
 * @brief The boundary-pick warning text, or an empty string when the pick
 *        is healthy (`d.boundary_pick == false`).
 *
 * @param d         Filled record of the run.
 * @param class_tag Selector name for the message prefix.
 * @param method    Name of the lambda_2 method that ran.
 * @param lambda2   The value returned (working scale, as the caller sees it).
 */
inline std::string boundary_pick_message(const lambda2_cv_diagnostics& d,
                                         std::string_view class_tag,
                                         std::string_view method,
                                         double lambda2) {
    if (!d.boundary_pick) { return {}; }

    std::ostringstream os;
    os << class_tag << "::computeLambda2: " << method
       << " returned lambda_2 = " << lambda2 << " as a BOUNDARY pick";
    if (d.at_grid_edge) {
        // "first" / "last" by index: the IEN tuner's grid ascends, the EN
        // CCD tuner's descends (glmnet order), so no value-order word here.
        os << ": it is the " << (d.grid_index == 0 ? "first" : "last")
           << " point of the lambda_2 grid (index " << d.grid_index << " of "
           << d.grid_size << "), so it is the grid boundary, not a CV "
              "optimum; the CV curve is flat or still improving there (low "
              "SNR, or the grid does not bracket the optimum)";
    }
    if (d.profile_at_edge) {
        os << (d.at_grid_edge ? "; and" : ":")
           << " the profiled lambda_1 optimum of this lambda_2 sits on the "
              "truncation edge of the lambda_1 path (support cap, deviance "
              "rule or end of the path)";
        if (d.profile_at_support_cap) {
            os << ", here the support cap (more than " << d.support_cap
               << " nonzero coefficients, cv_dfmax_ratio = " << d.dfmax_ratio
               << ")";
        }
        os << ", so the tuner would have preferred a smaller lambda_1";
    }
    os << ". The pick is returned unchanged. Remedies: prefer the 1SE rule "
          "over MIN, supply a fixed lambda_2";
    if (d.profile_at_support_cap) {
        os << ", raise the support cap (cv_dfmax_ratio)";
    }
    if (d.profile_at_edge) {
        os << ", lengthen the lambda_1 path (cv_n_lambda1)";
    }
    os << ", or refine the lambda_2 grid (cv_n_lambda2 / cv_n_lambda); "
          "read getLambda2CVDiagnostics().";
    return os.str();
}

}  // namespace model_selection
}  // namespace ml_methods
}  // namespace trex

#endif  // TREX_ML_METHODS_MODEL_SELECTION_LAMBDA2_CV_DIAGNOSTICS_HPP
