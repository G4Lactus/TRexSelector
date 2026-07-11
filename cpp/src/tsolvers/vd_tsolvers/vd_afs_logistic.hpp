// ====================================================================
// vd_afs_logistic.hpp
// ====================================================================
#ifndef TSOLVERS_VD_TSOLVERS_VD_AFS_LOGISTIC_HPP
#define TSOLVERS_VD_TSOLVERS_VD_AFS_LOGISTIC_HPP
// ====================================================================
/**
 * @file vd_afs_logistic.hpp
 *
 * @brief Virtual Dummy AFS with logistic link function.
 *
 * @details
 * score_direction = y_binary - sigmoid(X_A * beta)
 * Basis grows by orthogonalizing the score (not the predictor).
 * IRLS refit replaces OLS.
 * At rho=1: reduces to GLM-OMP (full IRLS refit each step).
 * At rho<1: damped blend of old coefficients and IRLS solution.
 */
// ====================================================================

// VD includes
#include <tsolvers/vd_tsolvers/vd_base.hpp>

// ====================================================================

// Embedded into namespace trex::tsolvers::vd
namespace trex::tsolvers::vd {

class VD_AFS_Logistic : public VD_Base {
public:
    using VD_Base::VD_Base;

    MatC run(int T = 1) override;

private:
    // ---- Logistic-specific state ----
    double rho_ = 1.0;
    Vec y_binary_;     // {0,1} labels, reconstructed from centered y
    Vec eta_;          // linear predictor: X_A * beta (active part)
    Vec prob_;         // sigmoid(eta)
    Vec score_;        // y_binary - prob
    Vec nu_active_;    // IRLS solution on active set
    bool irls_stale_ = true;

    // ---- Score override ----
    const Vec& score_direction_() const override { return score_; }

    // ---- Logistic helpers ----
    static Vec sigmoid_(const Vec& eta);
    void update_score_();           // recompute prob, score from eta

    // ---- Selection ----
    struct Candidate {
        enum class Pool : uint8_t { Real, VD, RealizedDummy };
        Pool pool; int index; double abs_corr; bool is_new;
    };
    std::optional<Candidate> find_best_candidate_() const;

    // ---- IRLS ----
    void irls_solve_(int max_iter = 25, double tol = 1e-8);

    // ---- Blend ----
    void afs_blend_();

    // ---- Init ----
    void init_logistic_();
    bool logistic_inited_ = false;
};

}  // namespace trex::tsolvers::vd

// ====================================================================
#endif /* TSOLVERS_VD_TSOLVERS_VD_AFS_LOGISTIC_HPP */
// ====================================================================
