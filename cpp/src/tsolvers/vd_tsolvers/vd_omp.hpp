// ====================================================================
// vd_omp.hpp
// ====================================================================
#ifndef TSOLVERS_VD_TSOLVERS_VD_OMP_HPP
#define TSOLVERS_VD_TSOLVERS_VD_OMP_HPP
// ====================================================================
/**
 * @file vd_omp.hpp
 *
 * @brief Virtual Dummy OMP (greedy argmax, full OLS refit).
 */
// ====================================================================

// VD includes
#include <tsolvers/vd_tsolvers/vd_base.hpp>

// ====================================================================

// Embedded into namespace trex::tsolvers::vd
namespace trex::tsolvers::vd {

class VD_OMP : public VD_Base {
public:
    using VD_Base::VD_Base;  // inherit constructors

    MatC run(int T = 1) override;

private:
    Vec Xty_active_;

    struct Candidate {
        enum class Pool : uint8_t { Real, VD, Realized };
        Pool pool; int index; double abs_corr;
    };
    std::optional<Candidate> find_best_candidate_() const;
    void append_to_factor_(const Vec& x_col);
    void ols_refit_();

    void init_omp_();  // called once at start of run()
    bool omp_inited_ = false;
};

}  // namespace trex::tsolvers::vd

// ====================================================================
#endif /* TSOLVERS_VD_TSOLVERS_VD_OMP_HPP */
// ====================================================================
