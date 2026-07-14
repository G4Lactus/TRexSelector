// ===================================================================================
// sd_tlars_solver.hpp
// ===================================================================================
#ifndef TSOLVERS_SD_TLARS_SOLVER_HPP
#define TSOLVERS_SD_TLARS_SOLVER_HPP

#include "sd_general_base.hpp"

namespace trex::tsolvers::linear_model::lars_based {

class SD_TLARS_Solver : public SDGeneralSolver_Base {
private:
    double A_A_{0.0};
    std::vector<int> Sign_;

    bool updateActiveSet(std::size_t winning_j);

    Eigen::VectorXd signVector() const;
    Eigen::VectorXd equiangularDirection(const Eigen::Ref<const Eigen::VectorXd>& sign);
    Eigen::VectorXd equiangularVector(const Eigen::Ref<const Eigen::VectorXd>& w_A) const;
    std::pair<double, Eigen::VectorXd> computeStepSize(double Cmax, const Eigen::Ref<const Eigen::VectorXd>& u);

public:
    /**
     * @brief SD-TLARS over sparse balanced Rademacher dummies.
     *
     * @param rho_d Dummy non-zero fraction (2k ~= rho_d * n). Pass 0 to
     *              auto-calibrate k from the data via sd_calibration
     *              (pilot screen + MC; seconds at large p). The result is
     *              readable via getAutoCalibration().
     * @param L_max Total dummy budget. Pass 0 for the auto budget:
     *              the calibration's L when rho_d == 0, else 2p.
     */
    SD_TLARS_Solver(Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::VectorXd>& y,
                    double rho_d, std::size_t L_max, std::size_t T_stop, bool intercept = true,
                    uint64_t seed = 0);

    void executeStep(std::size_t T_stop = 0, bool early_stop = true) override;
};

} // namespace trex::tsolvers::linear_model::lars_based
#endif
