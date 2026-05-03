// ==============================================================================
// demo_trex_da_08_groups.cpp
// ==============================================================================
/**
 * @file demo_trex_da_08_groups.cpp
 *
 * @brief DA-TRex demo and MC simulation for the prior-groups DGP.
 *
 * @details
 *
 *  Part 1: Single-run demo (n=150, p=500, s=5, 3-level hierarchy).
 *  Part 2: MC simulation sweeping SNR (n=300, p=1000, s=10, 200 MC, 3 solvers).
 */
// ==============================================================================

#include "trex_da_sim_common.hpp"

using namespace da_sim;


// ==============================================================================
// Group structure builder
// ==============================================================================

/**
 * @brief Build a 3-level hierarchical group structure for p predictors.
 *
 * Level 1: groups of 10   (p/10 groups)
 * Level 2: groups of 50   (p/50 groups)
 * Level 3: groups of 250  (p/250 groups)
 */
std::vector<std::vector<Eigen::Index>> make_group_structure(int p) {
    std::vector<std::vector<Eigen::Index>> groups(3);
    for (Eigen::Index j = 0; j < p; ++j) {
        groups[0].push_back(j / 10);
        groups[1].push_back(j / 50);
        groups[2].push_back(j / 250);
    }
    return groups;
}

const std::vector<double> rho_levels = {0.80, 0.50, 0.20};


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);

    // ══════════════════════════════════════════════════════════════════════════
    //  Part 1: Single-run demo
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        SimConfig cfg;
        cfg.n = 150; cfg.p = 500; cfg.s = 5;
            constexpr int max_gap = 20;
        const auto policy = SupportPolicy::Random;

        auto groups = make_group_structure(cfg.p);

        cdianostics::print_section_header(
            "Prior Groups Demo  |  n=" + std::to_string(cfg.n)
            + "  p=" + std::to_string(cfg.p) + "  s=" + std::to_string(cfg.s)
            + "  3-level hierarchy"
            + "  support=" + support_policy_label(policy));

        auto dat = dgp_groups(cfg.n, cfg.p,
                              make_support(policy, cfg.s, cfg.p, max_gap,
                                           static_cast<unsigned>(cfg.base_seed)),
                              cfg.amplitude,
                              groups, rho_levels,
                              cfg.snr, static_cast<unsigned>(cfg.base_seed));

        auto trex_ctrl = make_trex_control(cfg.K);
        trex_ctrl.solver_type = SolverTypeForTRex::TOMP;

        TRexDAControlParameter da_ctrl;
        da_ctrl.prior_groups    = groups;
        da_ctrl.rho_grid_labels = rho_levels;

        Eigen::Map<Eigen::MatrixXd> X_map(
            dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexDASelector selector(
            X_map, y_map, cfg.tFDR, da_ctrl, trex_ctrl,
            cfg.base_seed, /*verbose=*/true);
        selector.select();

        auto m = evaluate_selection("Prior Groups",
                                   selector.getSelectedIndices(),
                                   dat.true_support);

        print_single_result(m, cfg.tFDR);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Part 2: Monte Carlo simulation — SNR sweep
    // ══════════════════════════════════════════════════════════════════════════
    if (false)
    {
        SimConfig cfg;  // n=300, p=1000, s=10
        constexpr int max_gap = 20;
        const auto policy = SupportPolicy::Random;

        auto groups = make_group_structure(cfg.p);

        TRexDAControlParameter da_ctrl;
        da_ctrl.prior_groups    = groups;
        da_ctrl.rho_grid_labels = rho_levels;

        run_snr_sweep(
            "da_groups",
            {0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
            cfg.num_MC,
            cfg.tFDR,
            cfg.base_seed,
            default_solvers(),
            [&](double snr, unsigned seed) {
                return dgp_groups(
                    cfg.n,
                    cfg.p,
                    make_support(policy, cfg.s, cfg.p, max_gap, seed),
                    cfg.amplitude,
                    groups,
                    rho_levels,
                    snr,
                    seed
                );
            },
            da_ctrl,
            make_trex_control(cfg.K),
            "Prior Groups 3-level, n=" + std::to_string(cfg.n)
            + ", p=" + std::to_string(cfg.p)
            + ", support=" + support_policy_label(policy)
            + ", max_gap=" + std::to_string(max_gap));
    }

    std::cout << "\nPrior Groups demo complete.\n";
    return 0;
}
