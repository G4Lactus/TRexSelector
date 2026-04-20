// ==============================================================================
// demo_trx_scr_06_screen_trex_solvers.cpp
// ==============================================================================
/**
 * @file demo_trx_scr_06_screen_trex_solvers.cpp
 *
 * @brief Screen-TRex Selector — solver comparison: TLARS, TAFS (ρ=0.3), TOMP.
 *
 * @details
 *  Compares the three T-Rex solver backends across both Screen-TRex method
 *  variants (ordinary Phi > 0.5 threshold and bootstrap-CI).
 *
 *  Demo 01 — Monte Carlo SNR sweep: three solvers, ordinary Screen-TRex only.
 *  Demo 02 — Monte Carlo SNR sweep: three solvers × ordinary + bootstrap (6 series).
 *
 *  In-memory variant (X and D matrices held in RAM).
 */
// ==============================================================================

// std includes
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <utils/datageneration/utils_datagen.hpp>

// Demo utilities
#include "demo_trx_scr_common.hpp"

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace datagen = trex::utils::datageneration::datagen;
using namespace scr_demo;

// ==============================================================================
// Local helpers — combined solver × method-variant descriptors
// ==============================================================================

namespace {

/**
 * @brief Combined descriptor for a Screen-TRex solver × method variant.
 *
 * @details Extends the library's ScreenMethodInfo with the solver backend
 *          and the AFS penalty parameter so that the outer MC loop can build
 *          both the TRexControlParameter and the ScreenTRexControlParameter
 *          from a single object.
 */
struct SolverVariant {
    std::string       name;                             ///< Display name
    SolverTypeForTRex solver_type;                      ///< T-Rex solver backend
    double            rho_afs   = 0.0;                  ///< AFS penalty (TAFS only)
    ScreenTRexMethod  method    = ScreenTRexMethod::TREX; ///< Screen-TRex variant
    bool              bootstrap = false;                 ///< Use bootstrap-CI threshold
};

/** @brief Three-solver list — ordinary Screen-TRex only. */
inline std::vector<SolverVariant> ordinary_solver_variants()
{
    return {
        {"TLARS (Ordinary)",    SolverTypeForTRex::TLARS, 0.0,
         ScreenTRexMethod::TREX, false},
        {"TAFS-0.3 (Ordinary)", SolverTypeForTRex::TAFS,  0.3,
         ScreenTRexMethod::TREX, false},
        {"TOMP (Ordinary)",     SolverTypeForTRex::TOMP,  0.0,
         ScreenTRexMethod::TREX, false},
    };
}

/** @brief Six combined variants — three solvers × ordinary + bootstrap. */
inline std::vector<SolverVariant> all_solver_variants()
{
    std::vector<SolverVariant> v;
    v.reserve(6);
    for (const bool bs : {false, true}) {
        const std::string suffix = bs ? " (Bootstrap)" : " (Ordinary)";
        v.push_back({"TLARS"    + suffix, SolverTypeForTRex::TLARS, 0.0,
                     ScreenTRexMethod::TREX, bs});
        v.push_back({"TAFS-0.3" + suffix, SolverTypeForTRex::TAFS,  0.3,
                     ScreenTRexMethod::TREX, bs});
        v.push_back({"TOMP"     + suffix, SolverTypeForTRex::TOMP,  0.0,
                     ScreenTRexMethod::TREX, bs});
    }
    return v;
}

/**
 * @brief Build a TRexControlParameter configured for the given SolverVariant.
 *
 * @param sv  Solver variant whose type and rho_afs are applied to the default
 *            TRexControlParameter produced by make_trex_control().
 * @return    Ready-to-use TRexControlParameter.
 */
inline TRexControlParameter make_trex_ctrl_for(const SolverVariant& sv)
{
    auto ctrl = make_trex_control();
    ctrl.solver_type           = sv.solver_type;
    ctrl.solver_params.rho_afs = sv.rho_afs;
    ctrl.tloop_stagnation_stop = true;
    ctrl.tloop_max_stagnant_steps = 5;
    return ctrl;
}

/**
 * @brief Convert a SolverVariant list to ScreenMethodInfo (name only).
 *
 * @details Used to pass the ordered method list to save_and_print_mc_results(),
 *          which needs names to index into the FDR/TPR maps.
 */
inline std::vector<ScreenMethodInfo>
to_screen_method_info(const std::vector<SolverVariant>& variants)
{
    std::vector<ScreenMethodInfo> info;
    info.reserve(variants.size());
    for (const auto& sv : variants)
        info.push_back({sv.name, sv.method, sv.bootstrap});
    return info;
}

} // anonymous namespace


// ==============================================================================
// Demo 01: MC SNR sweep — three solvers, ordinary Screen-TRex
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo comparing TLARS, TAFS (ρ=0.3), and TOMP.
 *
 * @details Uses ordinary Screen-TRex (Phi > 0.5 threshold) for all three
 *          solvers.  Reports FDR, TPR, and estimated FDR averaged over num_MC
 *          Monte Carlo runs at each SNR grid point.
 *
 * @param num_MC   Number of Monte Carlo runs per (solver, SNR) pair.
 * @param high_dim If true, use high-dimensional setting (n=300, p=1000).
 * @param rnd_coef If true, draw support coefficients from N(0,1) each run.
 */
void demo_SolverComparison_MC(std::size_t num_MC, bool high_dim, bool rnd_coef)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo: Screen-TRex MC — Solver Comparison (TLARS / TAFS-0.3 / TOMP)");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 :  300;
    std::cout << (high_dim ? "High-dimensional (p > n)"
                           : "Low-dimensional (n > p)") << "\n";

    const std::size_t         support_size = 10;
    const std::vector<double> snr_values   =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    const auto variants = ordinary_solver_variants();
    const auto methods  = to_screen_method_info(variants);

    // Storage: variant name → SNR-indexed accumulator
    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& sv : variants) {
        const auto sz        = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[sv.name] = Eigen::VectorXd::Zero(sz);
        fdr_map    [sv.name] = Eigen::VectorXd::Zero(sz);
        tpr_map    [sv.name] = Eigen::VectorXd::Zero(sz);
    }

    std::mt19937                              rng(24);
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
    std::normal_distribution<double>           coef_dist(0.0, 1.0);

    // -----------------------------------------------------------------------
    // Solver loop
    // -----------------------------------------------------------------------
    for (const auto& sv : variants) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Solver: " << sv.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto trex_ctrl = make_trex_ctrl_for(sv);

        // SNR loop -----------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr          = snr_values[snr_idx];
            double       total_est_fdr = 0.0;
            double       total_fdp    = 0.0;
            double       total_tpp    = 0.0;

            // Monte Carlo loop -----------------------------------------------
            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Draw random support of size support_size
                std::unordered_set<std::size_t> support_set;
                while (support_set.size() < support_size)
                    support_set.insert(idx_dist(rng));
                const std::vector<std::size_t> true_support(
                    support_set.begin(), support_set.end());

                // Draw regression coefficients
                std::vector<double> true_coefs;
                true_coefs.reserve(support_size);
                for (std::size_t i = 0; i < support_size; ++i)
                    true_coefs.push_back(rnd_coef ? coef_dist(rng) : 1.0);

                datagen::SyntheticData data(
                    static_cast<Eigen::Index>(n),
                    static_cast<Eigen::Index>(p),
                    true_support, true_coefs, snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc));

                Eigen::Map<Eigen::MatrixXd> X_map(
                    data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(
                    data.getY().data(), data.rows());

                auto screen_ctrl = make_screen_control(
                    sv.method, sv.bootstrap);

                ScreenTRexSelector sctrex(
                    X_map,
                    y_map,
                    screen_ctrl,
                    trex_ctrl,
                    /*seed=*/-1,
                    /*verbose=*/false);
                sctrex.select();

                const auto selected = sctrex.getSelectedIndices();
                total_est_fdr += sctrex.getScreenResult().estimated_FDR;
                total_fdp     += rates::compute_fdp(selected, true_support);
                total_tpp     += rates::compute_tpp(selected, true_support);

                print_mc_progress(snr, mc, num_MC);
            }
            print_mc_done(snr, num_MC);

            const double inv_MC = 1.0 / static_cast<double>(num_MC);
            const auto   ei     = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[sv.name](ei) = total_est_fdr * inv_MC;
            fdr_map    [sv.name](ei) = total_fdp     * inv_MC;
            tpr_map    [sv.name](ei) = total_tpp     * inv_MC;
        }
    }

    const std::string file_stem =
        "d06_screen_trex_solvers_mc_snr_n" + std::to_string(n)
        + "_p" + std::to_string(p)
        + "_s" + std::to_string(support_size);
    save_and_print_mc_results(num_MC, file_stem, snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// Demo 02: MC SNR sweep — three solvers × ordinary + bootstrap
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo over all six (solver × method-variant) combinations.
 *
 * @details Extends Demo 02 by also running bootstrap-CI Screen-TRex for each
 *          solver, producing six series:
 *            TLARS / TAFS-0.3 / TOMP  ×  Ordinary / Bootstrap.
 *
 * @param num_MC   Number of Monte Carlo runs per (variant, SNR) pair.
 * @param high_dim If true, use high-dimensional setting (n=300, p=1000).
 * @param rnd_coef If true, draw support coefficients from N(0,1) each run.
 */
void demo_SolverAndMethod_MC(std::size_t num_MC, bool high_dim, bool rnd_coef)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo: Screen-TRex MC — Solver × Method Comparison (6 series)");

    const std::size_t n = high_dim ? 300  : 1000;
    const std::size_t p = high_dim ? 1000 :  300;
    std::cout << (high_dim ? "High-dimensional (p > n)"
                           : "Low-dimensional (n > p)") << "\n";

    const std::size_t         support_size = 10;
    const std::vector<double> snr_values   =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    const auto variants = all_solver_variants();
    const auto methods  = to_screen_method_info(variants);

    // Storage: variant name → SNR-indexed accumulator
    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& sv : variants) {
        const auto sz        = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[sv.name] = Eigen::VectorXd::Zero(sz);
        fdr_map    [sv.name] = Eigen::VectorXd::Zero(sz);
        tpr_map    [sv.name] = Eigen::VectorXd::Zero(sz);
    }

    std::mt19937                              rng(24);
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
    std::normal_distribution<double>           coef_dist(0.0, 1.0);

    // -----------------------------------------------------------------------
    // Variant loop (solver × method-variant)
    // -----------------------------------------------------------------------
    for (const auto& sv : variants) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Variant: " << sv.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto trex_ctrl = make_trex_ctrl_for(sv);

        // SNR loop -----------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr          = snr_values[snr_idx];
            double       total_est_fdr = 0.0;
            double       total_fdp    = 0.0;
            double       total_tpp    = 0.0;

            // Monte Carlo loop -----------------------------------------------
            for (std::size_t mc = 0; mc < num_MC; ++mc) {

                // Draw random support of size support_size
                std::unordered_set<std::size_t> support_set;
                while (support_set.size() < support_size)
                    support_set.insert(idx_dist(rng));
                const std::vector<std::size_t> true_support(
                    support_set.begin(), support_set.end());

                // Draw regression coefficients
                std::vector<double> true_coefs;
                true_coefs.reserve(support_size);
                for (std::size_t i = 0; i < support_size; ++i)
                    true_coefs.push_back(rnd_coef ? coef_dist(rng) : 1.0);

                datagen::SyntheticData data(
                    static_cast<Eigen::Index>(n),
                    static_cast<Eigen::Index>(p),
                    true_support, true_coefs, snr,
                    static_cast<int>(24 + snr_idx * 1000 + mc));

                Eigen::Map<Eigen::MatrixXd> X_map(
                    data.getX().data(), data.rows(), data.cols());
                Eigen::Map<Eigen::VectorXd> y_map(
                    data.getY().data(), data.rows());

                auto screen_ctrl = make_screen_control(sv.method, sv.bootstrap);

                ScreenTRexSelector sctrex(
                    X_map, y_map, screen_ctrl, trex_ctrl, /*seed=*/-1,
                    /*verbose=*/false);
                sctrex.select();

                const auto selected = sctrex.getSelectedIndices();
                total_est_fdr += sctrex.getScreenResult().estimated_FDR;
                total_fdp     += rates::compute_fdp(selected, true_support);
                total_tpp     += rates::compute_tpp(selected, true_support);

                print_mc_progress(snr, mc, num_MC);
            }
            print_mc_done(snr, num_MC);

            const double inv_MC = 1.0 / static_cast<double>(num_MC);
            const auto   ei     = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[sv.name](ei) = total_est_fdr * inv_MC;
            fdr_map    [sv.name](ei) = total_fdp     * inv_MC;
            tpr_map    [sv.name](ei) = total_tpp     * inv_MC;
        }
    }

    const std::string file_stem =
        "d06_screen_trex_solver_method_mc_snr_n" + std::to_string(n)
        + "_p" + std::to_string(p)
        + "_s" + std::to_string(support_size);
    save_and_print_mc_results(num_MC, file_stem, snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    // -------------------------------------------------------------------------
    // Demo 01: MC SNR sweep — three solvers, ordinary Screen-TRex
    // -------------------------------------------------------------------------
    demo_SolverComparison_MC(500, true, false);

    // -------------------------------------------------------------------------
    // Demo 02: MC SNR sweep — three solvers × ordinary + bootstrap (6 series)
    // -------------------------------------------------------------------------
    demo_SolverAndMethod_MC(500, true, false);

    return 0;
}
