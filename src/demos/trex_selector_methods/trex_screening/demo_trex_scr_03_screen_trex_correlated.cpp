// ==============================================================================
// demo_trex_scr_03_screen_trex_correlated.cpp
// ==============================================================================
/**
 * @file demo_trex_scr_03_screen_trex_correlated.cpp
 *
 * @brief Screen-TRex on correlated predictor structures (AR1 and equi).
 *
 * @details
 *    1. Generate X with AR(1) or equicorrelation structure.
 *    2. Run Screen-TRex with the matching DA variant.
 *    3. Report FDP, TPP, estimated FDR, and estimated ρ.
 *
 *  Demo 01 — DA-AR1 Screen-TRex on AR(1)-correlated data.
 *  Demo 02 — DA-EQUI Screen-TRex on equicorrelated data.
 *  Demo 03 — Ordinary Screen-TRex on AR(1) data (no DA correction, baseline).
 *  Demo 04 — Monte Carlo: Ordinary vs. Bootstrap vs. DA-AR1 Screen-TRex on AR(1) data.
 *             Parallelised via OpenMP.
 *  Demo 05 — Monte Carlo: Ordinary vs. Bootstrap vs. DA-EQUI Screen-TRex on equicorrelated data.
 *             Parallelised via OpenMP.
 *  Demo 06 — rho sweep: DA-AR1 on AR(1) data. Parallelised via OpenMP.
 *  Demo 07 — rho sweep: DA-EQUI on equicorrelated data. Parallelised via OpenMP.
 *  Demo 08 — DA-BLOCK-EQUI Screen-TRex on block-equicorrelated data.
 *  Demo 09 — rho sweep: DA-BLOCK-EQUI on block-equicorrelated data. Parallelised via OpenMP.
 */
// ==============================================================================

// std includes
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Screen-TRex demo common header
#include "demo_trex_scr_common.hpp"

using namespace scr_demo;

// ==============================================================================
// DGP helpers (local to this demo)
// ==============================================================================

namespace {

/** @brief Storage structure for DGP data. */
struct DGPData {
    /** @brief Design matrix. */
    Eigen::MatrixXd X;

    /** @brief Response vector. */
    Eigen::VectorXd y;

    /** @brief True support indices. */
    std::vector<std::size_t> true_support;
};


/**
 * @brief Generate evenly-spaced sparse beta vector.
 *
 * @param p         Total number of predictors.
 * @param p1        Number of active predictors (non-zero coefficients).
 * @param beta_val  Coefficient value for active predictors.
 *
 * @return Vector of length p with p1 non-zero entries of value beta_val, evenly spaced.
 */
Eigen::VectorXd make_beta(Eigen::Index p, Eigen::Index p1, double beta_val) {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (Eigen::Index i = 0; i < p1; ++i) {
        // evenly spaced: skip first and last boundary
        const Eigen::Index idx = static_cast<Eigen::Index>(
            std::round(static_cast<double>(i + 1) * static_cast<double>(p)
                       / static_cast<double>(p1 + 2)));
        beta(idx) = beta_val;
    }
    return beta;
}


/**
 * @brief Generate response vector y = X * beta + eps.
 *
 * @param X     Design matrix.
 * @param beta  Coefficient vector.
 * @param snr   Signal-to-noise ratio.
 * @param rng   Random number generator.
 *
 * @return Response vector y.
 */
Eigen::VectorXd make_response(const Eigen::MatrixXd& X,
                              const Eigen::VectorXd& beta,
                              double snr, std::mt19937& rng) {

    const Eigen::VectorXd signal = X * beta;
    const Eigen::Index n = X.rows();
    const double sigma = std::sqrt(signal.squaredNorm() / (static_cast<double>(n) * snr));

    std::normal_distribution<double> dist(0.0, sigma);
    Eigen::VectorXd y = signal;
    for (Eigen::Index i = 0; i < n; ++i) { y(i) += dist(rng); }

    y.array() -= y.mean();
    return y;
}


/**
 * @brief Get support indices from a beta vector.
 *
 * @param beta  Coefficient vector.
 *
 * @return Vector of indices corresponding to non-zero entries in beta.
 */
std::vector<std::size_t> support_from_beta(const Eigen::VectorXd& beta) {
    std::vector<std::size_t> support;
    for (Eigen::Index j = 0; j < beta.size(); ++j) {
        if (std::abs(beta(j)) > 1e-12) {
            support.push_back(static_cast<std::size_t>(j));
        }
    }
    return support;
}


// ── DGP: AR(1) column correlation ────────────────────────────────────────────

/**
 * @brief Data generating process with AR(1) column correlation structure.
 *
 * @param n         Number of observations.
 * @param p         Number of predictors.
 * @param p1        Number of active predictors (non-zero coefficients).
 * @param rho       AR(1) correlation coefficient.
 * @param snr       Signal-to-noise ratio.
 * @param beta_val  Coefficient value for active predictors.
 * @param seed      Random seed.
 *
 * @return DGPData  Generated data.
 */
DGPData dgp_ar1(Eigen::Index n,
                Eigen::Index p,
                Eigen::Index p1,
                double rho,
                double snr,
                double beta_val,
                unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    // x_1 ~ N(0, I),  x_j = rho * x_{j-1} + sqrt(1 - rho^2) * w_j
    Eigen::MatrixXd X(n, p);
    const double innovation_sd = std::sqrt(1.0 - rho * rho);
    for (Eigen::Index i = 0; i < n; ++i) {
        X(i, 0) = norm(rng);
        for (Eigen::Index j = 1; j < p; ++j) {
            X(i, j) = rho * X(i, j - 1) + innovation_sd * norm(rng);
        }
    }

    // Standardise columns (scale to unit variance)
    for (Eigen::Index j = 0; j < p; ++j) {
        X.col(j).array() -= X.col(j).mean();
        const double sd = std::sqrt(X.col(j).squaredNorm()
                                    / static_cast<double>(n - 1));
        if (sd > 1e-12) { X.col(j) /= sd; }
    }

    auto beta = make_beta(p, p1, beta_val);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support_from_beta(beta)};
}


// ── DGP: Equicorrelation ─────────────────────────────────────────────────────

/**
 * @brief Data generating process with equicorrelation structure.
 *
 * @param n        Number of observations.
 * @param p        Number of predictors.
 * @param p1       Number of active predictors (non-zero coefficients).
 * @param rho      Equicorrelation coefficient.
 * @param snr      Signal-to-noise ratio.
 * @param beta_val Coefficient value for active predictors.
 * @param seed     Random seed.
 *
 * @return DGPData Generated data.
 */
DGPData dgp_equi(Eigen::Index n,
                 Eigen::Index p,
                 Eigen::Index p1,
                 double rho,
                 double snr,
                 double beta_val,
                 unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    // x_j = sqrt(rho) * z + sqrt(1 - rho) * w_j
    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double f_i = norm(rng);
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = load_f * f_i + load_eta * norm(rng);
        }
    }

    // Standardise columns
    for (Eigen::Index j = 0; j < p; ++j) {
        X.col(j).array() -= X.col(j).mean();
        const double sd = std::sqrt(X.col(j).squaredNorm()
                                    / static_cast<double>(n - 1));
        if (sd > 1e-12) { X.col(j) /= sd; }
    }

    auto beta = make_beta(p, p1, beta_val);
    auto y    = make_response(X, beta, snr, rng);
    return {
        std::move(X), std::move(y), support_from_beta(beta)
    };
}


// ── DGP: Block-Equicorrelation ───────────────────────────────────────────────

/**
 * @brief Data generating process with block-equicorrelation structure.
 *
 * @param n         Number of observations.
 * @param p         Number of predictors.
 * @param p1        Number of active predictors (non-zero coefficients).
 * @param rho       Within-block equicorrelation coefficient.
 * @param n_blocks  Number of contiguous blocks.
 * @param snr       Signal-to-noise ratio.
 * @param beta_val  Coefficient value for active predictors.
 * @param seed      Random seed.
 *
 * @return DGPData  Generated data.
 */
DGPData dgp_block_equi(Eigen::Index n,
                        Eigen::Index p,
                        Eigen::Index p1,
                        double rho,
                        std::size_t n_blocks,
                        double snr,
                        double beta_val,
                        unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    // Block sizes: base_size + 1 for the first (p % n_blocks) blocks
    const auto p_sz = static_cast<std::size_t>(p);
    const std::size_t base_size = p_sz / n_blocks;
    const std::size_t remainder = p_sz % n_blocks;

    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(n, p);
    std::size_t col_idx = 0;

    for (std::size_t k = 0; k < n_blocks; ++k) {
        const std::size_t bk = base_size + (k < remainder ? 1 : 0);

        // Shared factor for block k
        Eigen::VectorXd z_k(n);
        for (Eigen::Index i = 0; i < n; ++i) { z_k(i) = norm(rng); }

        // x_j = sqrt(rho) * z_k + sqrt(1 - rho) * w_j
        for (std::size_t j = 0; j < bk; ++j) {
            const auto col = static_cast<Eigen::Index>(col_idx + j);
            for (Eigen::Index i = 0; i < n; ++i) {
                X(i, col) = load_f * z_k(i) + load_eta * norm(rng);
            }
        }

        col_idx += bk;
    }

    // Standardise columns
    for (Eigen::Index j = 0; j < p; ++j) {
        X.col(j).array() -= X.col(j).mean();
        const double sd = std::sqrt(X.col(j).squaredNorm()
                                    / static_cast<double>(n - 1));
        if (sd > 1e-12) { X.col(j) /= sd; }
    }

    auto beta = make_beta(p, p1, beta_val);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support_from_beta(beta)};
}

} // anonymous namespace


// ==============================================================================
// Shared Screen-TRex runner
// ==============================================================================

/**
 * @brief Run the Screen-TRex selector on the given data.
 *
 * @param label     Label for the run.
 * @param data      Data to be used.
 * @param method    Screen-TRex method to use.
 * @param bootstrap Whether to use bootstrap.
 * @param seed      Random seed.
 */
static void run_screen_trex(const std::string& label,
                            DGPData& data,
                            ScreenTRexMethod method,
                            bool bootstrap,
                            int seed)
{
    std::cout << "Running " << label << "...\n";

    Eigen::Map<Eigen::MatrixXd> X_map(
        data.X.data(), data.X.rows(), data.X.cols()
    );
    Eigen::Map<Eigen::VectorXd> y_map(
        data.y.data(), data.y.size()
    );

    auto trex_ctrl = make_trex_control();
    auto screen_ctrl = make_screen_control(method, bootstrap);

    ScreenTRexSelector sctrex(
        X_map, y_map, screen_ctrl, trex_ctrl,
        seed, /*verbose=*/true
    );
    sctrex.select();

    print_screen_result(sctrex.getScreenResult(),
                        sctrex.getSelectedIndices(),
                        data.true_support);
}


// ==============================================================================
// Demo 01: DA-AR1 Screen-TRex on AR(1)-correlated data
// ==============================================================================

/**
 * @brief Demo 01: DA-AR1 Screen-TRex on AR(1)-correlated data.
 */
void demo_DA_AR1()
{
    cdiag::print_section_header(
        "Demo 01: DA-AR1 Screen-TRex on AR(1) Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    const double rho = 0.5, snr = 1.0, beta_val = 1.0;

    std::cout << "DGP: AR(1), n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << ", SNR=" << snr << "\n\n";

    auto data = dgp_ar1(n, p, p1, rho, snr, beta_val, 1);

    run_screen_trex("Screen-TRex DA-AR1",
                    data,
                    ScreenTRexMethod::TREX_DA_AR1,
                    false,
                    42);
}


// ==============================================================================
// Demo 02: DA-EQUI Screen-TRex on equicorrelated data
// ==============================================================================

/**
 * @brief Demo 02: DA-EQUI Screen-TRex on equicorrelated data.
 */
void demo_DA_EQUI()
{
    cdiag::print_section_header(
        "Demo 02: DA-EQUI Screen-TRex on Equicorrelated Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    const double rho = 0.4, snr = 1.0, beta_val = 1.0;

    std::cout << "DGP: Equicorrelated, n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << ", SNR=" << snr << "\n\n";

    auto data = dgp_equi(n, p, p1, rho, snr, beta_val, 1);

    run_screen_trex("Screen-TRex DA-EQUI",
                    data,
                    ScreenTRexMethod::TREX_DA_EQUI,
                    false,
                    42);
}


// ==============================================================================
// Demo 03: Ordinary Screen-TRex on AR(1) data (baseline, no DA)
// ==============================================================================

/**
 * @brief Demo 03: Ordinary Screen-TRex on AR(1) data (baseline, no DA).
 */
void demo_Ordinary_on_AR1()
{
    cdiag::print_section_header(
        "Demo 03: Ordinary Screen-TRex on AR(1) Data (no DA correction)"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    const double rho = 0.7, snr = 5.0, beta_val = 3.0;

    std::cout << "DGP: AR(1), n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << ", SNR=" << snr << "\n"
              << "Method: Ordinary Screen-TRex (no DA) — baseline comparison\n\n";

    auto data = dgp_ar1(n, p, p1, rho, snr, beta_val, 1);

    run_screen_trex("Screen-TRex Ordinary (no DA)",
                    data,
                    ScreenTRexMethod::TREX,
                    false,
                    42);
}


// ==============================================================================
// Demo 04: Monte Carlo — Ordinary vs. Bootstrap DA-AR1 on AR(1) data
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo: Ordinary vs. Bootstrap DA-AR1 Screen-TRex
 *        on AR(1)-correlated data.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, SNR) pair.
 * @param rho       AR(1) lag-1 correlation.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_AR1_MonteCarlo(std::size_t num_MC,
                             double     rho      = 0.4,
                             double     beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo 04: DA-AR1 Screen-TRex Monte Carlo on AR(1) Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: AR(1), n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << "\n";

    const std::vector<double> snr_values =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    // Methods to compare
    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary", ScreenTRexMethod::TREX, false},
        {"Screen-TRex Bootstrap", ScreenTRexMethod::TREX, true},
        {"Screen-TRex Ordinary+DA-AR1", ScreenTRexMethod::TREX_DA_AR1,
            false},
        {"Screen-TRex Bootstrap+DA-AR1", ScreenTRexMethod::TREX_DA_AR1,
            true}
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    // -------------------------------------------------------------------
    // Method loop
    // -------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        // SNR loop -------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                auto d = dgp_ar1(n, p, p1, rho, snr, beta_val, seed);
                return {std::move(d.X), std::move(d.y), std::move(d.true_support)};
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(snr_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  SNR=" << std::fixed << std::setprecision(2) << snr;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "d03_screen_trex_da_ar1_mc_n" << n << "_p" << p
         << "_rho" << std::fixed << std::setprecision(2) << rho;
    save_and_print_mc_results(num_MC, stem.str(), snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// Demo 05: Monte Carlo — Ordinary vs. Bootstrap DA-EQUI on equicorrelated data
// ==============================================================================

/**
 * @brief SNR-sweep Monte Carlo: Ordinary vs. Bootstrap DA-EQUI Screen-TRex
 *        on equicorrelated data.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, SNR) pair.
 * @param rho       Equicorrelation coefficient.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_Equi_MonteCarlo(std::size_t num_MC,
                              double     rho      = 0.4,
                              double     beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo 05: DA-EQUI Screen-TRex Monte Carlo on Equicorrelated Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: Equicorrelated, n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << "\n";

    const std::vector<double> snr_values =
        {0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0};

    // Methods to compare
    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary", ScreenTRexMethod::TREX, false},
        {"Screen-TRex Bootstrap", ScreenTRexMethod::TREX, true},
        {"Screen-TRex Ordinary DA-EQUI", ScreenTRexMethod::TREX_DA_EQUI,
             false},
        {"Screen-TRex Bootstrap DA-EQUI", ScreenTRexMethod::TREX_DA_EQUI,
            true}
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(snr_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    // -------------------------------------------------------------------
    // Method loop
    // -------------------------------------------------------------------
    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        // SNR loop -------------------------------------------------------
        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {
            const double snr = snr_values[snr_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                auto d = dgp_equi(n, p, p1, rho, snr, beta_val, seed);
                return {std::move(d.X), std::move(d.y), std::move(d.true_support)};
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(snr_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  SNR=" << std::fixed << std::setprecision(2) << snr;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(snr_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "d03_screen_trex_da_equi_mc_n" << n << "_p" << p
         << "_rho" << std::fixed << std::setprecision(2) << rho;
    save_and_print_mc_results(num_MC, stem.str(), snr_values, methods,
                              fdr_map, tpr_map, est_fdr_map);
}


// ==============================================================================
// Demo 06: rho Sweep — DA-AR1 on AR(1)-correlated data
// ==============================================================================

/**
 * @brief rho-sweep Monte Carlo: Ordinary vs. Bootstrap vs. DA-AR1 Screen-TRex
 *        on AR(1)-correlated data at fixed SNR.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, rho) pair.
 * @param snr       Fixed signal-to-noise ratio.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_AR1_RhoSweep_MonteCarlo(std::size_t num_MC,
                                     double      snr      = 1.0,
                                     double      beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo 06: DA-AR1 Screen-TRex rho Sweep on AR(1) Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: AR(1), n=" << n << ", p=" << p << ", p1=" << p1
              << ", SNR=" << snr << "\n";

    const std::vector<double> rho_values =
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary",         ScreenTRexMethod::TREX,
            false},
        {"Screen-TRex Bootstrap",        ScreenTRexMethod::TREX,
            true },
        {"Screen-TRex Ordinary+DA-AR1",  ScreenTRexMethod::TREX_DA_AR1,
            false},
        {"Screen-TRex Bootstrap+DA-AR1", ScreenTRexMethod::TREX_DA_AR1,
            true }
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(rho_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        for (std::size_t rho_idx = 0; rho_idx < rho_values.size(); ++rho_idx) {
            const double rho = rho_values[rho_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                auto d = dgp_ar1(n, p, p1, rho, snr, beta_val, seed);
                return {std::move(d.X), std::move(d.y), std::move(d.true_support)};
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(rho_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  rho=" << std::fixed << std::setprecision(2) << rho;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(rho_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "d03_screen_trex_da_ar1_rho_sweep_n" << n << "_p" << p
         << "_snr" << std::fixed << std::setprecision(2) << snr;
    save_and_print_mc_results(num_MC, stem.str(), rho_values, methods,
                              fdr_map, tpr_map, est_fdr_map, "rho");
}


// ==============================================================================
// Demo 07: rho Sweep — DA-EQUI on equicorrelated data
// ==============================================================================

/**
 * @brief rho-sweep Monte Carlo: Ordinary vs. Bootstrap vs. DA-EQUI Screen-TRex
 *        on equicorrelated data at fixed SNR.  Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, rho) pair.
 * @param snr       Fixed signal-to-noise ratio.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_Equi_RhoSweep_MonteCarlo(
    std::size_t num_MC,
    double      snr      = 1.0,
    double      beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo 07: DA-EQUI Screen-TRex rho Sweep on Equicorrelated Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: Equicorrelated, n=" << n << ", p=" << p << ", p1=" << p1
              << ", SNR=" << snr << "\n";

    const std::vector<double> rho_values =
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary",          ScreenTRexMethod::TREX,
            false},
        {"Screen-TRex Bootstrap",         ScreenTRexMethod::TREX,
            true },
        {"Screen-TRex Ordinary DA-EQUI",  ScreenTRexMethod::TREX_DA_EQUI,
             false},
        {"Screen-TRex Bootstrap DA-EQUI", ScreenTRexMethod::TREX_DA_EQUI,
             true }
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(rho_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap);

        for (std::size_t rho_idx = 0; rho_idx < rho_values.size(); ++rho_idx) {
            const double rho = rho_values[rho_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                auto d = dgp_equi(n, p, p1, rho, snr, beta_val, seed);
                return {std::move(d.X), std::move(d.y), std::move(d.true_support)};
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(rho_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  rho=" << std::fixed << std::setprecision(2) << rho;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(rho_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "d03_screen_trex_da_equi_rho_sweep_n" << n << "_p" << p
         << "_snr" << std::fixed << std::setprecision(2) << snr;
    save_and_print_mc_results(num_MC, stem.str(), rho_values, methods,
                              fdr_map, tpr_map, est_fdr_map, "rho");
}


// ==============================================================================
// Demo 08: DA-BLOCK-EQUI Screen-TRex on block-equicorrelated data
// ==============================================================================

/**
 * @brief Demo 08: DA-BLOCK-EQUI Screen-TRex on block-equicorrelated data.
 */
void demo_DA_BLOCK_EQUI()
{
    cdiag::print_section_header(
        "Demo 08: DA-BLOCK-EQUI Screen-TRex on Block-Equicorrelated Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    const double rho = 0.5, snr = 1.0, beta_val = 1.0;
    const std::size_t n_blocks = 5;

    std::cout << "DGP: Block-Equicorr, n=" << n << ", p=" << p << ", p1=" << p1
              << ", rho=" << rho << ", n_blocks=" << n_blocks
              << ", SNR=" << snr << "\n\n";

    auto data = dgp_block_equi(n, p, p1, rho, n_blocks, snr, beta_val, 1);

    Eigen::Map<Eigen::MatrixXd> X_map(
        data.X.data(), data.X.rows(), data.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(
        data.y.data(), data.y.size());

    auto trex_ctrl = make_trex_control();
    auto screen_ctrl = make_screen_control(
        ScreenTRexMethod::TREX_DA_BLOCK_EQUI, false, n_blocks);

    ScreenTRexSelector sctrex(
        X_map,
        y_map,
        screen_ctrl,
        trex_ctrl,
        42,
        true
    );
    sctrex.select();

    print_screen_result(sctrex.getScreenResult(),
                        sctrex.getSelectedIndices(),
                        data.true_support);
}


// ==============================================================================
// Demo 09: rho Sweep — DA-BLOCK-EQUI on block-equicorrelated data
// ==============================================================================

/**
 * @brief rho-sweep Monte Carlo: Ordinary vs. Bootstrap vs. DA-BLOCK-EQUI
 *        Screen-TRex on block-equicorrelated data at fixed SNR.
 *        Parallelised via OpenMP.
 *
 * @param num_MC    Number of Monte Carlo runs per (method, rho) pair.
 * @param n_blocks  Number of blocks in DGP.
 * @param snr       Fixed signal-to-noise ratio.
 * @param beta_val  Coefficient magnitude for active variables.
 */
void demo_DA_BlockEqui_RhoSweep_MonteCarlo(
    std::size_t num_MC,
    std::size_t n_blocks = 5,
    double      snr      = 1.0,
    double      beta_val = 1.0)
{
    std::cout.setf(std::ios::unitbuf);
    cdiag::print_section_header(
        "Demo 09: DA-BLOCK-EQUI Screen-TRex rho Sweep on Block-Equicorr Data"
    );

    const Eigen::Index n = 300, p = 1000, p1 = 10;
    std::cout << "DGP: Block-Equicorr, n=" << n << ", p=" << p << ", p1=" << p1
              << ", n_blocks=" << n_blocks << ", SNR=" << snr << "\n";

    const std::vector<double> rho_values =
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    const std::vector<ScreenMethodInfo> methods = {
        {"Screen-TRex Ordinary",          ScreenTRexMethod::TREX,
            false},
        {"Screen-TRex Bootstrap",         ScreenTRexMethod::TREX,
            true },
        {"Screen-TRex Ord+DA-BLOCK-EQUI", ScreenTRexMethod::TREX_DA_BLOCK_EQUI,
            false},
        {"Screen-TRex Boot+DA-BLOCK-EQUI", ScreenTRexMethod::TREX_DA_BLOCK_EQUI,
            true }
    };

    std::map<std::string, Eigen::VectorXd> est_fdr_map, fdr_map, tpr_map;
    for (const auto& m : methods) {
        const auto len = static_cast<Eigen::Index>(rho_values.size());
        est_fdr_map[m.name] = Eigen::VectorXd::Zero(len);
        fdr_map[m.name]     = Eigen::VectorXd::Zero(len);
        tpr_map[m.name]     = Eigen::VectorXd::Zero(len);
    }

    auto trex_ctrl = make_trex_control();

    for (const auto& method : methods) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Method: " << method.name << "\n";
        std::cout << std::string(70, '=') << "\n";

        auto screen_ctrl = make_screen_control(method.method, method.bootstrap,
                                               n_blocks);

        for (std::size_t rho_idx = 0; rho_idx < rho_values.size(); ++rho_idx) {
            const double rho = rho_values[rho_idx];

            const auto make_data = [=](unsigned seed) -> ScrDGPData {
                auto d = dgp_block_equi(n, p, p1, rho, n_blocks, snr, beta_val, seed);
                return {std::move(d.X), std::move(d.y), std::move(d.true_support)};
            };

            const unsigned base_seed = 42u + static_cast<unsigned>(rho_idx) * 1000u;

            std::ostringstream lbl;
            lbl << method.name << "  rho=" << std::fixed << std::setprecision(2) << rho;
            const auto result = run_mc_trials_screen_trex(
                num_MC, lbl.str(), make_data, trex_ctrl, screen_ctrl, base_seed);

            const auto idx = static_cast<Eigen::Index>(rho_idx);
            est_fdr_map[method.name](idx) = result.avg_est_fdr;
            fdr_map[method.name](idx)     = result.avg_fdr;
            tpr_map[method.name](idx)     = result.avg_tpr;
        }
    }

    std::ostringstream stem;
    stem << "d03_screen_trex_da_block_equi_rho_sweep_n" << n << "_p" << p
         << "_blocks" << n_blocks
         << "_snr" << std::fixed << std::setprecision(2) << snr;
    save_and_print_mc_results(num_MC, stem.str(), rho_values, methods,
                              fdr_map, tpr_map, est_fdr_map, "rho");
}


// ==============================================================================
// Main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);

    // Demo 01: Ordinary on AR(1) — baseline without DA correction
    demo_Ordinary_on_AR1();

    // Demo 02: DA-AR1 on AR(1) data
    demo_DA_AR1();

    // Demo 03: DA-EQUI on equicorrelated data
    demo_DA_EQUI();

    // Demo 04: DA-BLOCK-EQUI on block-equicorrelated data
    demo_DA_BLOCK_EQUI();

    // Demo 05: Monte Carlo — Ordinary vs. Bootstrap vs. DA-AR1 awareness on AR(1) data
    demo_DA_AR1_MonteCarlo(500, 0.5, 1.0);

    // Demo 06: Monte Carlo — Ordinary vs. Bootstrap vs. DA-EQUI awareness on equicorrelated data
    demo_DA_Equi_MonteCarlo(500, 0.4, 3.0);

    // Demo 07: rho sweep — Ordinary vs. Bootstrap vs. DA-AR1 on AR(1) data
    demo_DA_AR1_RhoSweep_MonteCarlo(500, 1.0, 1.0);

    // Demo 08: rho sweep — Ordinary vs. Bootstrap vs. DA-EQUI on equicorrelated data
    demo_DA_Equi_RhoSweep_MonteCarlo(500, 1.0, 3.0);

    // Demo 09: rho sweep — Ordinary vs. Bootstrap vs. DA-BLOCK-EQUI on block-equicorr data
    demo_DA_BlockEqui_RhoSweep_MonteCarlo(500, 5, 1.0, 3.0);

    return 0;
}
