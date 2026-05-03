// ==============================================================================
// demo_trex_gvs_01_basic.cpp
// ==============================================================================
/**
 * @file demo_trex_gvs_01_basic.cpp
 *
 * @brief Basic single-run demo of the TRex+GVS selector on a block-AR(1) DGP.
 *
 * @details
 *  The design has B = 10 blocks of size p_block = 10.  Within a block,
 *  predictors follow an AR(1) correlation rho = 0.9; across blocks the
 *  predictors are independent.  The true support places one active
 *  variable in each of the first s = 5 blocks.  Because variables within
 *  a block are highly correlated, vanilla T-Rex (TLARS) tends to randomly
 *  pick at most one in-block representative; T-Rex+GVS should select the
 *  block-aware structure systematically.
 */
// ==============================================================================

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

using namespace trex::trex_selector_methods;
using trex_gvs::GVSType;
using trex_gvs::TRexGVSControlParameter;
using trex_gvs::TRexGVSSelector;
using trex_core::TRexControlParameter;
using trex_core::LLoopStrategy;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;


// ==============================================================================
// Block-AR(1) data generator
// ==============================================================================

struct DGPResult {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    std::vector<Eigen::Index> true_support;
    std::vector<Eigen::Index> block_id;   // 0-based block id per column.
};


static DGPResult make_block_ar1(std::size_t n,
                                std::size_t p_block,
                                std::size_t n_blocks,
                                double rho,
                                std::size_t s,
                                double snr,
                                unsigned seed)
{
    const std::size_t p = p_block * n_blocks;
    DGPResult out;
    out.X.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p));
    out.block_id.assign(p, 0);

    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Build per-block AR(1) Cholesky factor.
    Eigen::MatrixXd Sigma(p_block, p_block);
    for (std::size_t i = 0; i < p_block; ++i) {
        for (std::size_t j = 0; j < p_block; ++j) {
            Sigma(i, j) = std::pow(rho,
                static_cast<double>(i > j ? i - j : j - i));
        }
    }
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    Eigen::MatrixXd L = llt.matrixL();

    // Generate each block independently.
    for (std::size_t b = 0; b < n_blocks; ++b) {
        Eigen::MatrixXd Z(static_cast<Eigen::Index>(n),
                          static_cast<Eigen::Index>(p_block));
        for (Eigen::Index i = 0; i < Z.rows(); ++i)
            for (Eigen::Index j = 0; j < Z.cols(); ++j)
                Z(i, j) = N01(rng);

        Eigen::MatrixXd block = Z * L.transpose();
        out.X.block(0, static_cast<Eigen::Index>(b * p_block),
                    static_cast<Eigen::Index>(n),
                    static_cast<Eigen::Index>(p_block)) = block;

        for (std::size_t j = 0; j < p_block; ++j) {
            out.block_id[b * p_block + j] = static_cast<Eigen::Index>(b);
        }
    }

    // True support: one active variable in each of the first s blocks.
    out.true_support.reserve(s);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));
    for (std::size_t b = 0; b < s; ++b) {
        // First column of each block is "active".
        const auto j = static_cast<Eigen::Index>(b * p_block);
        out.true_support.push_back(j);
        beta(j) = 1.0;
    }

    // y = X * beta + noise (calibrate noise to SNR).
    Eigen::VectorXd signal = out.X * beta;
    const double signal_var = (signal.array() - signal.mean()).square().mean();
    const double sigma_e    = std::sqrt(signal_var / snr);

    out.y.resize(static_cast<Eigen::Index>(n));
    for (Eigen::Index i = 0; i < out.y.size(); ++i) {
        out.y(i) = signal(i) + sigma_e * N01(rng);
    }

    return out;
}


// ==============================================================================
// Selection metrics
// ==============================================================================

struct Metrics {
    std::size_t TP{0};
    std::size_t FP{0};
    std::size_t FN{0};
    double TPR{0.0};
    double FDP{0.0};
};


static Metrics evaluate(const std::vector<Eigen::Index>& selected,
                        const std::vector<Eigen::Index>& truth)
{
    Metrics m;
    std::vector<bool> truth_mask;
    Eigen::Index max_idx = -1;
    for (auto i : truth)    if (i > max_idx) max_idx = i;
    for (auto i : selected) if (i > max_idx) max_idx = i;
    truth_mask.assign(static_cast<std::size_t>(max_idx + 1), false);
    for (auto i : truth) truth_mask[static_cast<std::size_t>(i)] = true;

    for (auto i : selected) {
        if (truth_mask[static_cast<std::size_t>(i)]) ++m.TP;
        else ++m.FP;
    }
    m.FN = truth.size() - m.TP;
    m.TPR = truth.empty() ? 0.0 : static_cast<double>(m.TP) /
                                   static_cast<double>(truth.size());
    const std::size_t R = m.TP + m.FP;
    m.FDP = R == 0 ? 0.0 : static_cast<double>(m.FP) / static_cast<double>(R);
    return m;
}


static void print_metrics(const std::string& tag, const Metrics& m,
                          double tFDR)
{
    std::cout << "\n----- " << tag << " -----\n";
    std::cout << "  TP  = " << m.TP << "\n";
    std::cout << "  FP  = " << m.FP << "\n";
    std::cout << "  FN  = " << m.FN << "\n";
    std::cout << "  TPR = " << std::fixed << std::setprecision(3) << m.TPR << "\n";
    std::cout << "  FDP = " << std::fixed << std::setprecision(3) << m.FDP
              << "  (target FDR = " << tFDR << ")\n";
}


// ==============================================================================
// Main
// ==============================================================================

int main()
{
    std::cout.setf(std::ios::unitbuf);

    // ---- DGP ---------------------------------------------------------
    constexpr std::size_t n        = 200;
    constexpr std::size_t p_block  = 10;
    constexpr std::size_t n_blocks = 20;
    constexpr std::size_t s        = 5;
    constexpr double      rho      = 0.9;
    constexpr double      snr      = 2.0;
    constexpr unsigned    seed     = 42;
    constexpr double      tFDR     = 0.10;

    std::cout << "================================================================\n";
    std::cout << " TRex+GVS Demo  |  n=" << n
              << " p=" << (p_block * n_blocks)
              << " s=" << s
              << " rho=" << rho
              << " snr=" << snr << "\n";
    std::cout << "================================================================\n";

    auto dat = make_block_ar1(n, p_block, n_blocks, rho, s, snr, seed);

    Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.size());

    // ---- T-Rex base control -----------------------------------------
    TRexControlParameter trex_ctrl;
    trex_ctrl.K                    = 20;
    trex_ctrl.max_dummy_multiplier = 5;
    trex_ctrl.lloop_strategy       = LLoopStrategy::HCONCAT;
    trex_ctrl.tloop_stagnation_stop = false;

    // ---- (1) IEN with prior groups ----------------------------------
    {
        std::cout << "\n>>> Running TRex+GVS (IEN, prior block groups) <<<\n";

        TRexGVSControlParameter gvs_ctrl;
        gvs_ctrl.gvs_type     = GVSType::IEN;
        gvs_ctrl.prior_groups = dat.block_id;

        auto trex_ctrl_run = trex_ctrl;
        trex_ctrl_run.solver_type = SolverTypeForTRex::TLASSO;

        TRexGVSSelector sel(X_map, y_map, tFDR, gvs_ctrl,
                            trex_ctrl_run, /*seed=*/static_cast<int>(seed),
                            /*verbose=*/true);
        sel.select();
        const auto& res = sel.getGVSResult();

        std::cout << "lambda_2 used  = " << res.lambda2_used << "\n";
        std::cout << "M (clusters)   = " << res.max_clusters << "\n";
        std::cout << "T_stop         = " << res.T_stop << "\n";

        std::vector<Eigen::Index> selected_idx;
        for (Eigen::Index i = 0; i < res.selected_var.size(); ++i) {
            if (res.selected_var(i) != 0) selected_idx.push_back(i);
        }
        print_metrics("IEN / prior_groups", evaluate(selected_idx, dat.true_support), tFDR);
    }

    // ---- (2) EN with hierarchical clustering ------------------------
    {
        std::cout << "\n>>> Running TRex+GVS (EN, hierarchical clustering) <<<\n";

        TRexGVSControlParameter gvs_ctrl;
        gvs_ctrl.gvs_type   = GVSType::EN;
        gvs_ctrl.corr_max   = 0.5;
        gvs_ctrl.hc_linkage = trex::ml_methods::clustering::hierarchical::
                              agglomerative::LinkageMethod::Single;

        auto trex_ctrl_run = trex_ctrl;
        trex_ctrl_run.solver_type            = SolverTypeForTRex::TENET;
        trex_ctrl_run.solver_params.lambda2  = 0.1;

        TRexGVSSelector sel(X_map, y_map, tFDR, gvs_ctrl,
                            trex_ctrl_run, /*seed=*/static_cast<int>(seed),
                            /*verbose=*/true);
        sel.select();
        const auto& res = sel.getGVSResult();

        std::cout << "lambda_2 used  = " << res.lambda2_used << "\n";
        std::cout << "M (clusters)   = " << res.max_clusters << "\n";
        std::cout << "T_stop         = " << res.T_stop << "\n";

        std::vector<Eigen::Index> selected_idx;
        for (Eigen::Index i = 0; i < res.selected_var.size(); ++i) {
            if (res.selected_var(i) != 0) selected_idx.push_back(i);
        }
        print_metrics("EN / clustered", evaluate(selected_idx, dat.true_support), tFDR);
    }

    return 0;
}
