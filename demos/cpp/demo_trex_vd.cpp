// ===================================================================================
// demo_trex_vd.cpp
// ===================================================================================
/**
 * @file demo_trex_vd.cpp
 *
 * @brief Small end-to-end scenario for the Virtual-Dummy T-Rex (TRexVD).
 *
 * @details Generates a sparse high-dimensional linear model with
 *          utils_datagen::SyntheticData, preprocesses it with the package's
 *          data normalizer (column-centered, unit-L2 X; centered y — the
 *          contract of the VD solvers), then runs TRexVD in CalibrateT mode
 *          and reports selection quality against the known ground truth.
 *
 *          Mirrors the smoke tests in tests/test_trex_selector_methods/
 *          test_trex_vd.cpp, but as a runnable, human-readable scenario.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// project trex includes
#include <trex_selector_methods/trex_vd/trex_vd.hpp>
#include <trex_selector_methods/trex_utils/trex_data_normalizer.hpp>
#include <utils/datageneration/utils_datagen.hpp>

// ===================================================================================

namespace {

namespace tvd = trex::trex_selector_methods::trex_vd;
namespace dn  = trex::trex_selector_methods::utils::data_normalizer;
namespace dg  = trex::utils::datageneration::datagen;

void report(const char* label,
            const tvd::TRexVDResult& result,
            const std::vector<std::size_t>& support,
            int p)
{
    std::vector<int> sel(result.selected_var.data(),
                         result.selected_var.data() + result.selected_var.size());
    std::sort(sel.begin(), sel.end());

    int tp = 0;
    for (auto j : support)
        if (std::find(sel.begin(), sel.end(), (int)j) != sel.end()) ++tp;
    const int fp = (int)sel.size() - tp;
    const double fdp = sel.empty() ? 0.0 : double(fp) / double(sel.size());
    const double tpr = support.empty() ? 1.0 : double(tp) / double(support.size());

    std::cout << "\n--- " << label << " ---\n"
              << "selected " << sel.size() << " of p=" << p
              << " variables at v*=" << result.v_thresh
              << ", T*=" << result.T_stop
              << ", L=" << result.num_dummies << "\n"
              << "indices:";
    for (int j : sel) std::cout << " " << j;
    std::cout << "\nTP=" << tp << "  FP=" << fp
              << "  FDP=" << fdp << "  TPR=" << tpr << "\n";
}

}  // namespace

// ===================================================================================

int main() {

    // ---- 1. Scenario: sparse linear model, p >> n ----
    constexpr int n = 300;
    constexpr int p = 1'000'000;
    const std::vector<std::size_t> support = {0, 100, 200, 300, 400};
    const std::vector<double> coefs(support.size(), 1.0);
    constexpr double snr = 3.0;
    //constexpr int seed = 42;
    const int seed = std::random_device{}();

    std::cout << "[demo_trex_vd] generating data: n=" << n << " p=" << p
              << " |support|=" << support.size() << " snr=" << snr
              << " seed=" << seed << "\n";

    dg::SyntheticData data(n, p, support, coefs, snr, seed);

    // ---- 2. Preprocessing (VD solver contract) ----
    Eigen::MatrixXd X = data.getX();
    Eigen::VectorXd y = data.getY();

    dn::NormalizationParams params;
    dn::centerAndL2NormalizeX(X, params);
    dn::centerY(y, params);

    // ---- 3. FDR-controlled selection with TRexVD ----
    tvd::TRexVDOptions opt;
    opt.tFDR      = 0.1;
    opt.K         = 20;
    opt.L_factor  = 5;
    opt.solver    = tvd::SolverType::OMP;
    opt.calib     = tvd::CalibMode::CalibrateT;
    opt.dummy_law = tvd::VDDummyLaw::Spherical;
    opt.seed      = seed;
    opt.verbose   = true;

    tvd::TRexVD selector(opt);
    tvd::TRexVDResult result = selector.run(X, y);

    report("TRexVD / LARS / CalibrateT", result, support, p);

    // ---- 4. Same scenario, fixed-T mode for comparison ----
    opt.calib   = tvd::CalibMode::FixedTL;
    opt.T_stop  = 5;
    opt.verbose = false;

    tvd::TRexVD selector_fixed(opt);
    tvd::TRexVDResult result_fixed = selector_fixed.run(X, y);

    report("TRexVD / LARS / FixedTL (T=5)", result_fixed, support, p);

    return 0;
}
