// ==============================================================================
// demo_trex_sd_mc_variable_support.cpp
// ==============================================================================
/**
 * @file demo_trex_sd_mc_variable_support.cpp
 *
 * @brief Monte Carlo simulation for the Sparse-Dummy T-Rex Selector (TRexSD)
 *        with variable support indices but same cardinality across all Monte
 *        Carlo trials.
 *
 * @details Mirrors the classical demo_trex_03_mc_sim_variable_support one to
 *          one — same DGP (datagen::SyntheticData, fresh random support and
 *          noise per trial), same SNR grid, same aggregation and table format
 *          — with the TRexSD configs (SD-LARS / SD-OMP / SD-AFS under the
 *          General dummy law) in place of the classical solver list, plus
 *          two reference families on the same paired data — TRexVD (Gaussian
 *          virtual dummies) and the classical explicit-dummy TRexSelector
 *          (Normal dummies, ONDEMAND generation so nothing is stored), each
 *          with the same algorithm triple (LARS / OMP / AFS rho=0.3): if SD
 *          is to be an alternative, it has to hold up against both.
 *
 *          TRexSD runs the genuine T-Rex procedure (adaptive L-loop, then
 *          the (T, v) grid; classic-parity defaults). Trials are the
 *          parallel axis; each trial owns its data and selector.
 *
 *          Usage: demo_trex_sd_mc_variable_support [num_MC]   (default 200)
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_sd/trex_sd.hpp>
#include <trex_selector_methods/trex_vd/trex_vd.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdiagnostics = trex::utils::eval::cdiagnostics;
namespace datagen = trex::utils::datageneration::datagen;
namespace dummygen = trex::utils::datageneration::dummygen;
namespace rates = trex::utils::eval::rates;
namespace tsd = trex::trex_selector_methods::trex_sd;
namespace tvd = trex::trex_selector_methods::trex_vd;
namespace tcx = trex::trex_selector_methods::trex_core;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================
// Configuration helpers
// ==============================================================================

static tsd::TRexSDOptions make_base_sd_options() {
    tsd::TRexSDOptions opt;
    opt.K            = 20;
    opt.max_L_factor = 10;                          // classic max_dummy_multiplier
    opt.T_stop       = -1;                          // auto: min(L, ceil(n/2))
    opt.rho_d        = 0.047;                       // k = 7 at n = 300
    opt.calib        = tsd::CalibMode::CalibrateBoth;  // adaptive L-loop + (T, v)
    // stagnation_stop stays "auto": disabled for LARS, enabled for greedy
    opt.verbose      = false;
    opt.n_threads    = 1;                           // trials are the parallel axis
    return opt;
}

static tvd::TRexVDOptions make_base_vd_options() {
    tvd::TRexVDOptions opt;
    opt.K            = 20;
    opt.L_factor     = 10;
    opt.max_L_factor = 10;                          // same L-loop ceiling as SD
    opt.T_stop       = -1;                          // auto: min(L, n/2)
    opt.calib        = tvd::CalibMode::CalibrateBoth;
    opt.verbose      = false;
    opt.n_threads    = 1;                           // trials are the parallel axis
    return opt;
}

static tcx::TRexControlParameter make_base_classic_control() {
    tcx::TRexControlParameter ctrl;
    ctrl.K                    = 20;
    ctrl.max_dummy_multiplier = 10;
    ctrl.use_max_T_stop       = true;
    ctrl.dummy_distribution   = dummygen::Distribution::Normal();
    ctrl.lloop_strategy       = tcx::LLoopStrategy::ONDEMAND;  // nothing stored
    // tloop_stagnation_stop stays "auto" (disabled for LARS-path, on for greedy)
    ctrl.parallel_rnd_experiments = false;   // trials are the parallel axis
    return ctrl;
}

/** @brief Selector families compared on the same paired data. */
enum class ConfigFamily : uint8_t { SD, VD, Classic };

/** @brief Selector config descriptor: TRexSD (sparse dummies, General law),
 *         the TRexVD reference (Gaussian virtual dummies), or the classical
 *         explicit-dummy TRexSelector. */
struct DemoConfigInfo {
    std::string       name;
    ConfigFamily      family;
    tsd::SDAlgo       sd_algo;
    tvd::SolverType   vd_algo;
    SolverTypeForTRex classic_solver;
    double            rho;       // AFS blend (ignored by LARS/OMP)
};

static std::vector<DemoConfigInfo> make_default_configs_to_test() {
    using CF = ConfigFamily;
    using ST = SolverTypeForTRex;
    return {
        {"SD-LARS",         CF::SD, tsd::SDAlgo::LARS, tvd::SolverType::LARS, ST::TLARS, 0.0},
        {"SD-OMP",          CF::SD, tsd::SDAlgo::OMP,  tvd::SolverType::LARS, ST::TLARS, 0.0},
        {"SD-AFS_rho_0.3",  CF::SD, tsd::SDAlgo::AFS,  tvd::SolverType::LARS, ST::TLARS, 0.3},
        {"VD-LARS",         CF::VD, tsd::SDAlgo::LARS, tvd::SolverType::LARS, ST::TLARS, 0.0},
        {"VD-OMP",          CF::VD, tsd::SDAlgo::LARS, tvd::SolverType::OMP,  ST::TLARS, 0.0},
        {"VD-AFS_rho_0.3",  CF::VD, tsd::SDAlgo::LARS, tvd::SolverType::AFS,  ST::TLARS, 0.3},
        {"TLARS",           CF::Classic, tsd::SDAlgo::LARS, tvd::SolverType::LARS, ST::TLARS, 0.0},
        {"TOMP",            CF::Classic, tsd::SDAlgo::LARS, tvd::SolverType::LARS, ST::TOMP,  0.0},
        {"TAFS_rho_0.3",    CF::Classic, tsd::SDAlgo::LARS, tvd::SolverType::LARS, ST::TAFS,  0.3},
    };
}

// ==============================================================================
// MC loop infrastructure
// ==============================================================================

/** @brief Data container returned by the DGP factory for one MC trial. */
struct SDTrexDGPData {
    Eigen::MatrixXd          X;
    Eigen::VectorXd          y;
    std::vector<std::size_t> true_support;
};

/** @brief Factory callable: given a trial seed, produce one MC dataset. */
using SDTrexDGPFactory = std::function<SDTrexDGPData(unsigned seed)>;

/** @brief Aggregate MC result for one grid point (config x SNR). */
struct SDTrexGridPointResult {
    double avg_fdr = 0.0;
    double avg_tpr = 0.0;
    double avg_L   = 0.0;   ///< Mean dummy multiplier L/p
    double avg_T   = 0.0;   ///< Mean stopping time T*
    double sd_fdr  = 0.0;
    double sd_tpr  = 0.0;
};

/**
 * @brief Run num_MC parallel MC trials of one config (TRexSD or TRexVD) for
 *        one (config, SNR) point.
 *
 * Trials are the parallel axis (each trial owns its data and selector; the
 * inner K-experiment loop runs single-threaded). The solver contract
 * (X centered with unit-norm columns, y centered — identical for SD and VD)
 * is applied here, after the factory, so DGP factories stay pure.
 */
static SDTrexGridPointResult run_mc_trials(
    std::size_t             num_MC,
    const std::string&      progress_label,
    const SDTrexDGPFactory& make_data,
    double                  tFDR,
    const tsd::TRexSDOptions& base_opts,
    const DemoConfigInfo&   config,
    unsigned                base_seed_offset)
{
    const int iMC = static_cast<int>(num_MC);

    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);
    std::vector<double> L_vec  (num_MC, 0.0);
    std::vector<double> T_vec  (num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        // Solver contract: centered X with unit-norm columns, centered y.
        for (Eigen::Index j = 0; j < dat.X.cols(); ++j) {
            dat.X.col(j).array() -= dat.X.col(j).mean();
            const double nrm = dat.X.col(j).norm();
            if (nrm > 0.0) dat.X.col(j) /= nrm;
        }
        dat.y.array() -= dat.y.mean();

        std::vector<std::size_t> sel;
        double T_star = 0.0, L_mult = 0.0;

        auto from_eigen = [&sel](const Eigen::VectorXi& selected) {
            sel.resize(static_cast<std::size_t>(selected.size()));
            for (int i = 0; i < selected.size(); ++i)
                sel[static_cast<std::size_t>(i)] =
                    static_cast<std::size_t>(selected(i));
        };

        switch (config.family) {
            case ConfigFamily::SD: {
                tsd::TRexSDOptions opt = base_opts;
                opt.tFDR   = tFDR;
                opt.algo   = config.sd_algo;
                opt.solver = tsd::SDSolverType::General;
                if (config.rho > 0.0) opt.rho = config.rho;
                opt.seed   = trial_seed + 900000u;  // decouple selector from DGP seed

                tsd::TRexSD selector(opt);
                tsd::TRexSDResult res = selector.run(dat.X, dat.y);
                from_eigen(res.selected_var);
                T_star = double(res.T_stop);
                L_mult = double(res.num_dummies) / double(dat.X.cols());
                break;
            }
            case ConfigFamily::VD: {
                tvd::TRexVDOptions opt = make_base_vd_options();
                opt.tFDR   = tFDR;
                opt.solver = config.vd_algo;
                if (config.rho > 0.0) opt.rho = config.rho;
                opt.seed   = trial_seed + 900000u;  // decouple selector from DGP seed

                tvd::TRexVD selector(opt);
                tvd::TRexVDResult res = selector.run(dat.X, dat.y);
                from_eigen(res.selected_var);
                T_star = double(res.T_stop);
                L_mult = double(res.num_dummies) / double(dat.X.cols());
                break;
            }
            case ConfigFamily::Classic: {
                tcx::TRexControlParameter ctrl = make_base_classic_control();
                ctrl.solver_type = config.classic_solver;
                if (config.rho > 0.0) ctrl.solver_params.rho_afs = config.rho;

                Eigen::Map<Eigen::MatrixXd> Xm(dat.X.data(), dat.X.rows(), dat.X.cols());
                Eigen::Map<Eigen::VectorXd> ym(dat.y.data(), dat.y.size());
                tcx::TRexSelector selector(Xm, ym, tFDR, ctrl,
                                           static_cast<int>(trial_seed + 900000u),
                                           /*verbose=*/false);
                selector.select();
                sel    = selector.getSelectedIndices();
                T_star = double(selector.getTStop());
                L_mult = double(selector.getDummyMultiplierL());
                break;
            }
        }

        fdp_vec[mc] = rates::compute_fdp(sel, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel, dat.true_support);
        L_vec[mc]   = L_mult;
        T_vec[mc]   = T_star;
    }

    // Means
    const double dMC = static_cast<double>(num_MC);
    SDTrexGridPointResult r;
    for (int mc = 0; mc < iMC; ++mc) {
        r.avg_fdr += fdp_vec[mc];
        r.avg_tpr += tpp_vec[mc];
        r.avg_L   += L_vec[mc];
        r.avg_T   += T_vec[mc];
    }
    r.avg_fdr /= dMC;  r.avg_tpr /= dMC;  r.avg_L /= dMC;  r.avg_T /= dMC;

    // Bessel-corrected standard deviations
    if (num_MC > 1) {
        double ssq_fdr = 0.0, ssq_tpr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            ssq_fdr += (fdp_vec[mc] - r.avg_fdr) * (fdp_vec[mc] - r.avg_fdr);
            ssq_tpr += (tpp_vec[mc] - r.avg_tpr) * (tpp_vec[mc] - r.avg_tpr);
        }
        r.sd_fdr = std::sqrt(ssq_fdr / (dMC - 1.0));
        r.sd_tpr = std::sqrt(ssq_tpr / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " — done. TPP=" << std::fixed << std::setprecision(3) << r.avg_tpr
              << "  FDR=" << r.avg_fdr << "\n\n" << std::flush;
    return r;
}

// ==============================================================================
// Results table (aligned console output, classic demo format)
// ==============================================================================

static void print_results(
    std::size_t num_MC,
    Eigen::Index n, Eigen::Index p,
    const std::vector<double>&                    snr_values,
    const std::vector<DemoConfigInfo>&            configs_to_test,
    const std::map<std::string, Eigen::VectorXd>& fdr_results_map,
    const std::map<std::string, Eigen::VectorXd>& tpr_results_map,
    const std::map<std::string, Eigen::VectorXd>& avg_L_results_map,
    const std::map<std::string, Eigen::VectorXd>& avg_T_results_map)
{
    // 1. File output (classic demo convention): aligned .txt + tidy .csv
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);
    std::ostringstream stem_ss;
    stem_ss << "demo_trex_sd_mc_variable_support_n" << n << "_p" << p
            << "_MC" << num_MC;
    const std::string file_stem = stem_ss.str();

    std::ofstream out_file(folder + file_stem + ".txt");
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    {
        std::ostringstream ss;
        ss << "\n"
           << "======================================================================\n"
           << "=== TRexSD vs TRexVD Results (averaged over " << num_MC
           << " Monte Carlo runs, n=" << n << ", p=" << p << ") ===\n"
           << "======================================================================\n\n";
        print_dual(ss.str());
    }

    std::size_t solver_width = 17;
    for (const auto& cfg : configs_to_test)
        solver_width = std::max(solver_width, cfg.name.size() + 2);
    const std::size_t metric_width  = 8;
    const std::size_t snr_col_width = 5;
    const std::size_t col_width     = 10;

    {
        std::ostringstream hdr;
        hdr << std::left  << std::setw(solver_width)  << "Solver"
            << std::left  << std::setw(metric_width)  << "Metric"
            << std::right << std::setw(snr_col_width) << "SNR";
        for (double snr : snr_values)
            hdr << std::fixed << std::setprecision(1) << std::setw(col_width) << snr;
        hdr << "\n"
            << std::string(solver_width + metric_width + snr_col_width +
                           col_width * snr_values.size(), '-') << "\n";
        print_dual(hdr.str());
    }

    auto print_metric_row = [&](const std::string& name, const std::string& metric,
                                const Eigen::VectorXd& data, bool first_row) {
        std::ostringstream row;
        row << std::left  << std::setw(solver_width) << (first_row ? name : "")
            << std::left  << std::setw(metric_width) << metric
            << std::setw(snr_col_width) << "" << std::right;
        for (Eigen::Index i = 0; i < data.size(); ++i)
            row << std::fixed << std::setprecision(4)
                << std::setw(col_width) << data(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& cfg : configs_to_test) {
        print_metric_row(cfg.name, "FDR",   fdr_results_map.at(cfg.name),   true);
        print_metric_row(cfg.name, "TPR",   tpr_results_map.at(cfg.name),   false);
        print_metric_row(cfg.name, "Avg L", avg_L_results_map.at(cfg.name), false);
        print_metric_row(cfg.name, "Avg T", avg_T_results_map.at(cfg.name), false);
        print_dual("\n");
    }

    if (out_file.is_open()) {
        std::cout << "[Info] Results saved to:     " << folder + file_stem + ".txt\n";
        out_file.close();
    } else {
        std::cout << "[Warning] Could not open output file: "
                  << folder + file_stem + ".txt\n";
    }

    // 2. Tidy long-format CSV (solver,metric,snr,value) for plotting
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "solver,metric,snr,value\n" << std::fixed << std::setprecision(6);
        for (const auto& cfg : configs_to_test) {
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                csv << cfg.name << ",FDR,"  << snr_values[i] << ","
                    << fdr_results_map.at(cfg.name)(ei) << "\n";
                csv << cfg.name << ",TPR,"  << snr_values[i] << ","
                    << tpr_results_map.at(cfg.name)(ei) << "\n";
                csv << cfg.name << ",AvgL," << snr_values[i] << ","
                    << avg_L_results_map.at(cfg.name)(ei) << "\n";
                csv << cfg.name << ",AvgT," << snr_values[i] << ","
                    << avg_T_results_map.at(cfg.name)(ei) << "\n";
            }
        }
        std::cout << "[Info] CSV results saved to: " << folder + file_stem + ".csv\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: "
                  << folder + file_stem + ".csv\n";
    }
}

// ===============================================================================
// Monte Carlo Simulation — Variable Active Support
// ===============================================================================

void demo_TRexSD_varMonteCarlo(std::size_t num_MC, bool high_dim, bool rnd_coef) {

    std::cout.setf(std::ios::unitbuf); // Flush output after each std::endl

    cdiagnostics::print_section_header("Demo: TRexSD vs TRexVD Monte Carlo Simulation");
    std::cout << (high_dim ? "High-dimensional (p > n)" : "Low-dimensional (n > p)") << "\n";

    const Eigen::Index n = high_dim ? 300 : 1000;
    const Eigen::Index p = high_dim ? 1000 : 300;
    std::cout << "n = " << n << ", p = " << p << "\n";
    const std::size_t cardinality_true_support = 10;

    // SNR values: 0.1, 0.2, ..., 2.0, 5.0
    std::vector<double> snr_values{0.1, 0.2, 0.5, 0.6, 1, 2, 5}; // (20);
    //std::iota(snr_values.begin(), snr_values.end(), 1);
    //for (auto& x : snr_values) x *= 0.1;
    //snr_values.push_back(5.0);

    // Target FDR level
    const double tFDR = 0.1;

    const std::vector<DemoConfigInfo> configs_to_test = make_default_configs_to_test();

    // Results: config x SNR
    std::map<std::string, Eigen::VectorXd> fdr_results_map;
    std::map<std::string, Eigen::VectorXd> tpr_results_map;
    std::map<std::string, Eigen::VectorXd> average_L_results_map;
    std::map<std::string, Eigen::VectorXd> average_T_results_map;

    for (const auto& cfg : configs_to_test) {
        fdr_results_map[cfg.name]       = Eigen::VectorXd(snr_values.size());
        tpr_results_map[cfg.name]       = Eigen::VectorXd(snr_values.size());
        average_L_results_map[cfg.name] = Eigen::VectorXd(snr_values.size());
        average_T_results_map[cfg.name] = Eigen::VectorXd(snr_values.size());
    }

    const tsd::TRexSDOptions base_opts = make_base_sd_options();

    for (const auto& current_config : configs_to_test) {

        cdiagnostics::print_section_header("Config: " + current_config.name);

        for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {

            const double snr = snr_values[snr_idx];
            std::cout << "SNR = " << std::fixed << std::setprecision(1) << snr << "\n";

            const auto make_data = [=](unsigned seed) -> SDTrexDGPData {
                // Variable support: fresh uniform draw per trial
                std::mt19937 rng_sup(seed + 500000u);
                std::vector<std::size_t> pool(static_cast<std::size_t>(p));
                std::iota(pool.begin(), pool.end(), std::size_t{0});
                std::shuffle(pool.begin(), pool.end(), rng_sup);
                pool.resize(cardinality_true_support);
                std::sort(pool.begin(), pool.end());
                std::vector<std::size_t> sup = pool;

                std::vector<double> coefs;
                coefs.reserve(cardinality_true_support);
                if (rnd_coef) {
                    std::mt19937 rng_coef(seed + 600000u);
                    std::normal_distribution<double> nd(0.0, 1.0);
                    for (std::size_t i = 0; i < cardinality_true_support; ++i)
                        coefs.push_back(nd(rng_coef));
                } else {
                    for (std::size_t i = 0; i < cardinality_true_support; ++i)
                        coefs.push_back(1.0);
                }

                datagen::SyntheticData data(
                    n, p,
                    sup, coefs, snr,
                    static_cast<int>(seed));

                return {data.getX(), data.getY(), sup};
            };

            std::ostringstream lbl;
            lbl << "SNR=" << std::fixed << std::setprecision(2) << snr
                << " [" << current_config.name << "]";
            const unsigned base_seed = 24u + static_cast<unsigned>(snr_idx) * 1000u;
            const auto result = run_mc_trials(
                num_MC,
                lbl.str(),
                make_data,
                tFDR,
                base_opts,
                current_config,
                base_seed);

            const auto ei = static_cast<Eigen::Index>(snr_idx);
            fdr_results_map[current_config.name](ei)       = result.avg_fdr;
            tpr_results_map[current_config.name](ei)       = result.avg_tpr;
            average_L_results_map[current_config.name](ei) = result.avg_L;
            average_T_results_map[current_config.name](ei) = result.avg_T;
        }
        std::cout << "\n";
    }

    print_results(
        num_MC,
        n,
        p,
        snr_values,
        configs_to_test,
        fdr_results_map,
        tpr_results_map,
        average_L_results_map,
        average_T_results_map);

    std::cout << "\n\n";
}


// ==============================================================================
// Main
// ==============================================================================

int main(int argc, char** argv) {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(6);
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    const std::size_t num_MC = (argc > 1)
        ? static_cast<std::size_t>(std::stoul(argv[1])) : 200;

    // ============================================================
    // Monte Carlo simulation: variable active support
    // ============================================================
    if (true)
        demo_TRexSD_varMonteCarlo(num_MC, /*high_dim=*/true, /*rnd_coef=*/false);

    if (false)
        demo_TRexSD_varMonteCarlo(num_MC, /*high_dim=*/false, /*rnd_coef=*/false);

    return 0;
}
