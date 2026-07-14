// ===================================================================================
// trex_sd.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_SD_HPP
#define TREX_SELECTOR_METHODS_TREX_SD_HPP
// ===================================================================================
/**
 * @file trex_sd.hpp
 *
 * @brief Sparse-Dummy T-Rex Selector (TRexSD): K-experiment orchestrator over
 *        the SD solver family. Two orthogonal solver switches: the dummy law
 *        (General k / Pair / PairGeometric) and the selection algorithm
 *        (LARS / OMP / AFS), nine combinations in total.
 *
 * @details Runs K random experiments (independent sparse-Rademacher dummy
 *          streams), accumulates the relative occurrences Phi_j(T) of every
 *          original variable, and calibrates (v*, T*) against the target FDR
 *          with the T-Rex FDP estimator. The statistical machinery
 *          (voting grid, Phi_prime, FDP_hat, (T, v) search, early-stop /
 *          posthoc execution) is kept verbatim from TRexVD for parity — the
 *          two selectors differ only in the inner solver family.
 *
 *          Contract: X centered with equal-norm columns (unit-L2 or
 *          z-scored), y centered — the SD solver contract. X is not copied
 *          and not modified; the non-const reference is required by the SD
 *          solver map API.
 *
 *          rho_d = 0 auto-calibrates the dummy sparsity once (sd_calibration
 *          at the chosen budget L) and reuses it for all K experiments.
 */
// ===================================================================================

// std includes
#include <memory>
#include <optional>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// SD solver includes
#include <tsolvers/sd_tsolvers/sd_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tlars_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tomp_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_tafs_solver.hpp>
#include <tsolvers/sd_tsolvers/sd2_tafs_solver.hpp>
#include <tsolvers/sd_tsolvers/sd_calibration.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_sd {

// ---------- Enums ----------
// Dummy law of the inner solvers.
//
// WARNING: Pair and PairGeometric (the k = 1 law) are research/validation
// devices, NOT valid selectors at scale — at p = 10^4 both catastrophically
// violate FDR control (empirical FDR up to 0.73 / 0.54 at target 0.2): the
// pair null's top quantiles are rare atoms, so dummies are under-represented
// among the extremes and FDP-hat grossly underestimates. Use General (with
// auto-calibrated k) for anything beyond moderate p/n; see trex_sd.md §8.
enum class SDSolverType : uint8_t { General = 0, Pair = 1, PairGeometric = 2 };

// Selection algorithm of the inner solvers (mirrors trex_vd::SolverType):
// LARS equiangular path, OMP greedy full-refit, AFS blended forward stepwise
// (rho = 1 reduces AFS to OMP).
enum class SDAlgo : uint8_t { LARS = 0, OMP = 1, AFS = 2 };

// FixedTL:       user-specified T ceiling and L, calibrate v only.
// CalibrateT:    fix L = L_factor * p, search the (T, v) grid.
// CalibrateL:    L-loop (L = p, 2p, ... until FDP_hat(T=1, v=0.75) <= tFDR),
//                then fixed-T at that L.
// CalibrateBoth: L-loop first, then the (T, v) grid search.
enum class CalibMode : uint8_t { FixedTL = 0, CalibrateT = 1,
                                 CalibrateL = 2, CalibrateBoth = 3 };

// ---------- Options ----------
struct TRexSDOptions {
    double tFDR      = 0.2;
    int    K         = 20;     // number of random experiments
    int    L_factor  = 2;      // L = L_factor * p (sparse dummies are cheap)
    int    max_L_factor = 10;  // L-loop ceiling (classic max_dummy_multiplier)
    double rho_d     = 0.0;    // 0 = auto-calibrate once (General solver only)
    int    T_stop    = -1;     // -1 = auto: min(L, ceil(n/2))

    int    stride_width      = 1;
    bool   posthoc_mode      = false;

    // T-loop stagnation stop (dummy-burn guard). Tri-state mirroring the
    // classic TRexControlParameter::tloop_stagnation_stop: nullopt ("auto")
    // resolves by algorithm — DISABLED for LARS (the equiangular path
    // terminates properly on its own; matches the R reference and the T-Rex
    // paper), ENABLED for the greedy solvers (OMP/AFS otherwise run into a
    // noise trap and burn dummies up to the T ceiling).
    std::optional<bool> stagnation_stop = std::nullopt;
    int    max_stale_strides = 5;    // classic recommendation: {3, 5, 7}

    double eps               = 1e-12;
    bool   verbose           = true;
    unsigned long long seed  = 0ULL;

    SDSolverType solver = SDSolverType::General;
    SDAlgo       algo   = SDAlgo::LARS;
    double       rho    = 0.3;   // AFS blend fraction (ignored by LARS/OMP)
    // CalibrateBoth = the genuine T-Rex procedure (adaptive L-loop, then the
    // (T, v) grid) — classic-parity default; fixed-L modes are for studies.
    CalibMode    calib  = CalibMode::CalibrateBoth;

    int n_threads = 0;         // 0 = auto
};

// ---------- Result ----------
struct TRexSDResult {
    Eigen::VectorXi selected_var;
    double          v_thresh    = 0.0;
    int             T_stop      = 0;
    int             num_dummies = 0;
    double          rho_d_used  = 0.0;
    std::size_t     k_used      = 0;
    Eigen::VectorXd V;
    Eigen::MatrixXd FDP_hat_mat;   // (T_stop × |V|)
    Eigen::MatrixXd Phi_mat;       // (T_stop × p)
    Eigen::VectorXd Phi_prime;
    int             K = 0;
};

// ---------- Selector ----------
class TRexSD {
public:
    explicit TRexSD(const TRexSDOptions& opts);

    TRexSDResult run(Eigen::MatrixXd& X, const Eigen::VectorXd& y);

private:
    TRexSDOptions opt_;
    double        rho_d_resolved_{0.0};
    std::size_t   k_resolved_{0};

    // ---- Solver factory ----
    std::unique_ptr<tsolvers::SDTSolver_Base> make_solver_(
        Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
        int num_dummies, unsigned long long seed, int T_stop) const;

    // ---- Execution paths ----
    TRexSDResult run_fixed_T_(
        Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
        int T_fixed, int num_dummies, int n_threads, const Eigen::VectorXd& V);

    TRexSDResult run_posthoc_(
        Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
        int Tstop, int num_dummies, int n_threads, const Eigen::VectorXd& V);

    TRexSDResult run_early_stop_(
        Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
        int Tstop, int num_dummies, int n_threads, const Eigen::VectorXd& V);

    // ---- L calibration (the T-Rex L-loop) ----
    int calibrate_L_(
        Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
        int p, int n_threads);

    // ---- Statistical helpers (verbatim parity with trex_vd) ----
    Eigen::VectorXd make_V_(int K, double eps) const;

    Eigen::VectorXd Phi_prime_fun_(int p, int T_stop, int num_dummies,
                                   const Eigen::MatrixXd& phi_T_mat,
                                   const Eigen::VectorXd& Phi) const;

    Eigen::VectorXd fdp_hat_(const Eigen::VectorXd& V, const Eigen::VectorXd& Phi,
                             const Eigen::VectorXd& Phi_prime) const;

    struct SelectResult {
        Eigen::VectorXi selected_var;
        double v_thresh;
    };

    SelectResult select_var_(int p, double tFDR, int T_stop,
                             const Eigen::MatrixXd& FDP_hat_mat,
                             const Eigen::MatrixXd& Phi_mat,
                             const Eigen::VectorXd& V) const;

    // ---- Seeds ----
    std::vector<unsigned long long> make_seeds_(int K) const;
};

}  // namespace trex::trex_selector_methods::trex_sd

// ===================================================================================
#endif /* TREX_SELECTOR_METHODS_TREX_SD_HPP */
// ===================================================================================
