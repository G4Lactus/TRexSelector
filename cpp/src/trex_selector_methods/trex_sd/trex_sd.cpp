// ===================================================================================
// trex_sd.cpp
// ===================================================================================

#include <trex_selector_methods/trex_sd/trex_sd.hpp>

// std includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

// ===================================================================================

namespace trex::trex_selector_methods::trex_sd {

namespace sd_solvers  = trex::tsolvers::linear_model::lars_based;
namespace omp_solvers = trex::tsolvers::linear_model::omp_based;
namespace afs_solvers = trex::tsolvers::linear_model::afs_based;

// ========================================================================
// Constructor
// ========================================================================
TRexSD::TRexSD(const TRexSDOptions& opts) : opt_(opts) {}

// ========================================================================
// Seeds (seed 0 means "random_device" to the SD solvers, so shift by +1
// to keep every experiment deterministic and reproducible)
// ========================================================================
std::vector<unsigned long long> TRexSD::make_seeds_(int K) const {
    std::mt19937_64 rng(opt_.seed);
    std::uniform_int_distribution<uint32_t> dist(0u, 2147483647u);
    std::vector<unsigned long long> seeds(K);
    for (int k = 0; k < K; ++k) seeds[k] = 1ULL + dist(rng);
    return seeds;
}

// ========================================================================
// Solver factory
// ========================================================================
std::unique_ptr<tsolvers::SDTSolver_Base> TRexSD::make_solver_(
    Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
    int num_dummies, unsigned long long seed, int T_stop) const
{
    const auto L = (std::size_t)num_dummies;
    const auto T = (std::size_t)T_stop;

    if (opt_.solver == SDSolverType::General) {
        switch (opt_.algo) {
            case SDAlgo::OMP:
                return std::make_unique<omp_solvers::SD_TOMP_Solver>(
                    Xm, ym, rho_d_resolved_, L, T, /*intercept=*/true, seed);
            case SDAlgo::AFS:
                return std::make_unique<afs_solvers::SD_TAFS_Solver>(
                    Xm, ym, rho_d_resolved_, L, T, /*intercept=*/true, seed,
                    opt_.rho);
            case SDAlgo::LARS:
            default:
                return std::make_unique<sd_solvers::SD_TLARS_Solver>(
                    Xm, ym, rho_d_resolved_, L, T, /*intercept=*/true, seed);
        }
    }

    const auto policy = (opt_.solver == SDSolverType::Pair)
        ? tsolvers::SD2GenPolicy::OnDemand
        : tsolvers::SD2GenPolicy::Geometric;
    switch (opt_.algo) {
        case SDAlgo::OMP:
            return std::make_unique<omp_solvers::SD2_TOMP_Solver>(
                Xm, ym, L, T, /*intercept=*/true, seed, policy);
        case SDAlgo::AFS:
            return std::make_unique<afs_solvers::SD2_TAFS_Solver>(
                Xm, ym, L, T, /*intercept=*/true, seed, policy, opt_.rho);
        case SDAlgo::LARS:
        default:
            return std::make_unique<sd_solvers::SD2_TLARS_Solver>(
                Xm, ym, L, T, /*intercept=*/true, seed, policy);
    }
}

// ========================================================================
// Voting grid (verbatim parity with trex_vd)
// ========================================================================
Eigen::VectorXd TRexSD::make_V_(int K, double eps) const {
    std::vector<double> v;
    v.reserve(K + 1);
    for (int i = 0; i < K; ++i) {
        double val = 0.5 + double(i) / double(K);
        if (val < 1.0) v.push_back(val);
    }
    v.push_back(1.0 - eps);
    Eigen::VectorXd V(v.size());
    for (int i = 0; i < (int)v.size(); ++i) V(i) = v[i];
    return V;
}

// ========================================================================
// Phi_prime (verbatim parity with trex_vd, incl. the [0, 1] clamp)
// ========================================================================
Eigen::VectorXd TRexSD::Phi_prime_fun_(
    int p, int T_stop, int num_dummies,
    const Eigen::MatrixXd& phi_T_mat, const Eigen::VectorXd& Phi) const
{
    Eigen::VectorXd av = phi_T_mat.colwise().sum();
    Eigen::VectorXd delta_av = Eigen::VectorXd::Zero(T_stop);
    for (int j = 0; j < p; ++j)
        if (Phi(j) > 0.5)
            delta_av.noalias() += phi_T_mat.row(j).transpose();

    Eigen::VectorXd delta_mod = delta_av;
    Eigen::MatrixXd phi_mod   = phi_T_mat;
    if (T_stop > 1) {
        delta_mod.segment(1, T_stop - 1) =
            delta_av.segment(1, T_stop - 1) - delta_av.segment(0, T_stop - 1);
        phi_mod.block(0, 1, p, T_stop - 1) =
            phi_T_mat.block(0, 1, p, T_stop - 1) - phi_T_mat.block(0, 0, p, T_stop - 1);
    }

    Eigen::VectorXd phi_scale = Eigen::VectorXd::Zero(T_stop);
    for (int t = 0; t < T_stop; ++t) {
        double denom = double(num_dummies) - (t + 1) + 1.0;
        if (delta_mod(t) > opt_.eps && denom > 0.0) {
            double numer = double(p) - av(t);
            phi_scale(t) = 1.0 - (numer / denom) / delta_mod(t);
        }
    }
    return (phi_mod * phi_scale).cwiseMax(0.0).cwiseMin(1.0);
}

// ========================================================================
// FDP hat (verbatim parity with trex_vd)
// ========================================================================
Eigen::VectorXd TRexSD::fdp_hat_(const Eigen::VectorXd& V, const Eigen::VectorXd& Phi,
                                 const Eigen::VectorXd& Phi_prime) const {
    const int n_v = (int)V.size(), p = (int)Phi.size();
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n_v);
    for (int i = 0; i < n_v; ++i) {
        double v = V(i);
        int R = 0; double num = 0.0;
        for (int j = 0; j < p; ++j)
            if (Phi(j) > v) { ++R; num += (1.0 - Phi_prime(j)); }
        out(i) = (R == 0) ? 0.0 : std::min(1.0, num / double(R));
    }
    return out;
}

// ========================================================================
// Variable selection: search (T, v) grid (verbatim parity with trex_vd)
// ========================================================================
TRexSD::SelectResult TRexSD::select_var_(
    int p, double tFDR, int T_stop,
    const Eigen::MatrixXd& FDP_hat_mat,
    const Eigen::MatrixXd& Phi_mat,
    const Eigen::VectorXd& V) const
{
    const int n_v = (int)V.size();

    std::vector<int> T_cands;
    T_cands.reserve(T_stop);
    for (int t = 0; t < T_stop; ++t) {
        for (int j = 0; j < n_v; ++j)
            if (FDP_hat_mat(t, j) <= tFDR) { T_cands.push_back(t); break; }
    }

    SelectResult result;
    if (T_cands.empty()) {
        result.v_thresh = V(n_v - 1);
        result.selected_var = Eigen::VectorXi(0);
        return result;
    }

    int T_select = T_cands.back() + 1;
    int best_R = -1, best_t = -1, best_j = -1;

    for (int t = 0; t < T_select; ++t) {
        for (int j = 0; j < n_v; ++j) {
            if (FDP_hat_mat(t, j) > tFDR) continue;
            int cnt = 0;
            for (int i = 0; i < p; ++i)
                if (Phi_mat(t, i) > V(j)) ++cnt;
            if (cnt > best_R ||
                (cnt == best_R && (j > best_j || (j == best_j && t > best_t)))) {
                best_R = cnt; best_t = t; best_j = j;
            }
        }
    }

    result.v_thresh = V(best_j);
    std::vector<int> sel;
    for (int i = 0; i < p; ++i)
        if (Phi_mat(best_t, i) > result.v_thresh) sel.push_back(i);
    result.selected_var.resize((int)sel.size());
    for (int k = 0; k < (int)sel.size(); ++k) result.selected_var(k) = sel[k];
    return result;
}

// ========================================================================
// Mask helper: which original variables are active in a solver
// ========================================================================
namespace {
inline void fill_mask_(const tsolvers::SDTSolver_Base& solver, int p,
                       Eigen::VectorXd& mask) {
    for (std::size_t j : solver.getSelectedOriginals())
        if (j < (std::size_t)p) mask((Eigen::Index)j) = 1.0;
}
}  // namespace

// ========================================================================
// Fixed-T mode: run K experiments to exactly T_fixed, sweep v only
// ========================================================================
TRexSDResult TRexSD::run_fixed_T_(
    Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
    int T_fixed, int num_dummies, int n_threads, const Eigen::VectorXd& V)
{
    const int p = (int)Xm.cols();
    const int n_v = (int)V.size();

    auto seeds = make_seeds_(opt_.K);

    std::vector<std::unique_ptr<tsolvers::SDTSolver_Base>> solvers(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        solvers[k] = make_solver_(Xm, ym, num_dummies, seeds[k], T_fixed);

    Eigen::VectorXd Phi = Eigen::VectorXd::Zero(p);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    #endif
    for (int k = 0; k < opt_.K; ++k) {
        solvers[k]->executeStep((std::size_t)T_fixed, true);
        Eigen::VectorXd mask = Eigen::VectorXd::Zero(p);
        fill_mask_(*solvers[k], p, mask);
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        Phi += mask;
    }

    solvers.clear();
    Phi.array() /= double(opt_.K);

    Eigen::MatrixXd phi_T_mat(p, 1);
    phi_T_mat.col(0) = Phi;

    Eigen::VectorXd pp = Phi_prime_fun_(p, 1, num_dummies, phi_T_mat, Phi);
    Eigen::VectorXd fh = fdp_hat_(V, Phi, pp);

    double best_v = V(n_v - 1);
    int best_R = 0;
    int best_j = -1;

    for (int j = 0; j < n_v; ++j) {
        if (fh(j) > opt_.tFDR) continue;
        int cnt = 0;
        for (int i = 0; i < p; ++i)
            if (Phi(i) > V(j)) ++cnt;
        if (cnt > best_R || (cnt == best_R && j > best_j)) {
            best_R = cnt; best_j = j; best_v = V(j);
        }
    }

    std::vector<int> sel;
    if (best_j >= 0) {
        for (int i = 0; i < p; ++i)
            if (Phi(i) > best_v) sel.push_back(i);
    }

    TRexSDResult out;
    out.selected_var.resize((int)sel.size());
    for (int k = 0; k < (int)sel.size(); ++k) out.selected_var(k) = sel[k];
    out.v_thresh    = best_v;
    out.T_stop      = T_fixed;
    out.num_dummies = num_dummies;
    out.V           = V;
    out.FDP_hat_mat.resize(1, n_v);
    out.FDP_hat_mat.row(0) = fh.transpose();
    out.Phi_mat.resize(1, p);
    out.Phi_mat.row(0) = Phi.transpose();
    out.Phi_prime   = std::move(pp);
    out.K           = opt_.K;
    return out;
}

// ========================================================================
// Posthoc mode: all K solvers run the full T grid independently
// ========================================================================
TRexSDResult TRexSD::run_posthoc_(
    Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
    int Tstop, int num_dummies, int n_threads, const Eigen::VectorXd& V)
{
    const int p = (int)Xm.cols();
    const int n_v = (int)V.size();

    auto seeds = make_seeds_(opt_.K);

    std::vector<std::unique_ptr<tsolvers::SDTSolver_Base>> solvers;
    solvers.reserve(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        solvers.push_back(make_solver_(Xm, ym, num_dummies, seeds[k], Tstop));

    std::vector<Eigen::MatrixXd> per_solver_phi(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        per_solver_phi[k] = Eigen::MatrixXd::Zero(p, Tstop);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    #endif
    for (int k = 0; k < opt_.K; ++k) {
        for (int t = 1; t <= Tstop; ++t) {
            solvers[k]->executeStep((std::size_t)t, true);
            Eigen::VectorXd mask = Eigen::VectorXd::Zero(p);
            fill_mask_(*solvers[k], p, mask);
            per_solver_phi[k].col(t - 1) = mask;
        }
    }

    solvers.clear();

    Eigen::MatrixXd phi_T_mat = Eigen::MatrixXd::Zero(p, Tstop);
    for (int k = 0; k < opt_.K; ++k) phi_T_mat += per_solver_phi[k];
    phi_T_mat.array() /= double(opt_.K);
    per_solver_phi.clear();

    Eigen::MatrixXd FDP_hat_mat(Tstop, n_v);
    Eigen::MatrixXd Phi_mat(Tstop, p);
    Eigen::VectorXd Phi_prime;

    for (int t = 1; t <= Tstop; ++t) {
        Eigen::VectorXd Phi_t = phi_T_mat.col(t - 1);
        Phi_mat.row(t - 1) = Phi_t.transpose();
        Eigen::VectorXd pp = Phi_prime_fun_(p, t, num_dummies, phi_T_mat.leftCols(t), Phi_t);
        Eigen::VectorXd fh = fdp_hat_(V, Phi_t, pp);
        FDP_hat_mat.row(t - 1) = fh.transpose();
        if (t == Tstop) Phi_prime = std::move(pp);
    }

    SelectResult sel = select_var_(p, opt_.tFDR, Tstop, FDP_hat_mat, Phi_mat, V);

    TRexSDResult out;
    out.selected_var = sel.selected_var;
    out.v_thresh     = sel.v_thresh;
    out.T_stop       = Tstop;
    out.num_dummies  = num_dummies;
    out.V            = V;
    out.FDP_hat_mat  = std::move(FDP_hat_mat);
    out.Phi_mat      = std::move(Phi_mat);
    out.Phi_prime    = std::move(Phi_prime);
    out.K            = opt_.K;
    return out;
}

// ========================================================================
// Early-stop mode: strided barriers (verbatim structure from trex_vd)
// ========================================================================
TRexSDResult TRexSD::run_early_stop_(
    Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
    int Tstop, int num_dummies, int n_threads, const Eigen::VectorXd& V)
{
    const int p = (int)Xm.cols();
    const int n_v = (int)V.size();
    const int SW = std::max(1, opt_.stride_width);

    // Dummy-burn guard, tri-state (classic parity): auto = disabled for the
    // LARS path (terminates on its own), enabled for the greedy solvers.
    const bool stagnation_guard =
        opt_.stagnation_stop.value_or(opt_.algo != SDAlgo::LARS);

    auto seeds = make_seeds_(opt_.K);

    std::vector<std::unique_ptr<tsolvers::SDTSolver_Base>> solvers;
    solvers.reserve(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        solvers.push_back(make_solver_(Xm, ym, num_dummies, seeds[k], Tstop));

    Eigen::MatrixXd phi_T_mat = Eigen::MatrixXd::Zero(p, Tstop);
    int T_stop = 0;
    int stale_count = 0;
    int prev_n_above = 0;

    while (T_stop < Tstop) {
        const int T_next = std::min(T_stop + SW, Tstop);
        const int steps  = T_next - T_stop;

        std::vector<Eigen::MatrixXd> local_masks(opt_.K);
        for (int k = 0; k < opt_.K; ++k)
            local_masks[k] = Eigen::MatrixXd::Zero(p, steps);

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        #endif
        for (int k = 0; k < opt_.K; ++k) {
            for (int s = 0; s < steps; ++s) {
                solvers[k]->executeStep((std::size_t)(T_stop + s + 1), true);
                Eigen::VectorXd mask = Eigen::VectorXd::Zero(p);
                fill_mask_(*solvers[k], p, mask);
                local_masks[k].col(s) = mask;
            }
        }

        for (int s = 0; s < steps; ++s) {
            Eigen::VectorXd Phi_t = Eigen::VectorXd::Zero(p);
            for (int k = 0; k < opt_.K; ++k) Phi_t += local_masks[k].col(s);
            Phi_t.array() /= double(opt_.K);
            phi_T_mat.col(T_stop + s) = Phi_t;
        }

        T_stop = T_next;

        Eigen::VectorXd Phi_last = phi_T_mat.col(T_stop - 1);
        Eigen::VectorXd pp = Phi_prime_fun_(p, T_stop, num_dummies,
                                            phi_T_mat.leftCols(T_stop), Phi_last);
        Eigen::VectorXd fh = fdp_hat_(V, Phi_last, pp);

        int n_above_half = 0;
        for (int j = 0; j < p; ++j)
            if (Phi_last(j) > 0.5) ++n_above_half;

        if (opt_.verbose)
            std::cout << "[TRexSD] T=" << T_stop
                      << " FDP_hat(max_v)=" << fh(n_v - 1)
                      << " n_real(Phi>0.5)=" << n_above_half << "\n";

        // Primary stop: FDP exceeds target
        if (fh(n_v - 1) > opt_.tFDR) break;

        // Dummy-burn stop (greedy solvers only by default; see the tri-state
        // stagnation_stop option)
        if (stagnation_guard) {
            if (n_above_half <= prev_n_above) {
                ++stale_count;
                if (stale_count >= opt_.max_stale_strides) {
                    if (opt_.verbose)
                        std::cout << "[TRexSD] stopping: reals above 0.5 stuck at "
                                  << n_above_half << " for "
                                  << stale_count << " consecutive strides\n";
                    break;
                }
            } else {
                stale_count = 0;
            }
            prev_n_above = n_above_half;
        }
    }

    solvers.clear();

    const int T_rows = T_stop;
    Eigen::MatrixXd FDP_hat_mat(T_rows, n_v);
    Eigen::MatrixXd Phi_mat(T_rows, p);
    Eigen::VectorXd Phi_prime;

    for (int t = 1; t <= T_rows; ++t) {
        Eigen::VectorXd Phi_t = phi_T_mat.col(t - 1);
        Phi_mat.row(t - 1) = Phi_t.transpose();
        Eigen::VectorXd pp = Phi_prime_fun_(p, t, num_dummies, phi_T_mat.leftCols(t), Phi_t);
        Eigen::VectorXd fh = fdp_hat_(V, Phi_t, pp);
        FDP_hat_mat.row(t - 1) = fh.transpose();
        if (t == T_rows) Phi_prime = std::move(pp);
    }

    SelectResult sel = select_var_(p, opt_.tFDR, T_rows, FDP_hat_mat, Phi_mat, V);

    TRexSDResult out;
    out.selected_var = sel.selected_var;
    out.v_thresh     = sel.v_thresh;
    out.T_stop       = T_rows;
    out.num_dummies  = num_dummies;
    out.V            = V;
    out.FDP_hat_mat  = std::move(FDP_hat_mat);
    out.Phi_mat      = std::move(Phi_mat);
    out.Phi_prime    = std::move(Phi_prime);
    out.K            = opt_.K;
    return out;
}

// ========================================================================
// L calibration: the T-Rex L-loop — scan L = p, 2p, 3p, ... at T=1,
// v=0.75, accept the first L with FDP_hat <= tFDR (verbatim parity with
// trex_vd::calibrate_L_)
// ========================================================================
int TRexSD::calibrate_L_(
    Eigen::Map<Eigen::MatrixXd>& Xm, Eigen::Map<Eigen::VectorXd>& ym,
    int p, int n_threads)
{
    const double v_calib = 0.75;

    for (int L_mult = 1; L_mult <= opt_.max_L_factor; ++L_mult) {
        const int L = L_mult * p;

        if (opt_.verbose)
            std::cout << "[TRexSD] calibrate_L: trying L=" << L
                      << " (" << L_mult << "p)\n";

        auto seeds = make_seeds_(opt_.K);

        // Run K experiments at T=1
        std::vector<std::unique_ptr<tsolvers::SDTSolver_Base>> solvers(opt_.K);
        for (int k = 0; k < opt_.K; ++k)
            solvers[k] = make_solver_(Xm, ym, L, seeds[k], /*T_stop=*/1);

        Eigen::MatrixXd phi_sum = Eigen::MatrixXd::Zero(p, 1);

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        #endif
        for (int k = 0; k < opt_.K; ++k) {
            solvers[k]->executeStep(1, true);
            Eigen::VectorXd mask = Eigen::VectorXd::Zero(p);
            fill_mask_(*solvers[k], p, mask);
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            phi_sum.col(0) += mask;
        }

        solvers.clear();
        phi_sum.array() /= double(opt_.K);
        Eigen::VectorXd Phi = phi_sum.col(0);

        Eigen::VectorXd pp = Phi_prime_fun_(p, 1, L, phi_sum, Phi);

        int R = 0; double num = 0.0;
        for (int j = 0; j < p; ++j)
            if (Phi(j) > v_calib) { ++R; num += (1.0 - pp(j)); }
        double fdp = (R == 0) ? 0.0 : std::min(1.0, num / double(R));

        if (opt_.verbose)
            std::cout << "[TRexSD] calibrate_L: L=" << L
                      << " FDP_hat(v=0.75)=" << fdp << "\n";

        if (fdp <= opt_.tFDR) {
            if (opt_.verbose)
                std::cout << "[TRexSD] calibrate_L: accepted L=" << L << "\n";
            return L;
        }
    }

    int max_L = opt_.max_L_factor * p;
    if (opt_.verbose)
        std::cout << "[TRexSD] calibrate_L: ceiling, using L=" << max_L << "\n";
    return max_L;
}

// ========================================================================
// Main entry point
// ========================================================================
TRexSDResult TRexSD::run(Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    const int n = (int)X.rows(), p = (int)X.cols();

    int n_threads;
    if (opt_.n_threads > 0)
        n_threads = opt_.n_threads;
    else
        n_threads = std::min(opt_.K, omp_get_max_threads());

    Eigen::VectorXd V = make_V_(opt_.K, opt_.eps);

    // ---- Dummy sparsity: auto-calibrate once, reuse across all K.
    //      (Uses the L_factor budget as the reference L; if the L-loop
    //      later raises L, the k choice only becomes more conservative.)
    rho_d_resolved_ = opt_.rho_d;
    k_resolved_ = 0;
    if (opt_.solver == SDSolverType::General && opt_.rho_d == 0.0) {
        tsolvers::sd_calibration::Options copt;
        copt.L = (std::size_t)(opt_.L_factor * p);
        auto calib = tsolvers::sd_calibration::calibrate(X, y, copt);
        rho_d_resolved_ = calib.rho_d;
        k_resolved_ = calib.k;
        if (opt_.verbose)
            std::cout << "[TRexSD] auto-calibration: k=" << calib.k
                      << " rho_d=" << calib.rho_d
                      << (calib.feasible ? "" : " [best effort]") << "\n";
    }

    // ---- Maps shared by all K solvers (X read-only, y copied per solver) ----
    Eigen::VectorXd y_local = y;
    Eigen::Map<Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> ym(y_local.data(), y_local.size());

    // ---- Budget: fixed L or the T-Rex L-loop ----
    int num_dummies;
    switch (opt_.calib) {
        case CalibMode::FixedTL:
        case CalibMode::CalibrateT:
            num_dummies = opt_.L_factor * p;
            break;
        case CalibMode::CalibrateL:
        case CalibMode::CalibrateBoth:
        default:
            num_dummies = calibrate_L_(Xm, ym, p, n_threads);
            break;
    }

    // ---- T ceiling ----
    int Tstop;
    if (opt_.T_stop > 0) {
        Tstop = opt_.T_stop;
    } else {
        Tstop = std::min(num_dummies, (int)std::ceil(n / 2.0));
    }

    if (opt_.verbose) {
        const char* solver_names[] = {"General", "Pair", "PairGeometric"};
        const char* algo_names[]   = {"LARS", "OMP", "AFS"};
        const char* calib_names[]  = {"FixedTL", "CalibrateT", "CalibrateL",
                                      "CalibrateBoth"};
        std::cout << "[TRexSD] p=" << p << " n=" << n
                  << " K=" << opt_.K
                  << " L=" << num_dummies
                  << " T_ceiling=" << Tstop
                  << " solver=" << solver_names[(int)opt_.solver]
                  << " algo=" << algo_names[(int)opt_.algo];
        if (opt_.algo == SDAlgo::AFS) std::cout << "(rho=" << opt_.rho << ")";
        std::cout << " calib=" << calib_names[(int)opt_.calib]
                  << " threads=" << n_threads
                  << " mode=" << (opt_.posthoc_mode ? "posthoc" : "early-stop")
                  << "\n";
    }

    // ---- Dispatch ----
    TRexSDResult out;
    switch (opt_.calib) {
        case CalibMode::FixedTL:
        case CalibMode::CalibrateL:
            out = run_fixed_T_(Xm, ym, Tstop, num_dummies, n_threads, V);
            break;
        case CalibMode::CalibrateT:
        case CalibMode::CalibrateBoth:
        default:
            if (opt_.posthoc_mode)
                out = run_posthoc_(Xm, ym, Tstop, num_dummies, n_threads, V);
            else
                out = run_early_stop_(Xm, ym, Tstop, num_dummies, n_threads, V);
            break;
    }

    out.rho_d_used = rho_d_resolved_;
    out.k_used     = k_resolved_;

    if (opt_.verbose)
        std::cout << "[TRexSD] selected " << out.selected_var.size()
                  << " variables at v=" << out.v_thresh
                  << " T=" << out.T_stop
                  << " L=" << out.num_dummies << "\n";

    return out;
}

}  // namespace trex::trex_selector_methods::trex_sd
