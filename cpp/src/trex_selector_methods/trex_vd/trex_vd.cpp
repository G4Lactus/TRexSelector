// ===================================================================================
// trex_vd.cpp
// ===================================================================================

#include <trex_selector_methods/trex_vd/trex_vd.hpp>

// std includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

// ===================================================================================

namespace trex::trex_selector_methods::trex_vd {

// ========================================================================
// Constructor
// ========================================================================
TRexVD::TRexVD(const TRexVDOptions& opts) : opt_(opts) {}

// ========================================================================
// Seeds
// ========================================================================
std::vector<unsigned long long> TRexVD::make_seeds_(int K) const {
    std::mt19937_64 rng(opt_.seed);
    std::uniform_int_distribution<uint32_t> dist(0u, 2147483647u);
    std::vector<unsigned long long> seeds(K);
    for (int k = 0; k < K; ++k) seeds[k] = dist(rng);
    return seeds;
}

// ========================================================================
// VDOptions factory
// ========================================================================
VDOptions TRexVD::make_vd_opts_(unsigned long long seed,
                                int T_stop, int n) const {
    VDOptions o;
    o.T_stop      = T_stop;
    o.max_vd_proj = std::min(n, (opt_.max_vd_proj > 0 ? opt_.max_vd_proj
                                                      : std::max(32, n)));
    o.eps         = opt_.eps;
    o.standardize = false;
    o.seed        = seed;
    o.dummy_law   = opt_.dummy_law;
    o.rho         = opt_.rho;
    return o;
}

// ========================================================================
// Solver factory
// ========================================================================
std::unique_ptr<vd::VD_Base> TRexVD::make_solver_(
    const Eigen::Ref<const MatC>& X,
    const Eigen::Ref<const Vec>& y,
    int num_dummies, unsigned long long seed, int T_stop) const
{
    VDOptions vdo = make_vd_opts_(seed, T_stop, (int)X.rows());
    switch (opt_.solver) {
        case SolverType::LARS:
            return std::make_unique<vd::VD_LARS>(X, y, num_dummies, vdo);
        case SolverType::OMP:
            return std::make_unique<vd::VD_OMP>(X, y, num_dummies, vdo);
        case SolverType::AFS:
            return std::make_unique<vd::VD_AFS>(X, y, num_dummies, vdo);
        case SolverType::AFS_Logistic:
            return std::make_unique<vd::VD_AFS_Logistic>(X, y, num_dummies, vdo);
    }
    return std::make_unique<vd::VD_LARS>(X, y, num_dummies, vdo);
}

// ========================================================================
// Voting grid
// ========================================================================
Vec TRexVD::make_V_(int K, double eps) const {
    std::vector<double> v;
    v.reserve(K + 1);
    for (int i = 0; i < K; ++i) {
        double val = 0.5 + double(i) / double(K);
        if (val < 1.0) v.push_back(val);
    }
    v.push_back(1.0 - eps);
    Vec V(v.size());
    for (int i = 0; i < (int)v.size(); ++i) V(i) = v[i];
    return V;
}

// ========================================================================
// Phi_prime
// ========================================================================
Vec TRexVD::Phi_prime_fun_(
    int p, int T_stop, int num_dummies,
    const MatC& phi_T_mat, const Vec& Phi) const
{
    Vec av = phi_T_mat.colwise().sum();
    Vec delta_av = Vec::Zero(T_stop);
    for (int j = 0; j < p; ++j)
        if (Phi(j) > 0.5)
            delta_av.noalias() += phi_T_mat.row(j).transpose();

    Vec  delta_mod = delta_av;
    MatC phi_mod   = phi_T_mat;
    if (T_stop > 1) {
        delta_mod.segment(1, T_stop - 1) =
            delta_av.segment(1, T_stop - 1) - delta_av.segment(0, T_stop - 1);
        phi_mod.block(0, 1, p, T_stop - 1) =
            phi_T_mat.block(0, 1, p, T_stop - 1) - phi_T_mat.block(0, 0, p, T_stop - 1);
    }

    Vec phi_scale = Vec::Zero(T_stop);
    for (int t = 0; t < T_stop; ++t) {
        double denom = double(num_dummies) - (t + 1) + 1.0;
        if (delta_mod(t) > opt_.eps && denom > 0.0) {
            double numer = double(p) - av(t);
            phi_scale(t) = 1.0 - (numer / denom) / delta_mod(t);
        }
    }
    // Clamp to [0, 1]: deliberate deviation from the virtual-dummies reference,
    // matching trex_core::TRexSelector::computePhiPrime (see the rationale
    // there). Parity tests against the vd_selectors Python package must
    // account for this wherever the unclipped Phi_prime leaves [0, 1].
    return (phi_mod * phi_scale).cwiseMax(0.0).cwiseMin(1.0);
}

// ========================================================================
// FDP hat
// ========================================================================
Vec TRexVD::fdp_hat_(const Vec& V, const Vec& Phi,
                     const Vec& Phi_prime) const {
    const int n_v = (int)V.size(), p = (int)Phi.size();
    // Zero-init instead of NaN: every entry is assigned below, and NaN is
    // unavailable under -ffast-math (Release build).
    Vec out = Vec::Zero(n_v);
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
// Variable selection: search (T, v) grid
// ========================================================================
TRexVD::SelectResult TRexVD::select_var_(
    int p, double tFDR, int T_stop,
    const Eigen::MatrixXd& FDP_hat_mat,
    const Eigen::MatrixXd& Phi_mat,
    const Vec& V) const
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
// Fixed-T mode: run K experiments to exactly T_fixed, sweep v only
// ========================================================================
TRexVDResult TRexVD::run_fixed_T_(
    const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
    int T_fixed, int num_dummies, int n_threads, const Vec& V)
{
    const int p = (int)X.cols();
    const int n_v = (int)V.size();

    auto seeds = make_seeds_(opt_.K);

    // Run K solvers to exactly T_fixed dummies
    std::vector<std::unique_ptr<vd::VD_Base>> solvers(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        solvers[k] = make_solver_(X, y, num_dummies, seeds[k], T_fixed);

    Vec Phi = Vec::Zero(p);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    #endif
    for (int k = 0; k < opt_.K; ++k) {
        solvers[k]->run(T_fixed);
        Vec mask = Vec::Zero(p);
        for (const auto& af : solvers[k]->active_features_copy()) {
            if (af.kind == vd::ActiveFeature::Kind::Real) {
                int j = af.index;
                if (j >= 0 && j < p) mask(j) = 1.0;
            }
        }
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        Phi += mask;
    }

    solvers.clear();
    Phi.array() /= double(opt_.K);

    // Need phi_T_mat (p × T_fixed) for Phi_prime_fun_.
    // We only have the final T, so construct a single-column version.
    MatC phi_T_mat(p, 1);
    phi_T_mat.col(0) = Phi;

    Vec pp = Phi_prime_fun_(p, 1, num_dummies, phi_T_mat, Phi);
    Vec fh = fdp_hat_(V, Phi, pp);

    // Sweep v only: find largest R with FDP_hat ≤ α
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

    TRexVDResult out;
    out.selected_var.resize((int)sel.size());
    for (int k = 0; k < (int)sel.size(); ++k) out.selected_var(k) = sel[k];
    out.v_thresh     = best_v;
    out.T_stop       = T_fixed;
    out.num_dummies  = num_dummies;
    out.L_calibrated = num_dummies;
    out.V            = V;
    // Single-row FDP_hat and Phi_mat
    out.FDP_hat_mat.resize(1, n_v);
    out.FDP_hat_mat.row(0) = fh.transpose();
    out.Phi_mat.resize(1, p);
    out.Phi_mat.row(0) = Phi.transpose();
    out.Phi_prime    = std::move(pp);
    out.K            = opt_.K;
    return out;
}

// ========================================================================
// Posthoc mode: one parallel region, all K solvers run T_stop independently
// ========================================================================
TRexVDResult TRexVD::run_posthoc_(
    const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
    int Tstop, int num_dummies, int n_threads, const Vec& V)
{
    const int p = (int)X.cols();
    const int n_v = (int)V.size();

    auto seeds = make_seeds_(opt_.K);

    std::vector<std::unique_ptr<vd::VD_Base>> solvers;
    solvers.reserve(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        solvers.push_back(make_solver_(X, y, num_dummies, seeds[k], Tstop));

    std::vector<MatC> per_solver_phi(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        per_solver_phi[k] = MatC::Zero(p, Tstop);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
    #endif
    for (int k = 0; k < opt_.K; ++k) {
        for (int t = 1; t <= Tstop; ++t) {
            solvers[k]->run(t);
            for (const auto& af : solvers[k]->active_features_copy()) {
                if (af.kind == vd::ActiveFeature::Kind::Real) {
                    int j = af.index;
                    if (j >= 0 && j < p) per_solver_phi[k](j, t - 1) = 1.0;
                }
            }
        }
    }

    solvers.clear();

    MatC phi_T_mat = MatC::Zero(p, Tstop);
    for (int k = 0; k < opt_.K; ++k) phi_T_mat += per_solver_phi[k];
    phi_T_mat.array() /= double(opt_.K);
    per_solver_phi.clear();

    Eigen::MatrixXd FDP_hat_mat(Tstop, n_v);
    Eigen::MatrixXd Phi_mat(Tstop, p);
    Vec Phi_prime;

    for (int t = 1; t <= Tstop; ++t) {
        Vec Phi_t = phi_T_mat.col(t - 1);
        Phi_mat.row(t - 1) = Phi_t.transpose();
        Vec pp = Phi_prime_fun_(p, t, num_dummies, phi_T_mat.leftCols(t), Phi_t);
        Vec fh = fdp_hat_(V, Phi_t, pp);
        FDP_hat_mat.row(t - 1) = fh.transpose();
        if (t == Tstop) Phi_prime = std::move(pp);
    }

    SelectResult sel = select_var_(p, opt_.tFDR, Tstop, FDP_hat_mat, Phi_mat, V);

    TRexVDResult out;
    out.selected_var  = sel.selected_var;
    out.v_thresh      = sel.v_thresh;
    out.T_stop        = Tstop;
    out.num_dummies   = num_dummies;
    out.L_calibrated  = num_dummies;
    out.V             = V;
    out.FDP_hat_mat   = std::move(FDP_hat_mat);
    out.Phi_mat       = std::move(Phi_mat);
    out.Phi_prime     = std::move(Phi_prime);
    out.K             = opt_.K;
    return out;
}

// ========================================================================
// Early-stop mode: strided barriers
// ========================================================================
TRexVDResult TRexVD::run_early_stop_(
    const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
    int Tstop, int num_dummies, int n_threads, const Vec& V)
{
    const int p = (int)X.cols();
    const int n_v = (int)V.size();
    const int SW = std::max(1, opt_.stride_width);

    auto seeds = make_seeds_(opt_.K);

    std::vector<std::unique_ptr<vd::VD_Base>> solvers;
    solvers.reserve(opt_.K);
    for (int k = 0; k < opt_.K; ++k)
        solvers.push_back(make_solver_(X, y, num_dummies, seeds[k], Tstop));

    MatC phi_T_mat = MatC::Zero(p, Tstop);
    int T_stop = 0;
    int stale_count = 0;
    int prev_n_above = 0;

    while (T_stop < Tstop) {
        const int T_next = std::min(T_stop + SW, Tstop);
        const int steps  = T_next - T_stop;

        std::vector<MatC> local_masks(opt_.K);
        for (int k = 0; k < opt_.K; ++k)
            local_masks[k] = MatC::Zero(p, steps);

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        #endif
        for (int k = 0; k < opt_.K; ++k) {
            for (int s = 0; s < steps; ++s) {
                solvers[k]->run(T_stop + s + 1);
                for (const auto& af : solvers[k]->active_features_copy()) {
                    if (af.kind == vd::ActiveFeature::Kind::Real) {
                        int j = af.index;
                        if (j >= 0 && j < p) local_masks[k](j, s) = 1.0;
                    }
                }
            }
        }

        for (int s = 0; s < steps; ++s) {
            Vec Phi_t = Vec::Zero(p);
            for (int k = 0; k < opt_.K; ++k) Phi_t += local_masks[k].col(s);
            Phi_t.array() /= double(opt_.K);
            phi_T_mat.col(T_stop + s) = Phi_t;
        }

        T_stop = T_next;

        Vec Phi_last = phi_T_mat.col(T_stop - 1);
        Vec pp = Phi_prime_fun_(p, T_stop, num_dummies,
                                phi_T_mat.leftCols(T_stop), Phi_last);
        Vec fh = fdp_hat_(V, Phi_last, pp);

        int n_above_half = 0;
        for (int j = 0; j < p; ++j)
            if (Phi_last(j) > 0.5) ++n_above_half;

        if (opt_.verbose)
            std::cout << "[TRexVD] T=" << T_stop
                      << " FDP_hat(max_v)=" << fh(n_v - 1)
                      << " n_real(Phi>0.5)=" << n_above_half << "\n";

        // Primary stop: FDP exceeds target
        if (fh(n_v - 1) > opt_.tFDR) break;

        // Dummy-burn stop: if the number of reals above the 0.5 voting
        // threshold isn't growing, the experiments are only selecting
        // dummies.  The FDP estimator is blind in this regime (R=0 or
        // R=const → FDP_hat=0), so it will never trigger.
        if (n_above_half <= prev_n_above) {
            ++stale_count;
            if (stale_count >= opt_.max_stale_strides) {
                if (opt_.verbose)
                    std::cout << "[TRexVD] stopping: reals above 0.5 stuck at "
                              << n_above_half << " for "
                              << stale_count << " consecutive strides\n";
                break;
            }
        } else {
            stale_count = 0;
        }
        prev_n_above = n_above_half;
    }

    solvers.clear();

    const int T_rows = T_stop;
    Eigen::MatrixXd FDP_hat_mat(T_rows, n_v);
    Eigen::MatrixXd Phi_mat(T_rows, p);
    Vec Phi_prime;

    for (int t = 1; t <= T_rows; ++t) {
        Vec Phi_t = phi_T_mat.col(t - 1);
        Phi_mat.row(t - 1) = Phi_t.transpose();
        Vec pp = Phi_prime_fun_(p, t, num_dummies, phi_T_mat.leftCols(t), Phi_t);
        Vec fh = fdp_hat_(V, Phi_t, pp);
        FDP_hat_mat.row(t - 1) = fh.transpose();
        if (t == T_rows) Phi_prime = std::move(pp);
    }

    SelectResult sel = select_var_(p, opt_.tFDR, T_rows, FDP_hat_mat, Phi_mat, V);

    TRexVDResult out;
    out.selected_var  = sel.selected_var;
    out.v_thresh      = sel.v_thresh;
    out.T_stop        = T_rows;
    out.num_dummies   = num_dummies;
    out.L_calibrated  = num_dummies;
    out.V             = V;
    out.FDP_hat_mat   = std::move(FDP_hat_mat);
    out.Phi_mat       = std::move(Phi_mat);
    out.Phi_prime     = std::move(Phi_prime);
    out.K             = opt_.K;
    return out;
}

// ========================================================================
// L calibration: scan L = p, 2p, 3p, ... at T=1, v=0.75
// ========================================================================
int TRexVD::calibrate_L_(
    const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
    int p, int n_threads)
{
    const double v_calib = 0.75;

    for (int L_mult = 1; L_mult <= opt_.max_L_factor; ++L_mult) {
        const int L = L_mult * p;

        if (opt_.verbose)
            std::cout << "[TRexVD] calibrate_L: trying L=" << L
                      << " (" << L_mult << "p)\n";

        auto seeds = make_seeds_(opt_.K);

        // Run K experiments at T=1
        std::vector<std::unique_ptr<vd::VD_Base>> solvers(opt_.K);
        for (int k = 0; k < opt_.K; ++k)
            solvers[k] = make_solver_(X, y, L, seeds[k], /*T_stop=*/1);

        MatC phi_sum = MatC::Zero(p, 1);

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        #endif
        for (int k = 0; k < opt_.K; ++k) {
            solvers[k]->run(1);
            Vec mask = Vec::Zero(p);
            for (const auto& af : solvers[k]->active_features_copy()) {
                if (af.kind == vd::ActiveFeature::Kind::Real) {
                    int j = af.index;
                    if (j >= 0 && j < p) mask(j) = 1.0;
                }
            }
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            phi_sum.col(0) += mask;
        }

        solvers.clear();
        phi_sum.array() /= double(opt_.K);
        Vec Phi = phi_sum.col(0);

        Vec pp = Phi_prime_fun_(p, 1, L, phi_sum, Phi);

        int R = 0; double num = 0.0;
        for (int j = 0; j < p; ++j)
            if (Phi(j) > v_calib) { ++R; num += (1.0 - pp(j)); }
        double fdp = (R == 0) ? 0.0 : std::min(1.0, num / double(R));

        if (opt_.verbose)
            std::cout << "[TRexVD] calibrate_L: L=" << L
                      << " FDP_hat(v=0.75)=" << fdp << "\n";

        if (fdp <= opt_.tFDR) {
            if (opt_.verbose)
                std::cout << "[TRexVD] calibrate_L: accepted L=" << L << "\n";
            return L;
        }
    }

    int max_L = opt_.max_L_factor * p;
    if (opt_.verbose)
        std::cout << "[TRexVD] calibrate_L: ceiling, using L=" << max_L << "\n";
    return max_L;
}

// ========================================================================
// Main entry point
// ========================================================================
TRexVDResult TRexVD::run(
    const Eigen::Ref<const MatC>& X,
    const Eigen::Ref<const Vec>& y)
{
    const int n = (int)X.rows(), p = (int)X.cols();

    // Thread count (omp_get_max_threads() is stubbed to 1 without OpenMP)
    int n_threads;
    if (opt_.n_threads > 0)
        n_threads = opt_.n_threads;
    else
        n_threads = std::min(opt_.K, omp_get_max_threads());

    // Voting grid
    Vec V = make_V_(opt_.K, opt_.eps);

    if (opt_.verbose) {
        const char* solver_names[] = {"LARS", "OMP", "AFS", "AFS_Logistic"};
        const char* calib_names[]  = {"FixedTL", "CalibrateT", "CalibrateL", "CalibrateBoth"};
        std::cout << "[TRexVD] p=" << p << " n=" << n
                  << " K=" << opt_.K
                  << " solver=" << solver_names[(int)opt_.solver]
                  << " calib=" << calib_names[(int)opt_.calib]
                  << " threads=" << n_threads
                  << " mode=" << (opt_.posthoc_mode ? "posthoc" : "early-stop")
                  << "\n";
    }

    // ---- Determine L ----
    int num_dummies;
    switch (opt_.calib) {
        case CalibMode::FixedTL:
        case CalibMode::CalibrateT:
            num_dummies = opt_.L_factor * p;
            break;
        case CalibMode::CalibrateL:
        case CalibMode::CalibrateBoth:
        default:
            num_dummies = calibrate_L_(X, y, p, n_threads);
            break;
    }

    // ---- Determine T_stop ----
    int Tstop;
    if (opt_.T_stop > 0) {
        Tstop = opt_.T_stop;
    } else {
        Tstop = std::min(num_dummies, (int)std::ceil(n / 2.0));
    }

    if (opt_.verbose)
        std::cout << "[TRexVD] using L=" << num_dummies
                  << " T_stop=" << Tstop << "\n";

    // ---- Dispatch ----
    TRexVDResult out;

    switch (opt_.calib) {
        case CalibMode::FixedTL:
        case CalibMode::CalibrateL:
            // Fixed T: run at exactly Tstop, sweep v only
            out = run_fixed_T_(X, y, Tstop, num_dummies, n_threads, V);
            break;

        case CalibMode::CalibrateT:
        case CalibMode::CalibrateBoth:
        default:
            // Search over T: use posthoc or early-stop per user setting
            if (opt_.posthoc_mode)
                out = run_posthoc_(X, y, Tstop, num_dummies, n_threads, V);
            else
                out = run_early_stop_(X, y, Tstop, num_dummies, n_threads, V);
            break;
    }

    if (opt_.verbose)
        std::cout << "[TRexVD] selected " << out.selected_var.size()
                  << " variables at v=" << out.v_thresh
                  << " T=" << out.T_stop
                  << " L=" << out.num_dummies << "\n";

    return out;
}

}  // namespace trex::trex_selector_methods::trex_vd
