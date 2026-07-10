// ============================================================================
// validation_ts_01_tenet_aug_comparison.cpp
// ============================================================================
/**
 * @file validation_ts_01_tenet_aug_comparison.cpp
 *
 * @brief Equivalence check: TENET_Solver (Gram-based EN) vs. TENETAug_Solver
 *        (augmented-LASSO EN) under two preprocessing scenarios.
 *
 * @details Reproduces an R elastic-net comparison workflow in C++.
 *
 *  Two scenarios, each using the standardiser from ml_methods/scaler_methods
 *  to pre-scale the data on the shared maps before both solvers run:
 *
 *    1. Z-score:  ZScoreScaler  (centre + divide by sample SD, Bessel-corrected).
 *    2. L2-norm:  LpNormScaler  (centre + divide by column L2 norm).
 *                 This matches the L2 preprocessing used by the R reference.
 *
 *  Both solvers receive the same pre-scaled maps (normalize=false, intercept=false).
 *  Pre-scaling via the standardiser ensures both TENET and TENETAug see identical
 *  data — essential when sharing Eigen::Map objects across two solver instances.
 *
 *  Problem dimensions (mirroring the R demo):
 *    n = 90,  p = 150,  L = 3*p = 450 dummies,  lambda2 = 100.01
 *  True signal:  indices {10, 50, 85}  with coefficients {2.5, -1.8, 3.2}
 *
 *  Comparison metrics reported at each LARS step:
 *    - RMSE to true beta (de-normalised back to the original data scale)
 *    - L2 distance between TENET and TENETAug X-coefficient paths
 *      (in the normalised space — same scale for both solvers)
 */
// ============================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_aug_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>

// ml_methods includes
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>

// utils includes
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>


// ============================================================================
// Namespace aliases
// ============================================================================
namespace datagen   = trex::utils::datageneration::datagen;
namespace scaler    = trex::ml_methods::scaler_methods;
namespace cdiagnost = trex::utils::eval::cdiagnostics;
namespace tsolvers  = trex::tsolvers::linear_model::lars_based;


// ============================================================================
// ScalerType: selects the preprocessing applied inside run_comparison
// ============================================================================
enum class ScalerType {
    ZScore,  ///< Centre + divide by sample SD (Bessel-corrected)
    L2Norm   ///< Centre + divide by column L2 norm  (matches the R reference)
};


// ============================================================================
// run_comparison
// ============================================================================
/**
 * @brief Apply a standardiser then run TENET and TENETAug on shared maps.
 *
 * The standardiser is applied once, in-place on the shared Eigen::Map objects,
 * before either solver runs.  Both solvers therefore see identical input and
 * are called with normalize=false / intercept=false.
 *
 * @param scenario_name  Label printed as section header.
 * @param X              Raw predictor matrix (n × p) — passed by value;
 *                       the standardiser modifies the local copy in-place.
 * @param D              Raw dummy matrix (n × L) — by value.
 * @param y              Raw response vector (n) — by value, centred here.
 * @param true_beta_X    True p-dimensional beta in the ORIGINAL (raw) space.
 * @param scaler_type    Which standardiser to apply.
 * @param lambda2        Elastic-net L2 (ridge) regularisation parameter.
 * @param T_stop         Dummy-entry threshold passed to executeStep().
 * @param p              Number of original X predictors.
 */
void run_comparison(
    const std::string&     scenario_name,
    Eigen::MatrixXd        X,
    Eigen::MatrixXd        D,
    Eigen::VectorXd        y,
    const Eigen::VectorXd& true_beta_X,
    ScalerType             scaler_type,
    double                 lambda2,
    std::size_t            T_stop,
    std::size_t            p)
{
    cdiagnost::print_section_header(scenario_name);

    const Eigen::Index p_idx = static_cast<Eigen::Index>(p);
    const double       n_d   = static_cast<double>(X.rows());

    // Shared maps — both solvers point here; scaling must happen once, here.
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::MatrixXd> D_map(D.data(), D.rows(), D.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    // ------------------------------------------------------------------
    // Compute scale_X from raw data BEFORE the standardiser runs.
    // Used later to de-normalise estimated betas for RMSE computation.
    //   Z-score:  scale_X[j] = SD_j  = ||X_centred_j||_2 / sqrt(n-1)
    //   L2-norm:  scale_X[j] = L2_j  = ||X_centred_j||_2
    // ------------------------------------------------------------------
    Eigen::VectorXd scale_X(p_idx);
    const double scale_div = (scaler_type == ScalerType::ZScore)
                             ? std::sqrt(n_d - 1.0)
                             : 1.0;
    for (Eigen::Index j = 0; j < p_idx; ++j) {
        const double mean_j = X_map.col(j).mean();
        scale_X[j] = (X_map.col(j).array() - mean_j).matrix().norm()
                     / scale_div;
    }

    // ------------------------------------------------------------------
    // Apply standardiser in-place (modifies X_map and D_map).
    // Centre y separately — both scalers handle only matrices.
    // ------------------------------------------------------------------
    if (scaler_type == ScalerType::ZScore) {
        scaler::ZScoreScaler zs_X, zs_D;
        zs_X.fit(X_map); zs_X.transform_inplace(X_map);
        zs_D.fit(D_map); zs_D.transform_inplace(D_map);
    } else {
        scaler::LpNormScaler l2_X(scaler::LpNormScaler::NormType::L2, /*center=*/true);
        scaler::LpNormScaler l2_D(scaler::LpNormScaler::NormType::L2, /*center=*/true);
        l2_X.fit(X_map); l2_X.transform_inplace(X_map);
        l2_D.fit(D_map); l2_D.transform_inplace(D_map);
    }
    y_map.array() -= y_map.mean();

    // ------------------------------------------------------------------
    // TENET_Solver — Gram-based elastic net
    // normalize=false / intercept=false: data already scaled above.
    // ------------------------------------------------------------------
    std::cout << "Running TENET_Solver   (Gram-based EN)...\n";
    tsolvers::TENET_Solver tenet(
        X_map, D_map, y_map,
        lambda2,
        /*normalize=*/false,
        /*intercept=*/false,
        /*verbose=*/false);
    tenet.executeStep(T_stop, /*early_stop=*/false);

    // ------------------------------------------------------------------
    // TENETAug_Solver — augmented-LASSO elastic net
    // normalize=false / intercept=false: same pre-scaled maps as TENET.
    // ------------------------------------------------------------------
    std::cout << "Running TENETAug_Solver (augmented LASSO)...\n";
    tsolvers::TENETAug_Solver tenet_aug(
        X_map, D_map, y_map,
        lambda2,
        /*normalize=*/false,
        /*intercept=*/false,
        /*verbose=*/false);
    tenet_aug.executeStep(T_stop, /*early_stop=*/false);

    // ------------------------------------------------------------------
    // Method 3: plain TLASSO on MANUALLY-augmented matrices.
    // Literal replica of the R `lasso_star` elastic-net construction:
    //     d1 = sqrt(lambda2),  d2 = 1/sqrt(1+lambda2)
    //     Xstar = d2 * rbind(X, d1*I),   ystar = c(y, 0)
    //     lasso_star <- lars(Xstar, ystar, normalize=FALSE, intercept=FALSE)
    //     beta       <- lasso_star$beta / d2
    // Crucially this uses normalize=FALSE (NO inner re-normalisation), so the
    // augmented columns are fed to the LASSO exactly as constructed. Under L2
    // they are unit-norm by construction (d2^2*(1+lambda2)=1); under z-score
    // they are NOT, and this is the genuine reference behaviour. Comparing this
    // manual replica against TENETAug (constructed above with normalize=false on
    // the same pre-scaled maps) checks that the augmented-LASSO construction
    // reproduces the reference elastic-net path.
    // ------------------------------------------------------------------
    std::cout << "Running TLASSO on manual augmented data (R lasso_star)...\n";
    const double d1   = std::sqrt(lambda2);
    const double d2   = 1.0 / std::sqrt(1.0 + lambda2);
    const Eigen::Index n_idx = X_map.rows();
    const Eigen::Index L_idx = D_map.cols();
    const Eigen::Index naug  = n_idx + p_idx + L_idx;

    Eigen::MatrixXd X_aug = Eigen::MatrixXd::Zero(naug, p_idx);
    X_aug.topRows(n_idx) = d2 * X_map;
    for (Eigen::Index j = 0; j < p_idx; ++j) {
        X_aug(n_idx + j, j) = d2 * d1;
    }

    Eigen::MatrixXd D_aug = Eigen::MatrixXd::Zero(naug, L_idx);
    D_aug.topRows(n_idx) = d2 * D_map;
    for (Eigen::Index j = 0; j < L_idx; ++j) {
        D_aug(n_idx + p_idx + j, j) = d2 * d1;
    }

    Eigen::VectorXd y_aug = Eigen::VectorXd::Zero(naug);
    y_aug.head(n_idx) = y_map;

    Eigen::Map<Eigen::MatrixXd> Xa_map(X_aug.data(), X_aug.rows(), X_aug.cols());
    Eigen::Map<Eigen::MatrixXd> Da_map(D_aug.data(), D_aug.rows(), D_aug.cols());
    Eigen::Map<Eigen::VectorXd> ya_map(y_aug.data(), y_aug.size());

    tsolvers::TLASSO_Solver tlasso_aug(
        Xa_map, Da_map, ya_map,
        /*normalize=*/false,
        /*intercept=*/false,
        /*verbose=*/false);
    tlasso_aug.executeStep(T_stop, /*early_stop=*/false);

    // ------------------------------------------------------------------
    // Retrieve beta-coefficient paths
    // Each returns a (p + L) × n_steps matrix: columns = steps, rows = predictors.
    // The manual TLASSO path is de-normalised by /d2 (R: lasso_star$beta / d2).
    // ------------------------------------------------------------------
    const Eigen::MatrixXd path_tenet  = tenet.getBetaPath();
    const Eigen::MatrixXd path_aug    = tenet_aug.getBetaPath();
    const Eigen::MatrixXd path_manual = tlasso_aug.getBetaPath() / d2;

    const Eigen::Index n_steps = std::min({
        path_tenet.cols(), path_aug.cols(), path_manual.cols() });

    std::cout << "\nPath lengths:  TENET=" << path_tenet.cols()
              << "  TENETAug=" << path_aug.cols()
              << "  TLASSOaug=" << path_manual.cols()
              << "  (comparing " << n_steps << " steps)\n\n";

    // ------------------------------------------------------------------
    // Per-step metrics for all three methods
    // ------------------------------------------------------------------
    const auto n_steps_sz = static_cast<std::size_t>(n_steps);
    std::vector<double> rmse_tenet (n_steps_sz);
    std::vector<double> rmse_aug   (n_steps_sz);
    std::vector<double> rmse_manual(n_steps_sz);
    std::vector<double> d_ten_aug  (n_steps_sz);  // TENET   vs TENETAug
    std::vector<double> d_ten_man  (n_steps_sz);  // TENET   vs TLASSOaug
    std::vector<double> d_aug_man  (n_steps_sz);  // TENETAug vs TLASSOaug

    for (Eigen::Index k = 0; k < n_steps; ++k) {
        const auto sz_k = static_cast<std::size_t>(k);

        const Eigen::VectorXd bx_ten = path_tenet .col(k).head(p_idx);
        const Eigen::VectorXd bx_aug = path_aug   .col(k).head(p_idx);
        const Eigen::VectorXd bx_man = path_manual.col(k).head(p_idx);

        // De-normalise X block to original space for RMSE
        const Eigen::VectorXd o_ten = bx_ten.cwiseQuotient(scale_X);
        const Eigen::VectorXd o_aug = bx_aug.cwiseQuotient(scale_X);
        const Eigen::VectorXd o_man = bx_man.cwiseQuotient(scale_X);

        const double inv_sqrt_p = 1.0 / std::sqrt(static_cast<double>(p));
        rmse_tenet [sz_k] = (o_ten - true_beta_X).norm() * inv_sqrt_p;
        rmse_aug   [sz_k] = (o_aug - true_beta_X).norm() * inv_sqrt_p;
        rmse_manual[sz_k] = (o_man - true_beta_X).norm() * inv_sqrt_p;

        // Pairwise L2 distances in the normalised space (equal footing)
        d_ten_aug[sz_k] = (bx_ten - bx_aug).norm();
        d_ten_man[sz_k] = (bx_ten - bx_man).norm();
        d_aug_man[sz_k] = (bx_aug - bx_man).norm();
    }

    // ------------------------------------------------------------------
    // Print sparse comparison table (every 5th step + last step)
    // ------------------------------------------------------------------
    constexpr int cw = 13;
    std::cout << std::setw(6)  << "step"
              << std::setw(cw) << "RMSE_TEN"
              << std::setw(cw) << "RMSE_AUG"
              << std::setw(cw) << "RMSE_TLS"
              << std::setw(cw) << "d(TEN,AUG)"
              << std::setw(cw) << "d(TEN,TLS)"
              << std::setw(cw) << "d(AUG,TLS)"
              << "\n"
              << std::string(6 + 6 * cw, '-') << "\n";

    constexpr Eigen::Index stride = 5;
    for (Eigen::Index k = 0; k < n_steps; ++k) {
        if (k % stride == 0 || k == n_steps - 1) {
            const auto sz_k = static_cast<std::size_t>(k);
            std::cout << std::setw(6) << k
                      << std::setw(cw) << std::fixed << std::setprecision(5)
                      << rmse_tenet[sz_k]
                      << std::setw(cw) << std::fixed << std::setprecision(5)
                      << rmse_aug[sz_k]
                      << std::setw(cw) << std::fixed << std::setprecision(5)
                      << rmse_manual[sz_k]
                      << std::setw(cw) << std::scientific << std::setprecision(2)
                      << d_ten_aug[sz_k]
                      << std::setw(cw) << std::scientific << std::setprecision(2)
                      << d_ten_man[sz_k]
                      << std::setw(cw) << std::scientific << std::setprecision(2)
                      << d_aug_man[sz_k]
                      << "\n";
        }
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    const auto max_of  = [](const std::vector<double>& v) {
        return *std::max_element(v.begin(), v.end()); };
    const auto mean_of = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0)
               / static_cast<double>(v.size()); };
    const auto min_of  = [](const std::vector<double>& v) {
        return *std::min_element(v.begin(), v.end()); };

    std::cout << "\n--- Summary (normalised-space pairwise L2 distances) ---\n"
              << std::scientific << std::setprecision(4)
              << "  max  d(TEN,AUG): " << max_of(d_ten_aug)
              << "   mean: " << mean_of(d_ten_aug) << "\n"
              << "  max  d(TEN,TLS): " << max_of(d_ten_man)
              << "   mean: " << mean_of(d_ten_man) << "\n"
              << "  max  d(AUG,TLS): " << max_of(d_aug_man)
              << "   mean: " << mean_of(d_aug_man) << "\n"
              << std::fixed << std::setprecision(6)
              << "\n--- Best / Final RMSE ---\n"
              << "  Best  RMSE  TENET="   << min_of(rmse_tenet)
              << "  AUG=" << min_of(rmse_aug)
              << "  TLS=" << min_of(rmse_manual) << "\n"
              << "  Final RMSE  TENET="   << rmse_tenet.back()
              << "  AUG=" << rmse_aug.back()
              << "  TLS=" << rmse_manual.back() << "\n";

    // ------------------------------------------------------------------
    // Verdict: which two methods agree?  (TLASSOaug = R lasso_star reference)
    // ------------------------------------------------------------------
    const auto verdict = [](const char* a, const char* b, double maxd) {
        std::cout << "  " << a << " vs " << b << ": ";
        if      (maxd < 1e-8) std::cout << "EQUIVALENT (< 1e-8)\n";
        else if (maxd < 1e-4) std::cout << "NEARLY EQUIVALENT (< 1e-4)\n";
        else                  std::cout << "DIFFER (max L2 = "
                                        << std::scientific << std::setprecision(3)
                                        << maxd << ")\n";
    };
    std::cout << "\n--- Verdict (TLASSOaug == lasso_star reference) ---\n";
    verdict("TENET   ", "TENETAug ", max_of(d_ten_aug));
    verdict("TENET   ", "TLASSOaug", max_of(d_ten_man));
    verdict("TENETAug", "TLASSOaug", max_of(d_aug_man));
    std::cout << "\n\n";
}


// ============================================================================
// main
// ============================================================================
int main() {

    // -------------------------------------------------------- Problem constants
    // Dimensions and signal match the R elastic-net comparison reference.
    constexpr std::size_t n       = 90;
    constexpr std::size_t p       = 150;
    constexpr std::size_t L       = 3*p;
    constexpr double      lambda2 = 100.01;
    // T_stop = 10 (dummy-entry threshold), but every executeStep() call below
    // passes early_stop=false, which bypasses this threshold entirely — so each
    // solver runs its full natural path (until no inactive variables remain or
    // the active-set/rank limit is reached), independent of T_stop.
    constexpr std::size_t T_stop  = 10;
    constexpr double      snr     = 1.0;

    const std::vector<std::size_t> true_support = {10, 50, 85};
    const std::vector<double>      true_coefs   = {2.5, -1.8, 3.2};

    // True p-dimensional beta in the original (un-normalised) space
    Eigen::VectorXd true_beta_X =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));
    for (std::size_t i = 0; i < true_support.size(); ++i) {
        true_beta_X[static_cast<Eigen::Index>(true_support[i])] = true_coefs[i];
    }

    // -------------------------------------------------------- Generate raw data
    std::cout << "Generating data:  n=" << n << "  p=" << p
              << "  L=" << L << "  lambda2=" << lambda2 << "\n\n";

    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        static_cast<Eigen::Index>(L),
        true_support,
        true_coefs,
        snr,
        /*seed=*/42,
        /*X_seed=*/-1,
        /*dummy_seed=*/-1,
        datagen::predictor_policy::Normal(),
        datagen::dummygen::Distribution::Normal(),
        datagen::noisegen::noise_policy::Normal()
    );

    // Raw copies — each run_comparison call gets a fresh by-value copy.
    const Eigen::MatrixXd X_raw = data.getX();
    const Eigen::MatrixXd D_raw = data.getD();
    const Eigen::VectorXd y_raw = data.getY();

    // Scenario 1: Z-score scaling  (ZScoreScaler: centre + sample SD)
    run_comparison(
        "Scenario 1: Z-score scaling  (centre + SD)",
        X_raw, D_raw, y_raw,
        true_beta_X, ScalerType::ZScore,
        lambda2, T_stop, p);

    // Scenario 2: L2-norm scaling  (LpNormScaler L2: centre + L2-norm)
    run_comparison(
        "Scenario 2: L2-norm scaling  (centre + L2-normalise)",
        X_raw, D_raw, y_raw,
        true_beta_X, ScalerType::L2Norm,
        lambda2, T_stop, p);

    return 0;
}
