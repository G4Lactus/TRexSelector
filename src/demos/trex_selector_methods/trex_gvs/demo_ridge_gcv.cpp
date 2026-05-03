// demo_ridge_gcv.cpp
// Demonstrates the RidgeGCV class.
//
// Two scenarios:
//   (A) n > p regime: GCV has a clear interior minimum.
//   (B) p >> n regime: GCV is degenerate (rank = n, model interpolates).
//       Here we show that effective regularization can still be selected
//       by k-fold CV, or by the per-observation GCV formula (Eq. 6.1.19).

#include "ridge_gcv.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
struct SyntheticData {
    Eigen::MatrixXd X_train, X_test;
    Eigen::VectorXd y_train, y_test;
    Eigen::VectorXd beta_true;
    double noise_sigma;
};

SyntheticData generate_data(
    int n_train, int n_test, int p, int s_active,
    double signal_strength, double noise_sigma, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    int n_total = n_train + n_test;

    Eigen::MatrixXd X(n_total, p);
    for (int i = 0; i < n_total; ++i)
        for (int j = 0; j < p; ++j)
            X(i, j) = normal(rng);

    Eigen::VectorXd beta_true = Eigen::VectorXd::Zero(p);
    for (int j = 0; j < s_active; ++j)
        beta_true(j) = signal_strength * ((j % 2 == 0) ? 1.0 : -1.0);

    Eigen::VectorXd y(n_total);
    Eigen::VectorXd signal = X * beta_true;
    for (int i = 0; i < n_total; ++i)
        y(i) = signal(i) + noise_sigma * normal(rng);

    SyntheticData data;
    data.X_train = X.topRows(n_train);
    data.X_test  = X.bottomRows(n_test);
    data.y_train = y.head(n_train);
    data.y_test  = y.tail(n_test);
    data.beta_true = beta_true;
    data.noise_sigma = noise_sigma;
    return data;
}

double test_mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return (y_true - y_pred).squaredNorm() / y_true.size();
}

// ---------------------------------------------------------------------------
//  K-fold Cross-Validation (for p >> n where GCV degenerates)
// ---------------------------------------------------------------------------
double kfold_cv(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                double lambda, int K = 5, unsigned seed = 123)
{
    int n = X.rows();
    // Generate random fold assignments.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    double total_sse = 0.0;
    int total_test = 0;

    for (int fold = 0; fold < K; ++fold) {
        int fold_lo = fold * n / K;
        int fold_hi = (fold + 1) * n / K;
        int n_test = fold_hi - fold_lo;
        int n_train = n - n_test;

        // Split
        Eigen::MatrixXd X_tr(n_train, X.cols());
        Eigen::VectorXd y_tr(n_train);
        Eigen::MatrixXd X_te(n_test, X.cols());
        Eigen::VectorXd y_te(n_test);

        int tr_i = 0, te_i = 0;
        for (int i = 0; i < n; ++i) {
            if (i >= fold_lo && i < fold_hi) {
                X_te.row(te_i) = X.row(idx[i]);
                y_te(te_i) = y(idx[i]);
                ++te_i;
            } else {
                X_tr.row(tr_i) = X.row(idx[i]);
                y_tr(tr_i) = y(idx[i]);
                ++tr_i;
            }
        }

        ridge::RidgeGCV model;
        model.fit(X_tr, y_tr);
        Eigen::VectorXd y_pred = model.predict(X_te, lambda);
        total_sse += (y_te - y_pred).squaredNorm();
        total_test += n_test;
    }
    return total_sse / total_test;
}

// ---------------------------------------------------------------------------
void run_scenario(const char* label,
                  int n_train, int n_test, int p, int s, double signal,
                  double noise, unsigned seed)
{
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  Scenario: " << label << "\n"
              << "  n_train = " << n_train << ",  p = " << p
              << ",  s_active = " << s
              << ",  signal = " << signal << ",  noise = " << noise << "\n"
              << std::string(70, '=') << "\n\n";

    auto data = generate_data(n_train, n_test, p, s, signal, noise, seed);

    ridge::RidgeGCV model;
    auto t0 = std::chrono::high_resolution_clock::now();
    model.fit(data.X_train, data.y_train);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "SVD: rank = " << model.rank()
              << ",  sigma_max = " << std::fixed << std::setprecision(2)
              << model.singular_values()(0)
              << ",  sigma_min = " << model.singular_values()(model.rank()-1)
              << ",  fit time = " << std::setprecision(1) << fit_ms << " ms\n\n";

    // GCV-optimal lambda
    auto t2 = std::chrono::high_resolution_clock::now();
    double lambda_gcv = model.gcv_optimal();
    auto t3 = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd y_pred_gcv = model.predict(data.X_test, lambda_gcv);
    double mse_gcv = test_mse(data.y_test, y_pred_gcv);

    std::cout << "GCV-optimal:\n"
              << "  lambda          = " << std::scientific << std::setprecision(4)
              << lambda_gcv << "\n"
              << "  GCV score       = " << model.gcv(lambda_gcv) << "\n"
              << "  effective df    = " << std::fixed << std::setprecision(2)
              << model.effective_df(lambda_gcv)
              << "  (out of " << model.n_samples() << " samples)\n"
              << "  Test MSE        = " << std::scientific << std::setprecision(4) << mse_gcv << "\n"
              << "  Search time     = " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n\n";

    // Oracle (test-set optimal)
    Eigen::VectorXd grid = model.default_lambda_grid(200, 1e6);
    double best_mse = 1e30, lambda_oracle = 0;
    for (Eigen::Index k = 0; k < grid.size(); ++k) {
        Eigen::VectorXd yp = model.predict(data.X_test, grid(k));
        double m = test_mse(data.y_test, yp);
        if (m < best_mse) { best_mse = m; lambda_oracle = grid(k); }
    }
    std::cout << "Oracle (test-MSE optimal):\n"
              << "  lambda          = " << std::scientific << lambda_oracle << "\n"
              << "  Test MSE        = " << best_mse << "\n"
              << "  effective df    = " << std::fixed << std::setprecision(2)
              << model.effective_df(lambda_oracle) << "\n\n";

    // K-fold CV (for p >> n comparison)
    if (p > n_train) {
        double best_kfcv = 1e30, lambda_kfcv = 0;
        for (Eigen::Index k = 0; k < grid.size(); ++k) {
            double cv = kfold_cv(data.X_train, data.y_train, grid(k), 5, 123);
            if (cv < best_kfcv) { best_kfcv = cv; lambda_kfcv = grid(k); }
        }
        Eigen::VectorXd y_pred_kf = model.predict(data.X_test, lambda_kfcv);
        double mse_kf = test_mse(data.y_test, y_pred_kf);
        std::cout << "5-fold CV (alternative for p >> n):\n"
                  << "  lambda          = " << std::scientific << lambda_kfcv << "\n"
                  << "  CV score        = " << best_kfcv << "\n"
                  << "  Test MSE        = " << mse_kf << "\n"
                  << "  effective df    = " << std::fixed << std::setprecision(2)
                  << model.effective_df(lambda_kfcv) << "\n\n";
    }

    // Path table
    auto path = model.solve_path(grid);
    std::cout << std::setw(16) << "lambda"
              << std::setw(16) << "GCV"
              << std::setw(10) << "df"
              << std::setw(16) << "Test MSE" << "\n"
              << std::string(58, '-') << "\n";

    for (Eigen::Index k = 0; k < grid.size(); k += (grid.size() / 10)) {
        Eigen::VectorXd yp = model.predict(data.X_test, grid(k));
        double mse = test_mse(data.y_test, yp);
        std::cout << std::scientific << std::setprecision(3)
                  << std::setw(16) << grid(k)
                  << std::setw(16) << path.gcv_scores(k)
                  << std::fixed << std::setprecision(1)
                  << std::setw(10) << path.df_effective(k)
                  << std::scientific << std::setprecision(3)
                  << std::setw(16) << mse << "\n";
    }
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== Ridge Regression with GCV — Golub, Van Loan §6.1.4 ===\n";

    // Scenario A: Classical regime n > p (GCV works well).
    run_scenario("n > p  (classical, GCV works)",
                 500, 500, 50, 10, 3.0, 1.0, 42);

    // Scenario B: High-dimensional p >> n (GCV degenerates, show k-fold CV).
    run_scenario("p >> n  (high-dim, GCV degenerates)",
                 100, 500, 2000, 10, 3.0, 1.0, 42);

    std::cout << "\n=== Done ===\n";
    return 0;
}
