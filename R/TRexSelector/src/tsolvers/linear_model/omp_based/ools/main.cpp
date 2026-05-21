/**
 * @file main.cpp
 * @brief Demo & comparison of OLS-QR vs OLS-Cholesky on a synthetic sparse problem.
 *
 * Generates  Φ ∈ R^{n×p}  with random Gaussian entries and unit-norm columns,
 * a K-sparse coefficient vector  s,  and observation  x = Φ s + noise.
 * Then runs both OLS back-ends and compares support recovery & residuals.
 */

#include "ools/ools.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ─────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────
namespace {

Eigen::MatrixXd random_dictionary(Eigen::Index n, Eigen::Index p,
                                  std::mt19937& rng)
{
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::MatrixXd Phi(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = 0; i < n; ++i)
            Phi(i, j) = gauss(rng);
        Phi.col(j).normalize();
    }
    return Phi;
}

void print_result(const std::string& label,
                  const ools::OOLSResult& res,
                  double elapsed_us,
                  const std::vector<Eigen::Index>& true_support)
{
    std::cout << "\n═══════════════════════════════════════\n"
              << "  " << label << "\n"
              << "═══════════════════════════════════════\n"
              << "  Iterations     : " << res.iterations << "\n"
              << "  Residual ‖r‖₂  : " << std::scientific << std::setprecision(6)
              << res.residual_norm << "\n"
              << "  Time (µs)      : " << std::fixed << std::setprecision(0)
              << elapsed_us << "\n"
              << "  Support        : { ";
    for (auto idx : res.support) std::cout << idx << " ";
    std::cout << "}\n"
              << "  True support   : { ";
    for (auto idx : true_support) std::cout << idx << " ";
    std::cout << "}\n"
              << "  Coefficients   : " << std::setprecision(6) << std::fixed
              << res.coefficients.transpose() << "\n";

    // Support recovery check
    std::vector<Eigen::Index> recovered(res.support);
    std::sort(recovered.begin(), recovered.end());
    std::vector<Eigen::Index> truth(true_support);
    std::sort(truth.begin(), truth.end());
    const bool exact = (recovered == truth);
    std::cout << "  Exact recovery : " << (exact ? "YES" : "NO") << "\n";
}

}  // anonymous namespace


// ─────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────
int main()
{
    // ── Problem parameters ─────────────────────────────────────
    constexpr Eigen::Index n = 100;   // observations
    constexpr Eigen::Index p = 500;   // dictionary atoms
    constexpr int K          = 5;     // sparsity level
    constexpr double sigma   = 0.01;  // noise level

    std::mt19937 rng(42);

    // ── Build problem ──────────────────────────────────────────
    Eigen::MatrixXd Phi = random_dictionary(n, p, rng);

    // Choose K random support indices
    std::vector<Eigen::Index> indices(static_cast<std::size_t>(p));
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<Eigen::Index> true_support(indices.begin(),
                                           indices.begin() + K);

    // Build sparse signal
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::VectorXd s_true = Eigen::VectorXd::Zero(p);
    for (auto idx : true_support)
        s_true(idx) = gauss(rng);

    // Observation = Phi * s_true + noise
    Eigen::VectorXd noise(n);
    for (Eigen::Index i = 0; i < n; ++i) noise(i) = sigma * gauss(rng);
    Eigen::VectorXd x = Phi * s_true + noise;

    std::cout << "Problem:  n=" << n << "  p=" << p
              << "  K=" << K << "  σ=" << sigma << "\n"
              << "True coefficients on support: "
              << s_true(true_support).transpose() << "\n";

    // ── Run QR back-end ────────────────────────────────────────
    {
        ools::OOLS solver(Phi, x, ools::Backend::QR);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = solver.solve(static_cast<std::size_t>(K));
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        print_result("OLS  (QR / Modified Gram-Schmidt)", res, us, true_support);
    }

    // ── Run Cholesky back-end ──────────────────────────────────
    {
        ools::OOLS solver(Phi, x, ools::Backend::Cholesky);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = solver.solve(static_cast<std::size_t>(K));
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        print_result("OLS  (Cholesky rank-1 update)", res, us, true_support);
    }

    // ── Also compare against OMP-style selection (for reference) ──
    // (We skip this here — user already has a full OMP.)

    std::cout << "\n─────────────────────────────────────────\n"
              << "  Both back-ends should recover the same support\n"
              << "  and produce identical coefficients (up to numerics).\n"
              << "─────────────────────────────────────────\n";

    return 0;
}
