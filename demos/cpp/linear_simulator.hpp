// ===================================================================================
// linear_simulator.hpp
// ===================================================================================
#ifndef SIMULATION_LINEAR_SIMULATOR_HPP
#define SIMULATION_LINEAR_SIMULATOR_HPP

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace trex::simulation {

class LinearSimulator {
public:
    struct Dataset {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        Eigen::VectorXd beta_true;
        std::vector<std::size_t> support_true;
    };

    /**
     * @brief Generates a synthetic dataset for the linear model y = X*beta + epsilon.
     * * @param n Number of samples.
     * @param p Number of features.
     * @param sparsity Number of non-zero coefficients (cardinality of beta).
     * @param beta_val The constant value for active coefficients.
     * @param snr Signal-to-Noise Ratio to scale the variance of epsilon.
     * @param rho Toeplitz correlation parameter for X (0.0 = independent features).
     * @param seed Random seed for reproducibility across folds/runs.
     */
    static Dataset generate(
        std::size_t n,
        std::size_t p,
        std::size_t sparsity,
        double beta_val = 1.0,
        double snr = 3.0,
        double rho = 0.0,
        uint32_t seed = 42
    ) {
        std::mt19937 gen(seed);
        std::normal_distribution<double> norm_dist(0.0, 1.0);

        Dataset data;
        data.X = Eigen::MatrixXd::Zero(n, p);
        data.beta_true = Eigen::VectorXd::Zero(p);

        // 1. Generate Design Matrix X
        if (std::abs(rho) < 1e-9) {
            // Independent standard normal features
            for (std::size_t j = 0; j < p; ++j) {
                for (std::size_t i = 0; i < n; ++i) {
                    data.X(i, j) = norm_dist(gen);
                }
            }
        } else {
            // Autoregressive AR(1) / Toeplitz correlation structure
            // X_j = rho * X_{j-1} + sqrt(1 - rho^2) * Z
            double scale = std::sqrt(1.0 - rho * rho);
            for (std::size_t i = 0; i < n; ++i) {
                data.X(i, 0) = norm_dist(gen);
                for (std::size_t j = 1; j < p; ++j) {
                    data.X(i, j) = rho * data.X(i, j - 1) + scale * norm_dist(gen);
                }
            }
        }

        // 2. Generate True Support and Beta
        std::vector<std::size_t> indices(p);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        data.support_true.assign(indices.begin(), indices.begin() + sparsity);
        std::sort(data.support_true.begin(), data.support_true.end());

        for (std::size_t idx : data.support_true) {
            data.beta_true(idx) = beta_val;
        }

        // 3. Generate Signal and apply Noise based on SNR
        Eigen::VectorXd signal = data.X * data.beta_true;
        double var_signal = (signal.array() - signal.mean()).square().sum() / (n - 1.0);

        // Variance of noise = Variance of signal / SNR
        double sigma = std::sqrt(var_signal / snr);

        data.y = Eigen::VectorXd::Zero(n);
        for (std::size_t i = 0; i < n; ++i) {
            data.y(i) = signal(i) + sigma * norm_dist(gen);
        }

        return data;
    }
};

} // namespace trex::simulation

#endif /* SIMULATION_LINEAR_SIMULATOR_HPP */
