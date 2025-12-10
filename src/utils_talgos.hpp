#ifndef UTILS_TALGOS_HPP
#define UTILS_TALGOS_HPP

#include <iostream>
#include <algorithm>
#include <Eigen/Dense>

#include "utils_eval.hpp"

/**
 * @brief Generate synthetic regression data with known support
 */
struct SyntheticData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::VectorXd beta_true;
    std::vector<std::size_t> true_support;

    SyntheticData(std::size_t n,
                  std::size_t p,
                  const std::vector<std::size_t>& support,
                  const std::vector<double>& coefs,
                  double noise_level = 1.0)
        : true_support(support) {
        X = Eigen::MatrixXd::Random(n, p);
        y = Eigen::VectorXd::Zero(n);
        beta_true = Eigen::VectorXd::Zero(p);

        for (std::size_t i = 0; i < support.size(); ++i) {
            beta_true[support[i]] = coefs[i];
            y += coefs[i] * X.col(support[i]);
        }
        y += noise_level * Eigen::VectorXd::Random(n);
    }
};


/**
 * @brief Print T-LARS configuration
 */
inline void print_tlars_config(std::size_t n,
                               std::size_t p,
                               std::size_t num_dummies,
                               std::size_t T_stop,
                               const std::vector<std::size_t>& support,
                               const std::vector<double>& coefs,
                               double sigma = 1.0) {

    std::cout << "Configuration:\n"
              << "  Observations (n): " << n << "\n"
              << "  Predictors (p): " << p << "\n"
              << "  Dummies: " << num_dummies << "\n"
              << "  T_stop: " << T_stop << "\n"
              << "  True support: ";
    for (const auto& idx : support) {
        std::cout << idx << " ";
    }
    std::cout << "\n  Coefficients: ";
    for (const auto& coef : coefs) {
        std::cout << coef << " ";
    }
    std::cout << "\n  Noise (σ): " << sigma << "\n\n";
}


/**
 * @brief Print selected variables
 */
template<typename SolverType>
inline void print_selection(
    const SolverType& solver,
    const std::vector<std::size_t>& true_support)
{
    std::cout << "\nSelection:\n";
    // 1. Predictors in active set
    std::cout << "  Predictors: ";
    utils_eval::print_vector(solver.getActivePredictorIndices());

    // 2. Dummies in active set
    std::cout << "\n  Dummies: ";
    utils_eval::print_vector(solver.getActiveDummyIndices());

    // 3. Full selection path (with dummies colored)
    auto actions = solver.getActions();  // vector<vector<int>> - actions
    std::size_t dummy_start = solver.getDummyStartIndex();
    std::cout << "\n  Path: [";
    bool first = true;
    for (std::size_t step = 0; step < actions.size(); ++step) {
        for (int action : actions[step]) {
            if (!first) std::cout << ", ";
            first = false;

            if (action < 0) {
                // REMOVAL EVENT
                std::size_t removed_idx = static_cast<std::size_t>(-action);
                if (removed_idx >= dummy_start) {
                    // Dummy removal: magenta
                    std::cout << "\033[35m-D" << (removed_idx - dummy_start)
                              << "\033[0m";
                } else {
                    // Variable removal: yellow
                    std::cout << "\033[33m-v" << removed_idx << "\033[0m";
                }
            } else {
                // ADDITION EVENT
                std::size_t idx = static_cast<std::size_t>(action);
                bool is_tp = std::find(true_support.begin(),
                                       true_support.end(), idx)
                                       != true_support.end();
                if (idx >= dummy_start) {
                    // Dummy addition: blue
                    std::cout << "\033[34m+D" << (idx - dummy_start)
                              << "\033[0m";
                } else if (is_tp) {
                    // True positive variable addition: green
                    std::cout << "\033[32m+v" << idx << "\033[0m";
                } else {
                    // False positive variable addition: red
                    std::cout << "\033[31m+v" << idx << "\033[0m";
                }
            }
        }
    }
    std::cout << "]\n";
}


/**
 * @brief Print selection quality
 */
template<typename SolverType>
inline void print_quality(const SolverType& solver,
                         const std::vector<std::size_t>& true_support) {

    auto selected = solver.getActivePredictorIndices();
    std::size_t tp = 0;
    for (std::size_t idx : selected) {
        if (std::find(true_support.begin(),
                      true_support.end(), idx) != true_support.end()) {
            tp++;
        }
    }
    double precision = selected.size() > 0 ? static_cast<double>(tp)
                        / selected.size() : 0.0;
    double recall = true_support.size() > 0 ? static_cast<double>(tp)
                        / true_support.size() : 0.0;

    std::cout << "\nQuality: TP=" << tp << "/" << true_support.size()
              << ", FP=" << (selected.size() - tp)
              << ", Precision=" << std::fixed << std::setprecision(1)
              << (precision * 100) << "%"
              << ", Recall=" << (recall * 100) << "%\n";
}


#endif /* UTILS_TALGOS_HPP */
