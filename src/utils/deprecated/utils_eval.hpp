#ifndef UTILS_EVAL_HPP
#define UTILS_EVAL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>


/**
 * @brief Configurable logging utility for output redirection.
 */
namespace utils_eval {


inline std::ostream*& get_default_stream_ptr() {
    static std::ostream *default_stream = &std::cout;
    return default_stream;
}

inline std::ostream &get_default_stream() {
    return *get_default_stream_ptr();
}

inline void set_default_stream(std::ostream &stream) {
    get_default_stream_ptr() = &stream;
}



/**
 * @brief Print path metrics for Lars-based learning algorithms
 *
 * @param algo_name String with algorithm name.
 * @param R2 Path of R2 obtained from std::vector<double>.
 * @param RSS Path of RSS obtained from std::vector<double>.
 * @param lambda Path of lambda obtained from std::vector<double>.
 * @param os Output stream (default: std::cout)
 */
void print_path_metrics(const std::string &algo_name,
                        const std::vector<double> &R2,
                        const std::vector<double> &RSS,
                        const std::vector<double> &lambda,
                        std::ostream &os = get_default_stream()) {

    os << "\n" << algo_name << " - Path Metrics:\n";
    os << std::string(50, '-') << "\n";

    // Print first 10 and last 5 steps
    int n_steps = R2.size();
    int show_first = std::min(10, n_steps);
    int show_last = std::min(5, std::max(0, n_steps - show_first));

    os << std::setw(8) << "Step" << std::setw(12) << "R²"
       << std::setw(15) << "RSS" << std::setw(15) << "Lambda\n";
    os << std::string(50, '-') << "\n";

    for (int k = 0; k < show_first; ++k) {
        os << std::setw(8) << k;
        if (static_cast<std::size_t>(k) < R2.size()) {
            os << std::setw(12) << std::fixed
               << std::setprecision(6) << R2[k];
        }
        if (static_cast<std::size_t>(k) < RSS.size()) {
            os << std::setw(15) << std::fixed
               << std::setprecision(4) << RSS[k];
        }
        if (k > 0 && static_cast<std::size_t>(k - 1) < lambda.size()) {
            os << std::setw(15) << std::fixed
               << std::setprecision(4) << lambda[k - 1];
        }
        os << "\n";
    }

    if (show_last > 0 && n_steps > show_first) {
        os << std::setw(8) << "..." << "\n";
        for (int k = n_steps - show_last; k < n_steps; ++k) {
            os << std::setw(8) << k;
            if (static_cast<std::size_t>(k) < R2.size()) {
                os << std::setw(12)
                   << std::fixed << std::setprecision(6) << R2[k];
            }
            if (static_cast<std::size_t>(k) < RSS.size()) {
                os << std::setw(15)
                   << std::fixed << std::setprecision(4) << RSS[k];
            }
            if (k > 0 && static_cast<std::size_t>(k - 1) < lambda.size()) {
                os << std::setw(15) << std::fixed
                   << std::setprecision(4) << lambda[k - 1];
            }
            os << "\n";
        }
    }

    os << "\nTotal steps: " << n_steps << "\n";
    if (!R2.empty()) {
        os << "Final R²: " << R2.back() << "\n";
    }
    if (!RSS.empty()) {
        os << "Final RSS: " << RSS.back() << "\n";
    }
}


inline constexpr double UTILS_NEAR_ZERO = 1e-10;
inline constexpr double UTILS_SELECT_THR = 1e-4;

/**
 * @brief Print a comparison between estimated and true coefficients
 *
 * @tparam LarsClass Solver type providing `getBeta()` and `getIntercept()`.
 * @tparam VectorType Type of vector (e.g., arma::vec, Eigen::VectorXd).
 *
 * @param algo_name Algorithm name.
 * @param solver Solver instance (read-only).
 * @param true_support Indices of true non-zero coefficients.
 * @param true_coefs True coefficient values corresponding to `true_support`.
 * @param p Number of original predictors (before dummy variables).
 * @param os Output stream (default: std::cout)
 */
template <typename LarsClass, typename VectorType>
void print_coefficient_comparison(const std::string &algo_name,
                                  const LarsClass &solver,
                                  const std::vector<std::size_t> &true_support,
                                  const VectorType &true_coefs,
                                  std::size_t p,
                                  std::ostream &os = get_default_stream()) {

    os << "\n" << algo_name << " - Final Coefficients:\n";
    os << std::string(60, '-') << "\n";
    os << std::setw(10) << "Variable" << std::setw(15)
       << "Estimated" << std::setw(15) << "True" << std::setw(15)
       << "Error\n";
    os << std::string(60, '-') << "\n";

    // Get unscaled coefficients
    auto beta_orig = solver.getBeta();
    double intercept = solver.getIntercept();

    // Print true signal variables (keep an O(1) membership test available)
    std::unordered_set<std::size_t> true_set(
        true_support.begin(), true_support.end());

    for (std::size_t i = 0; i < true_support.size(); ++i) {
        std::size_t idx = true_support[i];
        double est = beta_orig(idx);
        double truth = true_coefs(i);
        double error = std::abs(est - truth);

        os << std::setw(10) << idx
           << std::setw(15) << est
           << std::setw(15) << truth
           << std::setw(15) << error
           << "\n";
    }

    // Print zero coefficients
    for (std::size_t j = 0; j < p; ++j) {
        if (true_set.count(j) == 0) {
            double est = beta_orig(j);
            if (std::abs(est) > UTILS_SELECT_THR) {
                os << std::setw(10) << j << std::setw(15)
                   << est << std::setw(15) << "0.0" << std::setw(15)
                   << std::abs(est) << "\n";
            }
        }
    }

    os << "\nIntercept: " << intercept << "\n";
}


/**
 * @brief Analyze the order in which variables enter the active set.
 *
 * Scans the recorded `actions` per LARS step and reports when the first
 * true-signal variable appears, when all true signals have been seen,
 * and when the first dummy variable (index >= p) is selected.
 *
 * @param algo_name Name shown in printed output
 * @param actions Sequence of actions (positive entries = add, negative = drop)
 * @param true_support Indices of the true signal variables
 * @param p Number of original predictors (indices >= p are dummies)
 */
void analyze_entry_order(const std::string &algo_name,
                         const std::vector<std::vector<int>> &actions,
                         const std::vector<std::size_t> &true_support,
                         std::size_t p) {

    std::cout << "\n" << algo_name << " - Variable Entry Order:\n";
    std::cout << std::string(50, '-') << "\n";

    int first_true_at_step = -1;
    int all_true_at_step = -1;
    int first_dummy_at_step = -1;
    std::unordered_set<int> true_found;
    std::unordered_set<std::size_t> true_set(
        true_support.begin(), true_support.end());

    for (size_t k = 0; k < actions.size(); ++k) {
        for (int var : actions[k]) {
            if (var > 0) {
                // Addition
                // Check if true signal
                if (true_set.count(static_cast<std::size_t>(var))) {
                    true_found.insert(var);
                    if (first_true_at_step == -1) {
                        first_true_at_step = k;
                    }
                    std::cout << "Step " << k
                              << ": True signal variable " << var
                              << " entered\n";
                }
                // Check if dummy
                else if (static_cast<std::size_t>(var) >= p) {
                    if (first_dummy_at_step == -1) {
                        first_dummy_at_step = k;
                        std::cout << "Step " << k
                                  << ": FIRST DUMMY (var " << var
                                  << ") entered\n";
                    }
                }

                // Check if all true found
                if (true_found.size() == true_support.size()
                    && all_true_at_step == -1) {
                    all_true_at_step = k;
                }
            }
        }
    }

    std::cout << "\nSummary:\n";
    std::cout << "  First true signal at step: " << first_true_at_step << "\n";
    std::cout << "  All true signals by step: "
              << (all_true_at_step >= 0 ?
                  std::to_string(all_true_at_step) : "Not all found")
              << "\n";
    std::cout << "  First dummy at step: "
              << (first_dummy_at_step >= 0 ?
                  std::to_string(first_dummy_at_step) : "No dummy") << "\n";

    if (first_dummy_at_step >= 0)
    {
        std::cout << "  → T-LARS would stop at step "
                  << first_dummy_at_step << " with T=1\n";
        std::cout << "  → Variables selected before first dummy: "
                  << first_dummy_at_step << "\n";
    }
}






/**
 * @brief Pretty-print the LARS/LASSO solution path
 *
 * Prints a table with step index, added variables, lambda, RSS, R^2, DoF
 * and Cp for up to `max_steps` steps. The `solver` must provide
 * `getActions()`, `getLambda()`, `getRSS()`, `getR2()`, `getDoF()`, and
 * `getCp()` accessors.
 *
 * @tparam SolverType Solver type exposing path accessors
 * @param solver Solver instance (const-ref)
 * @param max_steps Maximum number of steps to show
 */
template <typename SolverType>
void print_lars_path(const SolverType &solver, std::size_t max_steps) {

    const auto &actions = solver.getActions();
    const auto &lambdas = solver.getLambda();
    const auto &rss = solver.getRSS();
    const auto &r2 = solver.getR2();
    const auto &dof = solver.getDoF();
    const auto &cp = solver.getCp();

    std::size_t steps = std::min(lambdas.size(), max_steps + 1);
    std::cout << std::setw(6) << "Step" << std::setw(15)
              << "Added Variable" << std::setw(12) << "Lambda"
              << std::setw(14) << "RSS" << std::setw(10) << "R^2"
              << std::setw(6) << "DoF" << std::setw(12) << "Cp"
              << "\n";

    for (std::size_t k = 0; k < steps; ++k) {
        std::cout << std::setw(6) << k;

        std::ostringstream oss;
        if (k == 0) {
            oss << "-";
        }
        else if (k - 1 < actions.size()) {
            const auto &vec = actions[k - 1];
            if (vec.empty()) {
                oss << "-";
            } else {
                for (std::size_t a = 0; a < vec.size(); ++a) {
                    if (a > 0) {
                        oss << ',';
                    }
                    oss << vec[a];
                }
            }
        } else {
            oss << "?";
        }

        // Pad or trim to fill column
        std::cout << std::setw(15) << std::right << oss.str();

        // Pad numbers consistently, e.g., .3f for lambda, .2f for rss
        std::cout << std::setw(12) << std::right << std::fixed
                  << std::setprecision(3) << lambdas[k] << std::setw(14)
                  << std::right << std::setprecision(2) << rss[k]
                  << std::setw(10) << std::setprecision(4) << r2[k]
                  << std::setw(6) << dof[k] << std::setw(12) << cp[k] << "\n";
    }

    std::cout << "\nFinal active set: ";
    print_vector(std::vector<std::size_t>(solver.getActives().begin(),
     solver.getActives().end()));
    std::cout << "\n";
}



} /* End of namespace utils_eval */

#endif /* End of UTILS_EVAL_HPP */
