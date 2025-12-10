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
 * @brief Print section header
 *
 * @param title String with algorithm or section name
 * @param os Output stream (default: std::cout)
 */
inline void print_section_header(const std::string &title,
                                 std::ostream &os = get_default_stream()) {
    os << "\n" << std::string(70, '=') << "\n";
    os << title << "\n";
    os << std::string(70, '=') << "\n";
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


/** @brief Print vector in comma-separated format: {1, 2, 3} */
template <typename T>
inline void print_vector(const std::vector<T> &vec,
                         std::size_t max_show = SIZE_MAX) {

    std::size_t n_show = std::min(max_show, vec.size());

    std::cout << "{";
    for (std::size_t i = 0; i < n_show; ++i) {
        std::cout << vec[i];
        if (i < n_show - 1) {
            std::cout << ", ";
        }
    }
    if (n_show < vec.size()) {
        std::cout << ", ...";
    }
    std::cout << "}";
}


/**
 * @brief Print configuration summary for linear regression demos
 */
inline void print_config(std::size_t n,
                         std::size_t p,
                         double sigma,
                         const std::vector<std::size_t> &support,
                         const std::vector<double> &coefs,
                         std::size_t max_steps = 0) {

    std::cout << "\nConfiguration:\n"
              << "  Observations (n): " << n << "\n"
              << "  Predictors (p): " << p << "\n"
              << "  True support size: " << support.size() << "\n"
              << "  True support: ";

    print_vector(support);
    std::cout << "\n  True coefficients: ";
    print_vector(coefs);
    std::cout << "\n  Noise level (σ): " << sigma << "\n";

    if (max_steps > 0)
    {
        std::cout << "  Max LARS steps: " << max_steps << "\n";
    }

    std::cout << "\n";
}


/**
 * @brief Print simple timing breakdown (inline so header-only use is safe)
 */
inline void print_timing(const long data_ms,
                         const long prep_ms,
                         const long solver_ms) {

    std::cout << "Data:             " << data_ms << " ms\n";
    std::cout << "Standardization:  " << prep_ms << " ms\n";
    std::cout << "LARS:             " << solver_ms << " ms\n";
    std::cout << "Total:            " << (data_ms + prep_ms + solver_ms)
              << " ms\n\n";
}


/**
 * @brief Print a short report describing support recovery in the first steps
 *
 * Prints the true support and the first `steps` selected variables and
 * provides a simple diagnostic whether the recovery is perfect in those
 * steps.
 *
 * @param true_support Indices of the true non-zero coefficients
 * @param selected Sequence of selected indices (in selection order)
 * @param steps Number of initial selections to compare (default: 3)
 */
inline void print_support_recovery_report(
    const std::vector<size_t> &true_support,
    const std::vector<size_t> &selected, // Full or first k LARS actions
    std::size_t steps = 3) {

    std::cout << "True support:     {";
    for (size_t i = 0; i < true_support.size(); ++i)
        std::cout << true_support[i]
                  << (i + 1 < true_support.size() ? ", " : "");
    std::cout << "}\nSelected (first " << steps << "): {";
    for (size_t i = 0; i < steps && i < selected.size(); ++i)
        std::cout << selected[i]
                  << (i + 1 < steps && i + 1 < selected.size() ? ", " : "");
    std::cout << "}\n";

    // Diagnostic set comparison/feedback
    std::set<size_t> sel_set(selected.begin(),
                             selected.begin() +
                             std::min(steps, selected.size()));
    std::set<size_t> true_set(true_support.begin(), true_support.end());

    if (sel_set == true_set) {
        std::cout << "✓ Perfect support recovery in first "
                  << steps << " steps!\n";
    } else {
        std::cout << "Partial recovery. True variables found at positions: ";
        for (size_t i = 0; i < selected.size(); ++i)
            if (true_set.count(selected[i]))
                std::cout << (i + 1) << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}


/**
 * @brief Remove data files from disk. If `force` is true, silent;
 * else prompt user.
 */
inline void cleanup_files(const char *X_file,
                          const char *y_file,
                          bool force = false) {

    if (force) {
        std::remove(X_file);
        std::remove(y_file);
        std::cout << "✓ Files removed\n\n";
        return;
    }
    char resp;
    std::cout << "Remove data files? (y/n): ";
    std::cin >> resp;
    if (resp == 'y' || resp == 'Y') {
        std::remove(X_file);
        std::remove(y_file);
        std::cout << "✓ Files removed\n\n";
    } else {
        std::cout << "Files kept:\n  " << X_file << "\n  " << y_file << "\n\n";
    }
}


/**
 * @brief Check whether a file exists and can be opened for reading
 *
 * @param fname Path to file
 * @return true if file can be opened, false otherwise
 */
bool file_exists(const std::string &fname)
{
    std::ifstream file(fname);
    return file.good();
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


/**
 * @brief Round double value to numerical precision
 *
 * @param value Value to be rounded
 * @param tol Numerical precision
 *
 * @return Rounded value according to numerical precision
 */
inline double round_to(double value, double tol = 1e-12)
{
    return std::round(value / tol) * tol;
}

/**
 * @brief Print a small integer vector in square brackets
 *
 * Example output: [1, 2, -3]
 */
void print_member_vector(const std::vector<int> &v)
{
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        std::cout << v[i];
        if (i < v.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]";
}


/**
 * @brief Print a compact actions log
 *
 * Each step prints a line with all additions (+k) or drops (-k) performed.
 */
void print_actions_log(const std::vector<std::vector<int>> &actions)
{
    for (std::size_t step = 0; step < actions.size(); ++step)
    {
        std::cout << "Step " << std::setw(3) << step << ": ";
        for (int idx : actions[step]) {
            if (idx >= 0)
                std::cout << " +" << idx;
            else
                std::cout << " " << idx; // Negative is drop
        }
        std::cout << "\n";
    }
}


/**
 * @brief Print a small table comparing true and estimated coefficients
 *
 * Prints rows for the requested `indices` showing true value, estimate,
 * and the error in a compact tabular format.
 */
void print_coefficient_recovery_table(const std::vector<size_t> &indices,
                                      const Eigen::VectorXd &beta_true,
                                      const Eigen::VectorXd &beta_est) {
    std::cout << std::setw(8) << "   β[i]"
              << std::setw(14) << "True" << std::setw(14)
              << "Estimated" << std::setw(14)
              << "Error\n";
    for (auto idx : indices) {
        double err = beta_true[idx] - beta_est[idx];
        std::cout << std::setw(8) << ("β[" + std::to_string(idx) + "]")
                  << std::setw(14) << std::fixed
                  << std::setprecision(4) << beta_true[idx]
                  << std::setw(14) << std::fixed << std::setprecision(4)
                  << beta_est[idx] << std::setw(14) << std::scientific
                  << std::setprecision(4) << err << "\n";
    }
}

} /* End of namespace utils_eval */

#endif /* End of UTILS_EVAL_HPP */
