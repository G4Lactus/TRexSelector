#include <vector>
#include <limits>

#include <Eigen/Dense>


void conjugate_gradient_pursuit(
    Eigen::Ref<const Eigen::MatrixXd> Phi,
    Eigen::Ref<const Eigen::VectorXd> y,
    std::size_t max_iters,
    double tol,
    std::vector<int>& support_out,
    Eigen::Ref<Eigen::VectorXd> x_out
    )
{
    using Eigen::Index;
    const Index n = Phi.rows();
    const Index p = Phi.cols();

    x_out.setZero();
    Eigen::VectorXd r = y;  // residual
    Eigen::VectorXd g(p), g_prev(p); // gradients
    Eigen::VectorXd d(p), d_prev(p);  // search directions
    Eigen::VectorXd c(n);             // Phi * d

    std::vector<char> in_support(p, 0); // support indicator
    support_out.clear();
    support_out.reserve(max_iters);

    bool first_iter = true;

    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        // 1. Compute gradient g = Phi^T * r
        g = Phi.transpose() * r;

        // 2. greedy selection of new index
        double max_abs = 0.0;
        int j_star = -1;
        for (Index j = 0; j < p; ++j) {
            if (!in_support[j]) { continue; }
            double abs_gj = std::abs(g[j]);
            if (abs_gj > max_abs) {
                max_abs = abs_gj;
                j_star = static_cast<int>(j);
            }
        }

        if (j_star < 0 || max_abs == 0.0) {
            break;  // converged
        }
        in_support[j_star] = 1;
        support_out.push_back(j_star);

        // 3. Update search direction with conjugate gradient formula in direction d on support
        d.setZero();
        double beta = 0.0;
        if (!first_iter) {
            // compute beta using gradient energy on previous support
            double numerator = 0.0, denominator = 0.0;
            for (int idx : support_out) {
                // previous support is subset
                numerator += g[idx] * g[idx];
                denominator += g_prev[idx] * g_prev[idx];
            }
            if (denominator > 0.0) {
                beta = numerator / denominator;
            }
        }
        for (int idx : support_out) {
            double prev_dir = first_iter ? 0.0 : d_prev[idx];
            d[idx] = g[idx] + beta * prev_dir;
        }


        // 4. Compute step size: c = Phi * d
        c.noalias() = Phi * d;

        double denom = c.squaredNorm();
        if (denom <= std::numeric_limits<double>::epsilon()) {
            break;  // cannot make progress
        }
        double alpha = r.dot(c) / denom;

        // 5. Update coefficients and residual
        x_out.noalias() += alpha * d;
        r.noalias() -= alpha * c;

        // Check convergence
        if (r.norm() < tol) {
            break;
        }

        g_prev = g;
        d_prev = d;
        first_iter = first_iter ? false : true;
    }

}
