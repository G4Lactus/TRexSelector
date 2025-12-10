

template <class MatType>
void gradient_pursuit(
    const MatType& Phi,
    const Eigen::VectorXd& y,
    std::size_t max_iters,
    double tol,
    std::vector<int>& support_out,
    Eigen::VectorXd& x_out)
{
    using Scalar = typename MatType::Scalar;
    const Eigen::Index n = Phi.rows();
    const Eigen::Index p = Phi.cols();

    x_out.setZero(p);
    Eigen::VectorXd r = y;  // Initial residual is y
    Eigen::VectorXd g(p);   // gradient
    Eigen::VectorXd d(p);   // direction
    Eigen::VectorXd c(n);   // Phi * d
    std::vector<char> in_support(p, 0);

    support_out.clear();
    support_out.reserve(max_iters);

    for (std::size_t iter = 0; iter < max_iters; ++iter) {

        // 1. Compute gradient: g = Phi^T * r
        g.noalias() = Phi.transpose() * r;

        // 2. Find index of maximum absolute gradient (max |g_h|) not in support
        double abs_max = 0.0;
        int j_star = -1;
        for (Eigen::Index j = 0; j < p; ++j) {
            if (in_support[j]) { continue; }
            double val = std::abs(g[j]);
            if (val > abs_max) {
                abs_max = val;
                j_star = static_cast<int>(j);
            }
        }

        if (j_star < 0 || abs_max == 0.0) {
            break;  // Converged
        }

        in_support[j_star] = 1;
        support_out.push_back(j_star);

        // 3. Form direction vector d = g on support, 0 elsewhere
        d.setZero();
        for (int idx: support_out) {
            d[idx] = g[idx];
        }

        // 4. Compute c = Phi * d
        c.noalias() = Phi * d;

        // 5. Compute step size alpha = (r^T * c) / (c^T * c)
        double denom = c.squaredNorm();
        if (denom <= std::numeric_limits<double>::epsilon()) {
            break;  // Avoid division by zero
        }
        double alpha = r.dot(c) / denom;

        // 6. Update estimate x and residual r
        x_out += alpha * d;
        r -= alpha * c;

        // 7. Check convergence
        if (r.norm() < tol) {
            break;
        }
    }
}
