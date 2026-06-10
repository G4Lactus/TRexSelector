/// @file gmp_demo.cpp
/// @brief Demo / smoke-test for Norm-Corrective Generalized Matching Pursuit.
///
/// Three experiments:
///   1. Least-squares signal recovery with standard basis atoms (L1 ball).
///   2. Same with an overcomplete random dictionary.
///   3. Quadratic objective  f(x) = 0.5 x^T H x − c^T x  with random SPD H.
///
/// Build:
///   g++ -O2 -std=c++20 -I$(brew --prefix eigen)/include/eigen3 gmp_demo.cpp -o gmp_demo
/// or via CMake (see CMakeLists.txt).

#include "norm_corrective_gmp.hpp"
#include <iostream>
#include <iomanip>
#include <random>

static void print_header(const char* title) {
    std::cout << "\n══════════════════════════════════════════════\n"
              << "  " << title
              << "\n══════════════════════════════════════════════\n";
}

// ── Experiment 1: Sparse signal recovery, standard basis ────────────
static void experiment_ls_standard_basis()
{
    print_header("Exp 1: LS recovery, A = ±e_i  (d=20, 3 nonzeros)");

    const int d = 20;
    // Dictionary = identity (columns are standard basis vectors).
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(d, d);

    // Target signal: sparse, 3 nonzeros.
    Eigen::VectorXd y = Eigen::VectorXd::Zero(d);
    y(2)  =  3.0;
    y(7)  = -1.5;
    y(15) =  2.0;

    gmp::LeastSquaresObjective obj(y);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d);

    for (auto v : {gmp::Variant::LineSearch, gmp::Variant::FullyCorrective}) {
        const char* name = (v == gmp::Variant::LineSearch)
                           ? "Variant 0 (LineSearch)"
                           : "Variant 1 (FullyCorrective)";

        gmp::NormCorrectiveGMP solver(obj, A, v, /*max_iter=*/200, /*tol=*/1e-14);
        auto res = solver.solve(x0);

        const double err = (res.x - y).norm();
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  " << name << "\n"
                  << "    iterations : " << res.iterations << "\n"
                  << "    f(x_T)     : " << res.obj_values.back() << "\n"
                  << "    ||x − y||  : " << err << "\n"
                  << "    active atoms: " << res.active_atoms.cols() << "\n";
    }
}

// ── Experiment 2: Overcomplete dictionary ───────────────────────────
static void experiment_ls_overcomplete()
{
    print_header("Exp 2: LS recovery, overcomplete random dict  (d=10, k=50)");

    const int d = 10;
    const int k = 50;

    std::mt19937 rng(42);
    std::normal_distribution<double> randn(0.0, 1.0);

    // Random dictionary, normalise columns.
    Eigen::MatrixXd A(d, k);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < d; ++i)
            A(i, j) = randn(rng);
        A.col(j).normalize();
    }

    // Target: a random linear combination of 4 atoms.
    Eigen::VectorXd y = 2.0 * A.col(3) - 1.0 * A.col(17)
                       + 0.5 * A.col(31) + 3.0 * A.col(49);

    gmp::LeastSquaresObjective obj(y);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d);

    for (auto v : {gmp::Variant::LineSearch, gmp::Variant::FullyCorrective}) {
        const char* name = (v == gmp::Variant::LineSearch)
                           ? "Variant 0 (LineSearch)"
                           : "Variant 1 (FullyCorrective)";

        gmp::NormCorrectiveGMP solver(obj, A, v, /*max_iter=*/300, /*tol=*/1e-14);
        auto res = solver.solve(x0);

        const double err = (res.x - y).norm();
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  " << name << "\n"
                  << "    iterations : " << res.iterations << "\n"
                  << "    f(x_T)     : " << res.obj_values.back() << "\n"
                  << "    ||x − y||  : " << err << "\n"
                  << "    active atoms: " << res.active_atoms.cols() << "\n";
    }
}

// ── Experiment 3: General quadratic objective ───────────────────────
static void experiment_quadratic()
{
    print_header("Exp 3: Quadratic  f(x) = 0.5 x^T H x − c^T x  (d=15)");

    const int d = 15;

    std::mt19937 rng(123);
    std::normal_distribution<double> randn(0.0, 1.0);

    // Random SPD matrix H = M^T M + I.
    Eigen::MatrixXd M(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            M(i, j) = randn(rng);
    Eigen::MatrixXd H = M.transpose() * M + Eigen::MatrixXd::Identity(d, d);

    Eigen::VectorXd c(d);
    for (int i = 0; i < d; ++i)
        c(i) = randn(rng);

    // True solution:  x* = H^{-1} c.
    Eigen::VectorXd x_star = H.ldlt().solve(c);
    double f_star = 0.5 * x_star.dot(H * x_star) - c.dot(x_star);

    // Dictionary = identity.
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(d, d);

    gmp::QuadraticObjective obj(H, c);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d);

    for (auto v : {gmp::Variant::LineSearch, gmp::Variant::FullyCorrective}) {
        const char* name = (v == gmp::Variant::LineSearch)
                           ? "Variant 0 (LineSearch)"
                           : "Variant 1 (FullyCorrective)";

        gmp::NormCorrectiveGMP solver(obj, A, v, /*max_iter=*/500, /*tol=*/1e-14);
        auto res = solver.solve(x0);

        const double gap = res.obj_values.back() - f_star;
        const double err = (res.x - x_star).norm();
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  " << name << "\n"
                  << "    iterations     : " << res.iterations << "\n"
                  << "    f(x_T)         : " << res.obj_values.back() << "\n"
                  << "    f(x*) (exact)  : " << f_star << "\n"
                  << "    f(x_T) − f(x*) : " << gap << "\n"
                  << "    ||x − x*||     : " << err << "\n";
    }
}

int main()
{
    experiment_ls_standard_basis();
    experiment_ls_overcomplete();
    experiment_quadratic();
    return 0;
}
