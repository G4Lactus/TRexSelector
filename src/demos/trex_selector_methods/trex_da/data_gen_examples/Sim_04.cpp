#include <Eigen/Dense>
#include <random>
#include <cmath>

struct SimulationData {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::VectorXd beta;
};

SimulationData generate_heavy_tailed_data(
    int n = 150, int p = 500, int G = 5, int g = 5, double rho = 0.8, double snr = 2.0, 
    bool heavy_X = false, bool heavy_noise = false, uint64_t seed = 42) 
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> std_normal(0.0, 1.0);
    
    // Degrees of freedom for the heavy-tailed scenarios
    const double df = 3.0;
    std::chi_squared_distribution<double> chi2_dist(df);
    std::student_t_distribution<double> t_dist(df);

    // 1. Construct Sigma and Cholesky decomposition
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(p, p);
    for (int b = 0; b < p / g; ++b) {
        int start = b * g;
        for (int i = 0; i < g; ++i) {
            for (int j = 0; j < g; ++j) {
                Sigma(start + i, start + j) = std::pow(rho, std::abs(i - j));
            }
        }
    }
    Eigen::MatrixXd L = Eigen::LLT<Eigen::MatrixXd>(Sigma).matrixL();

    // 2. Generate Predictor Matrix X
    Eigen::MatrixXd Z(n, p);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) Z(i, j) = std_normal(rng);
    }

    Eigen::MatrixXd X(n, p);
    if (heavy_X) {
        for (int i = 0; i < n; ++i) {
            double u = chi2_dist(rng);
            double scale = std::sqrt(df / u);
            // Scale the row and multiply by L transpose
            X.row(i) = (Z.row(i) * scale) * L.transpose();
        }
    } else {
        X = Z * L.transpose();
    }

    // 3. Construct true beta
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (int b = 0; b < G; ++b) beta(b * g) = 1.0;

    // 4. Generate response y with target SNR
    Eigen::VectorXd X_beta = X * beta;
    double var_X_beta = (X_beta.array() - X_beta.mean()).square().sum() / (n - 1.0);
    double target_noise_var = var_X_beta / snr;

    Eigen::VectorXd epsilon(n);
    if (heavy_noise) {
        // Scale standard t-distribution to match target variance
        double t_inherent_var = df / (df - 2.0); // = 3.0 for df=3
        double scale_factor = std::sqrt(target_noise_var / t_inherent_var);
        for (int i = 0; i < n; ++i) {
            epsilon(i) = t_dist(rng) * scale_factor;
        }
    } else {
        double scale_factor = std::sqrt(target_noise_var);
        for (int i = 0; i < n; ++i) {
            epsilon(i) = std_normal(rng) * scale_factor;
        }
    }

    return {X, X_beta + epsilon, beta};
}