#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>

// Structure to hold the generated data
struct SimulationData {
    Eigen::MatrixXd X;    // Predictor matrix (n x p)
    Eigen::VectorXd y;    // Response vector (n x 1)
    Eigen::VectorXd beta; // True coefficient vector (p x 1)
};

/**
 * @brief Generates data for the Dependency-Aware T-Rex selector simulation.
 * 
 * @param n     Number of samples
 * @param p     Total number of candidate variables
 * @param G     Number of active groups (blocks containing a true active variable)
 * @param g     Size of each correlated group (block size)
 * @param rho   Correlation coefficient for the Toeplitz structure
 * @param snr   Signal-to-Noise Ratio
 * @param seed  Random seed for reproducibility
 * @return SimulationData containing X, y, and the true beta
 */
SimulationData generate_simulation_data(
    int n = 150, 
    int p = 500, 
    int G = 5, 
    int g = 5, 
    double rho = 0.7, 
    double snr = 2.0, 
    uint64_t seed = 42) 
{
    if (p % g != 0) {
        throw std::invalid_argument("Total variables 'p' must be divisible by group size 'g'.");
    }
    if (G * g > p) {
        throw std::invalid_argument("Number of active groups 'G' exceeds total available groups.");
    }

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    // 1. Construct the block-diagonal Toeplitz correlation matrix Sigma
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(p, p);
    int num_blocks = p / g;
    
    for (int b = 0; b < num_blocks; ++b) {
        int start_idx = b * g;
        for (int i = 0; i < g; ++i) {
            for (int j = 0; j < g; ++j) {
                Sigma(start_idx + i, start_idx + j) = std::pow(rho, std::abs(i - j));
            }
        }
    }

    // 2. Generate the predictor matrix X ~ N(0, Sigma)
    // We use Cholesky decomposition: Sigma = L * L^T -> X = Z * L^T
    Eigen::LLT<Eigen::MatrixXd> lltOfSigma(Sigma);
    if (lltOfSigma.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Correlation matrix is not positive definite!");
    }
    Eigen::MatrixXd L = lltOfSigma.matrixL();
    
    Eigen::MatrixXd Z(n, p);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            Z(i, j) = std_normal(rng);
        }
    }
    Eigen::MatrixXd X = Z * L.transpose();

    // 3. Construct the true sparse coefficient vector beta
    // The first G blocks get exactly one active variable (coefficient = 1)
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (int b = 0; b < G; ++b) {
        // Assign the first variable in the current block as the active variable
        beta(b * g) = 1.0; 
    }

    // 4. Generate the response vector y = X * beta + epsilon
    Eigen::VectorXd X_beta = X * beta;
    
    // Calculate empirical variance of X_beta to accurately scale the noise for the target SNR
    double mean_X_beta = X_beta.mean();
    double var_X_beta = (X_beta.array() - mean_X_beta).square().sum() / (n - 1.0);
    
    // Calculate noise standard deviation: sigma = sqrt(Var(X*beta) / SNR)
    double noise_sigma = std::sqrt(var_X_beta / snr);
    
    Eigen::VectorXd epsilon(n);
    for (int i = 0; i < n; ++i) {
        epsilon(i) = std_normal(rng) * noise_sigma;
    }
    
    Eigen::VectorXd y = X_beta + epsilon;

    return {X, y, beta};
}

int main() {
    try {
        // Base setting from the paper
        SimulationData data = generate_simulation_data(150, 500, 5, 5, 0.7, 2.0);
        
        std::cout << "Successfully generated data!\n";
        std::cout << "X dimensions: " << data.X.rows() << " x " << data.X.cols() << "\n";
        std::cout << "y dimensions: " << data.y.rows() << " x " << data.y.cols() << "\n";
        std::cout << "Active variables in beta: " << data.beta.nonZeros() << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}