#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
// Include the previously defined generate_simulation_data and SimulationData

// Placeholder structure for the output of the variable selection method (e.g., T-Rex)
struct EvalMetrics {
    double fdp;
    double tpr;
};

// Stub for the actual variable selection algorithm 
EvalMetrics evaluate_selector(const SimulationData& data) {
    // TODO: Implement LARS/Lasso and the T-Rex DA selector here
    // Compare selected variables with data.beta to compute False Discovery Proportion 
    // and True Positive Proportion.
    return {0.0, 0.0}; 
}

/**
 * @brief High-order function to run Monte Carlo simulations over a parameter grid.
 * 
 * @tparam ParamType Data type of the grid parameter (int or double)
 * @tparam ParamSetter Lambda function type to inject the grid parameter into the base settings
 */
template <typename ParamType, typename ParamSetter>
void run_monte_carlo_grid(
    const std::string& grid_name,
    const std::vector<ParamType>& grid_values,
    ParamSetter set_param,
    int mc_reps = 955) 
{
    std::cout << "\n=== Running Scenario: " << grid_name << " ===\n";
    
    for (const auto& val : grid_values) {
        double total_fdp = 0.0;
        double total_tpr = 0.0;
        
        // OpenMP parallel region with reduction on the performance metrics
        #pragma omp parallel for reduction(+:total_fdp, total_tpr) schedule(dynamic)
        for (int r = 0; r < mc_reps; ++r) {
            // 1. Initialize base setting parameters as defined in the paper
            int n = 150, p = 500, G = 5, g = 5;
            double rho = 0.7, snr = 2.0;
            
            // 2. Override the specific parameter for this grid point
            set_param(val, n, p, G, g, rho, snr);
            
            // 3. Construct a strictly unique, deterministic seed per thread/iteration
            // We use a mix of the replication index and the parameter value to avoid seed collisions.
            uint64_t iter_seed = 42 + r * 10000 + static_cast<uint64_t>(val * 1000);
            
            // 4. Generate the data
            SimulationData data = generate_simulation_data(n, p, G, g, rho, snr, iter_seed);
            
            // 5. Evaluate the selector
            EvalMetrics metrics = evaluate_selector(data);
            
            total_fdp += metrics.fdp;
            total_tpr += metrics.tpr;
        }
        
        double avg_fdp = total_fdp / mc_reps;
        double avg_tpr = total_tpr / mc_reps;
        
        std::cout << grid_name << " = " << val 
                  << " | Average FDP: " << avg_fdp 
                  << " | Average TPR: " << avg_tpr << "\n";
    }
}

int main() {
    // Scenario 1: SNR grid
    std::vector<double> snr_grid = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0};
    run_monte_carlo_grid("SNR", snr_grid, [](double val, int& n, int& p, int& G, int& g, double& rho, double& snr) {
        snr = val;
    });

    // Scenario 2: Rho grid (0.0 to 0.9 in steps of 0.1)
    std::vector<double> rho_grid = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    run_monte_carlo_grid("Rho", rho_grid, [](double val, int& n, int& p, int& G, int& g, double& rho, double& snr) {
        rho = val;
    });

    // Scenario 3: Group size grid
    std::vector<int> g_grid = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    run_monte_carlo_grid("Group Size (g)", g_grid, [](int val, int& n, int& p, int& G, int& g, double& rho, double& snr) {
        g = val;
    });

    // Scenario 4: Number of groups grid
    std::vector<int> G_grid = {1, 3, 5, 10, 15, 20, 30};
    run_monte_carlo_grid("Number of Groups (G)", G_grid, [](int val, int& n, int& p, int& G, int& g, double& rho, double& snr) {
        G = val;
    });

    return 0;
}