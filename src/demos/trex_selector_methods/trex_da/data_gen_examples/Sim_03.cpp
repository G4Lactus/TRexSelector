#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
// Include previous generate_simulation_data definition...

// Updated stub: The selector now takes the target FDR (alpha) as an argument
EvalMetrics evaluate_selector_with_alpha(const SimulationData& data, double target_fdr) {
    // TODO: Implement T-Rex DA calibration algorithm for the specific target_fdr
    // Return the empirical FDP and TPR for this dataset and alpha.
    if (target_fdr <= 0.0) {
        return {0.0, 0.0}; // Sanity boundary: select nothing if alpha=0
    }
    return {0.0, 0.0}; 
}

void run_figure5_simulation(int mc_reps = 955) {
    std::cout << "\n=== Running Figure 5 Scenarios (FDP & TPP vs Target FDR) ===\n";

    // Define the grids
    std::vector<double> rhos = {0.7, 0.8, 0.9};
    
    // alpha grid: 0, 0.05, 0.10, ..., 0.50
    std::vector<double> alphas;
    for (int i = 0; i <= 10; ++i) {
        alphas.push_back(i * 0.05);
    }

    // Base parameters for this specific scenario
    const int n = 150;
    const int p = 500;
    const int G = 5;   // Q in your notation
    const int g = 5;   // M in your notation
    const double snr = 2.0;

    for (double rho : rhos) {
        std::cout << "\n--- Evaluating for rho = " << rho << " ---\n";
        
        // Global accumulators for this rho
        std::vector<double> total_fdp(alphas.size(), 0.0);
        std::vector<double> total_tpr(alphas.size(), 0.0);

        #pragma omp parallel
        {
            // Thread-local accumulators to avoid false sharing and locks inside the loop
            std::vector<double> local_fdp(alphas.size(), 0.0);
            std::vector<double> local_tpr(alphas.size(), 0.0);

            #pragma omp for schedule(dynamic)
            for (int r = 0; r < mc_reps; ++r) {
                // Unique seed per replication and rho
                uint64_t iter_seed = 1337 + r * 1000 + static_cast<uint64_t>(rho * 100);
                
                // GENERATE ONCE: Huge performance save!
                SimulationData data = generate_simulation_data(n, p, G, g, rho, snr, iter_seed);
                
                // EVALUATE MULTIPLE TIMES
                for (size_t a = 0; a < alphas.size(); ++a) {
                    double alpha = alphas[a];
                    EvalMetrics metrics = evaluate_selector_with_alpha(data, alpha);
                    
                    local_fdp[a] += metrics.fdp;
                    local_tpr[a] += metrics.tpr;
                }
            }

            // Safely merge thread-local results into the global accumulators
            #pragma omp critical
            {
                for (size_t a = 0; a < alphas.size(); ++a) {
                    total_fdp[a] += local_fdp[a];
                    total_tpr[a] += local_tpr[a];
                }
            }
        }

        // Print results for this rho
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Alpha\t| Avg FDP\t| Avg TPR (Power)\n";
        std::cout << "-----------------------------------------\n";
        for (size_t a = 0; a < alphas.size(); ++a) {
            double avg_fdp = total_fdp[a] / mc_reps;
            double avg_tpr = total_tpr[a] / mc_reps;
            std::cout << alphas[a] << "\t| " 
                      << avg_fdp << "\t\t| " 
                      << avg_tpr << "\n";
        }
    }
}

int main() {
    run_figure5_simulation();
    return 0;
}