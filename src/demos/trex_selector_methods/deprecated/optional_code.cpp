void demo_TRexSelector_PhiMetaAnalysis(int num_MC) {

    // 1. Setup
    const int n = 300;
    const int p = 1000;
    const int k_true = 10;
    const int max_T = 50;
    const double tFDR = 0.1;

    // SNR levels to test
    const std::vector<double> snr_list = {0.1, 0.5, 0.7, 1.0};

    // 2. Fixed Random Support (Same support indices for ALL SNRs)
    std::vector<std::size_t> support(k_true);
    std::random_device rng{};
    std::uniform_int_distribution<std::size_t> uniform_dist(0, p - 1);
    for (int i = 0; i < k_true; ++i) { support[i] = uniform_dist(rng); }

    // Fixed Coefficients
    std::vector<double> coefs(k_true, 1.0);

    // Header
    std::cout << "\n========================================\n";
    std::cout << " Meta-Stability Analysis (LARS vs Greedy)\n";
    std::cout << " Support Indices: ";
    for(auto s : support) { std::cout << s << " "; }
    std::cout << "\n========================================\n\n";

    // 3. Loop over SNRs
    for (double snr : snr_list) {

        // Reset Accumulators for this SNR
        Eigen::MatrixXd Phi_Sum_LARS = Eigen::MatrixXd::Zero(max_T, p);
        Eigen::MatrixXd Phi_Sum_Greedy = Eigen::MatrixXd::Zero(max_T, p);

        // Setup TRex Control (Standard)
        TRexControlParameter trex_ctrl;
        trex_ctrl.K = 20;
        trex_ctrl.tloop_stagnation_stop = true;
        trex_ctrl.tloop_max_stagnant_steps = 5;

        std::cout << ">>> Simulating SNR = " << snr << " (" << num_MC << " runs)...\n";

        // 4. Monte Carlo Loop
        for (int mc = 0; mc < num_MC; ++mc) {
            // Progress
            if (mc % 10 == 0 || mc == num_MC - 1) {
                std::cout << "Run " << (mc+1) << "/" << num_MC << "\r" << std::flush;
            }

            // Generate Data
            datagen::SyntheticData data(
                n, p, support, coefs, snr,
                24 + mc, // Vary seed for noise
                datagen::detail::PredictorConfig::Normal(),
                datagen::NoiseConfig::Gaussian()
            );

            Eigen::Map<Eigen::MatrixXd> X_map(data.getX().data(), n, p);
            Eigen::Map<Eigen::VectorXd> y_map(data.getY().data(), n);

            // -----------------------------------------------------
            // A. Run LARS
            // -----------------------------------------------------
            trex_ctrl.solver_type = SolverTypeForTRex::TLARS;
            TRexSelector trex_tlars(X_map, y_map, tFDR, trex_ctrl, -1, false);
            trex_tlars.select();

            // Accumulate
            const Eigen::MatrixXd& phi_L = trex_tlars.getPhiMat();
            long t_stop_L = phi_L.rows();

            for (int t = 0; t < max_T; ++t) {
                int source_row = std::min((long)t, t_stop_L - 1);
                if (source_row >= 0) {
                    Phi_Sum_LARS.row(t) += phi_L.row(source_row);
                }
            }

            // -----------------------------------------------------
            // B. Run Greedy (Stepwise)
            // -----------------------------------------------------
            trex_ctrl.solver_type = SolverTypeForTRex::TSTEPWISE;
            TRexSelector trex_greedy(X_map, y_map, tFDR, trex_ctrl, -1, false);
            trex_greedy.select();

            // Accumulate
            const Eigen::MatrixXd& phi_G = trex_greedy.getPhiMat();
            long t_stop_G = phi_G.rows();

            for (int t = 0; t < max_T; ++t) {
                int source_row = std::min((long)t, t_stop_G - 1);
                if (source_row >= 0) {
                    Phi_Sum_Greedy.row(t) += phi_G.row(source_row);
                }
            }
        } // End MC Loop

        // 5. Averaging
        Phi_Sum_LARS /= static_cast<double>(num_MC);
        Phi_Sum_Greedy /= static_cast<double>(num_MC);

        // 6. Print Results Table
        auto get_metrics = [&](const Eigen::MatrixXd& Phi_Sum, int t) {
            double sum_signal_phi = 0.0;
            for (size_t idx : support) sum_signal_phi += Phi_Sum(t, idx);
            double avg_signal = sum_signal_phi / static_cast<double>(support.size());

            double max_noise = 0.0;
            for (int j = 0; j < p; ++j) {
                bool is_signal = false;
                for(size_t s : support) if(s == (size_t)j) { is_signal = true; break; }
                if (!is_signal && Phi_Sum(t, j) > max_noise) max_noise = Phi_Sum(t, j);
            }
            return std::make_pair(avg_signal, max_noise);
        };

        std::cout << "\n\nResults (Avg Phi) at SNR=" << snr << ":\n";
        std::cout << "T | LARS Signal | LARS Noise | Greedy Signal | Greedy Noise\n";
        std::cout << "-----|-------------|------------|---------------|-------------\n";

        for (int t : {1, 5, 10, 15, 20, 30, 40}) {
            if (t >= max_T) break;
            auto [lars_sig, lars_noise] = get_metrics(Phi_Sum_LARS, t);
            auto [greedy_sig, greedy_noise] = get_metrics(Phi_Sum_Greedy, t);

            std::cout << std::setw(4) << t << " | "
                      << std::fixed << std::setprecision(3)
                      << lars_sig << "       | "
                      << lars_noise << "      | "
                      << greedy_sig << "         | "
                      << greedy_noise << "\n";
        }
        std::cout << "\n";

    } // End SNR Loop
}
