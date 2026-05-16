import re

with open("src/demos/trex_selector_methods/trex_gvs/sim_trex_gvs_13_hapgen.cpp", "r") as f:
    text = f.read()

# For Hapgen, we need to inject the sweep array.
# The utilities expect a vector of pairs, like in other files.

part1_start = text.find('static void sim_hapgen_part1')
part1_end = text.find('static void sim_hapgen_part2')
part2_end = text.find('static void sim_hapgen_part3')
part3_end = text.find('int main()')

setup_code = """    struct MethodConfig {
        std::string name;
        GVSType type;
        LambdaSelectionMethod method;
    };

    std::vector<MethodConfig> methods = {
        {"EN [GCV]", GVSType::EN, LambdaSelectionMethod::GCV},
        {"IEN [GCV]", GVSType::IEN, LambdaSelectionMethod::GCV},
        {"EN [1SE]", GVSType::EN, LambdaSelectionMethod::CV_1SE},
        {"IEN [1SE]", GVSType::IEN, LambdaSelectionMethod::CV_1SE},
        {"EN [MIN]", GVSType::EN, LambdaSelectionMethod::CV_MIN},
        {"IEN [MIN]", GVSType::IEN, LambdaSelectionMethod::CV_MIN},
        {"EN [COND]", GVSType::EN, LambdaSelectionMethod::COND_NUM},
        {"IEN [COND]", GVSType::IEN, LambdaSelectionMethod::COND_NUM}
    };
"""

def process_part(part_text, part_num):
    if part_num == 1:
        # snr sweep
        # replace the en_snr / ien_snr stuff
        repl = """    auto trex_ctrl    = make_gvs_trex_control(cfg.K);

""" + setup_code + """
    cdiag::print_section_header(
        "Part 1: SNR Sweep  |  " + tag +
        "\\nn=" + std::to_string(cfg.n) + "  p=500" +
        "  s=130  MC=" + std::to_string(cfg.num_MC) +
        "  tFDR=" + std::to_string(cfg.tFDR).substr(0, 3) +
        "  rho_scale=1.0 (fixed)");

    std::vector<std::pair<std::string, std::vector<GVSGridPointResult>>> sweep_results;
    for (const auto& m : methods) {
        auto gvs_ctrl = make_hapgen_gvs_ctrl(m.type, cfg.corr_max);
        gvs_ctrl.lambda2_method = m.method;

        std::vector<GVSGridPointResult> res;
        res.reserve(snr_grid.size());
        for (double snr : snr_grid) {
            auto dgp = [&cfg, &L, snr](double /*param*/, unsigned seed) {
                return make_hapgen_dgp(cfg.n, snr, L, seed);
            };
            res.push_back(run_gvs_mc_trials(
                dgp, snr, cfg.num_MC, m.name,
                cfg.tFDR, gvs_ctrl, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed)));
        }
        sweep_results.push_back({m.name, res});
    }

    print_mc_snr_table(tag, snr_grid, sweep_results, cfg);
}"""
        return re.sub(r'    auto trex_ctrl[\s\S]*?print_mc_snr_table[\s\S]*?\}', repl, part_text)
    
    elif part_num == 2:
        repl = """    auto trex_ctrl    = make_gvs_trex_control(cfg.K);

""" + setup_code + """
    cdiag::print_section_header(
        "Part 2: rho_scale Sweep  |  " + tag +
        "  (SNR=" + std::to_string(cfg.snr).substr(0, 3) + ")");

    std::vector<std::pair<std::string, std::vector<GVSGridPointResult>>> sweep_results;
    for (const auto& m : methods) {
        auto gvs_ctrl = make_hapgen_gvs_ctrl(m.type, cfg.corr_max);
        gvs_ctrl.lambda2_method = m.method;

        std::vector<GVSGridPointResult> res;
        res.reserve(rs_grid.size());
        for (double rs : rs_grid) {
            const Eigen::MatrixXd L = make_hapgen_cholesky(rs);
            auto dgp = [&cfg, &L](double /*param*/, unsigned seed) {
                return make_hapgen_dgp(cfg.n, cfg.snr, L, seed);
            };
            res.push_back(run_gvs_mc_trials(
                dgp, rs, cfg.num_MC, m.name,
                cfg.tFDR, gvs_ctrl, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed)));
        }
        sweep_results.push_back({m.name, res});
    }

    print_mc_param_sweep_table(tag, "rho_sc", rs_grid, sweep_results, cfg);
}"""
        return re.sub(r'    auto trex_ctrl[\s\S]*?print_mc_param_sweep_table[\s\S]*?\}', repl, part_text)

    elif part_num == 3:
        # Part 3 is 2D, let's look at the original structure. We can reconstruct it simply.
        repl = """    auto trex_ctrl    = make_gvs_trex_control(cfg.K);

""" + setup_code + """
    cdiag::print_section_header("Part 3: 2-D SNR x rho_scale  |  " + tag);

    std::vector<std::pair<std::string, std::vector<std::vector<GVSGridPointResult>>>> sweep_results_2d;
    for (const auto& m : methods) {
        auto gvs_ctrl = make_hapgen_gvs_ctrl(m.type, cfg.corr_max);
        gvs_ctrl.lambda2_method = m.method;

        std::vector<std::vector<GVSGridPointResult>> matrix(n_snr, std::vector<GVSGridPointResult>(n_rs));

        for (std::size_t i_rs = 0; i_rs < n_rs; ++i_rs) {
            const double rs = rs_grid_2d[i_rs];
            const Eigen::MatrixXd L = make_hapgen_cholesky(rs);

            for (std::size_t i_snr = 0; i_snr < n_snr; ++i_snr) {
                const double snr = snr_grid_2d[i_snr];
                const std::size_t cell_idx = i_rs * n_snr + i_snr + 1;
                const std::size_t total_cells = n_rs * n_snr;

                std::cout << "  [" << m.name << "] [" << cell_idx << "/" << total_cells << "]"
                          << "  SNR=" << std::fixed << std::setprecision(2) << snr
                          << "  rho_scale=" << rs
                          << "  running " << cfg.num_MC << " MC trials...\\n"
                          << std::flush;

                auto dgp = [&cfg, &L, snr](double /*param*/, unsigned seed) {
                    return make_hapgen_dgp(cfg.n, snr, L, seed);
                };

                matrix[i_snr][i_rs] = run_gvs_mc_trials(
                    dgp, snr, cfg.num_MC, m.name,
                    cfg.tFDR, gvs_ctrl, trex_ctrl,
                    static_cast<unsigned>(cfg.base_seed));
            }
        }
        sweep_results_2d.push_back({m.name, matrix});
    }

    std::vector<std::string> snr_labels, rho_labels;
    for (double s : snr_grid_2d) snr_labels.push_back("snr=" + std::to_string(s).substr(0, 4));
    for (double r : rs_grid_2d)  rho_labels.push_back("rs=" + std::to_string(r).substr(0, 4));

    for (const auto& res : sweep_results_2d) {
        print_mc_matrix("mean_TPP " + res.first, snr_labels, rho_labels, res.second, 1);
        print_mc_matrix("mean_FDP " + res.first, snr_labels, rho_labels, res.second, 0);
        print_mc_matrix("mean_L2  " + res.first, snr_labels, rho_labels, res.second, 2);
    }
}"""
        return re.sub(r'    auto trex_ctrl[\s\S]*?print_mc_matrix[\s\S]*?\}', repl, part_text)

p1 = process_part(text[part1_start:part1_end], 1)
p2 = process_part(text[part1_end:part2_end], 2)
p3 = process_part(text[part2_end:part3_end], 3)

new_content = text[:part1_start] + p1 + "\n\n// ==============================================================================\n//  Part 2 \n// ==============================================================================\n\n" + p2 + "\n\n// ==============================================================================\n//  Part 3\n// ==============================================================================\n\n" + p3 + "\n\n// ==============================================================================\n//  main\n// ==============================================================================\n\n" + text[part3_end:]

with open("src/demos/trex_selector_methods/trex_gvs/sim_trex_gvs_13_hapgen.cpp", "w") as f:
    f.write(new_content)
    
print("Hapgen patched.")
