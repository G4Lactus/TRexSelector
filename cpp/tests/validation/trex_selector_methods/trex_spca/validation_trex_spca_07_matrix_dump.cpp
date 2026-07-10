// ==============================================================================
// validation_trex_spca_07_matrix_dump.cpp
// ==============================================================================
/**
 * @file validation_trex_spca_07_matrix_dump.cpp
 *
 * @brief Dump the Neo GVS selector's internal calibration matrices per trial.
 *
 * @details Split-pipeline probe for the residual C++-vs-CRAN-R FDR gap on the
 *  SPCA benchmark: runs TRexGVSSelector (EN/TENET, R's lambda_2) on the
 *  rdump10 X/PC1 data — exactly like validation_trex_spca_04 — and dumps, per
 *  trial, T_stop, the voting grid, Phi_mat (T x p), FDP_hat_mat (T x v) and
 *  the selected set. An R script can then replay the CRAN reference
 *  calibration (Phi_prime_fun / fdp_hat / select_var_fun) on these matrices
 *  to decide whether the gap arises in the vote matrices themselves or in the
 *  calibration/selection code.
 *
 *  Flags: --dir <rdump dir> (default: the validation trex_spca rdump10),
 *         --n <trials> (default 100), --seed <base> (default 9000),
 *         --out <output csv path> (default <dir>/neo_matrix_dump.csv).
 */
// ==============================================================================

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>

namespace tg = trex::trex_selector_methods::trex_gvs;
namespace sd = trex::trex_selector_methods::utils::solver_dispatch;

namespace {

Eigen::MatrixXd load_csv_matrix(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        rows.push_back(std::move(row));
    }
    Eigen::MatrixXd M(static_cast<Eigen::Index>(rows.size()),
                      static_cast<Eigen::Index>(rows[0].size()));
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            M(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return M;
}

// r_lambda2.csv: "mc,lambda2" with header. r_pc1.csv: "mc,v1,...,vn" no header.
std::vector<double> load_lambda2(const std::string& path, int n_trials) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line); // header
    std::vector<double> lam(static_cast<std::size_t>(n_trials), -1.0);
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string a, b;
        std::getline(ss, a, ','); std::getline(ss, b, ',');
        int mc = std::stoi(a);
        if (mc >= 0 && mc < n_trials) lam[static_cast<std::size_t>(mc)] = std::stod(b);
    }
    return lam;
}

} // namespace

int main(int argc, char** argv) {
    // R dump baked into the library beside this program (see the co-located
    // demo_trex_spca_02.R generator run with RDUMP_DIR=rdump10); TREX_SPCA_RDUMP10_DIR
    // is injected by CMake to that source folder. Override at runtime with --dir <path>.
#ifdef TREX_SPCA_RDUMP10_DIR
    std::string dir = TREX_SPCA_RDUMP10_DIR;
#else
    std::string dir =
        "trex_selector_methods/trex_spca/rdump10";
#endif
    int n_trials = 100;
    int seed_base = 9000;
    std::string out_path;

    for (int a = 1; a < argc - 1; ++a) {
        if (!std::strcmp(argv[a], "--dir"))  dir = argv[a + 1];
        if (!std::strcmp(argv[a], "--n"))    n_trials = std::stoi(argv[a + 1]);
        if (!std::strcmp(argv[a], "--seed")) seed_base = std::stoi(argv[a + 1]);
        if (!std::strcmp(argv[a], "--out"))  out_path = argv[a + 1];
    }
    if (out_path.empty()) out_path = dir + "/neo_matrix_dump.csv";

    const auto lam = load_lambda2(dir + "/r_lambda2.csv", n_trials);
    const Eigen::MatrixXd pc1 = load_csv_matrix(dir + "/r_pc1.csv");

    std::ofstream out(out_path);
    out.precision(12);
    // long format: mc,record,row,values...
    //   record in {meta, grid, phi, fdp, sel}
    for (int mc = 0; mc < n_trials; ++mc) {
        Eigen::MatrixXd X = load_csv_matrix(dir + "/X_" + std::to_string(mc) + ".csv");
        // y = PC1 scores row for this mc (col 0 = mc id)
        Eigen::VectorXd y;
        for (Eigen::Index r = 0; r < pc1.rows(); ++r) {
            if (static_cast<int>(pc1(r, 0)) == mc) {
                y = pc1.row(r).tail(pc1.cols() - 1).transpose();
                break;
            }
        }

        tg::TRexGVSControlParameter ctrl;
        ctrl.gvs_type              = tg::GVSType::EN;
        const bool use_aug = std::getenv("USE_AUG") != nullptr;
        ctrl.en_solver             = use_aug ? tg::ENSolverType::TENET_AUG
                                             : tg::ENSolverType::TENET;
        ctrl.lambda_2              = lam[static_cast<std::size_t>(mc)];
        ctrl.trex_ctrl.solver_type = use_aug ? sd::SolverTypeForTRex::TENET_AUG
                                             : sd::SolverTypeForTRex::TENET;

        Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());
        tg::TRexGVSSelector gvs(X_map, y_map, 0.10, ctrl, seed_base + mc, false);
        const auto res = gvs.select();

        out << mc << ",meta,0," << res.T_stop << "," << res.num_dummies << ","
            << res.v_thresh << "\n";
        out << mc << ",grid,0";
        for (Eigen::Index i = 0; i < res.voting_grid.size(); ++i)
            out << "," << res.voting_grid(i);
        out << "\n";
        for (Eigen::Index t = 0; t < res.Phi_mat.rows(); ++t) {
            out << mc << ",phi," << t;
            for (Eigen::Index j = 0; j < res.Phi_mat.cols(); ++j)
                out << "," << res.Phi_mat(t, j);
            out << "\n";
        }
        for (Eigen::Index t = 0; t < res.FDP_hat_mat.rows(); ++t) {
            out << mc << ",fdp," << t;
            for (Eigen::Index j = 0; j < res.FDP_hat_mat.cols(); ++j)
                out << "," << res.FDP_hat_mat(t, j);
            out << "\n";
        }
        out << mc << ",sel,0";
        for (Eigen::Index j = 0; j < res.selected_var.size(); ++j)
            if (res.selected_var(j) != 0) out << "," << j;
        out << "\n";

        if ((mc + 1) % 25 == 0)
            std::cout << "  ... " << (mc + 1) << " / " << n_trials << "\n";
    }
    std::cout << "[Info] dump written to " << out_path << "\n";
    return 0;
}
