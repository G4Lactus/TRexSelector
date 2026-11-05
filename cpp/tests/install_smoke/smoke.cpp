// A downstream consumer of the INSTALLED SDK: one header from each of the four
// subtrees (the include root is namespaced, the target carries it), one
// selection end to end. Exit status is the check.
#include <Eigen/Dense>
#include <cstdio>
#include <vector>

#include <ml_methods/clustering/hierarchical/agglomerative/dendrogram_utils.hpp>
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>
#include <utils/checked_arith.hpp>
#include <utils/datageneration/utils_datagen.hpp>

int main() {
    using namespace trex::trex_selector_methods::ad::trex_core;
    using namespace trex::utils::datageneration::datagen;

    std::vector<std::size_t> support = {1, 2, 3};
    std::vector<double> coefs = {5.0, -3.0, 2.0};
    SyntheticData data(80, 20, support, coefs, 1.0, 42);
    auto X = data.getX();
    auto y = data.getY();
    Eigen::Map<Eigen::MatrixXd> X_map(X.data(), X.rows(), X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());

    TRexControlParameter ctrl;
    ctrl.K = 3;
    ctrl.max_dummy_multiplier = 2;
    TRexSelector trex(X_map, y_map, 0.1, ctrl, 42, false);
    const auto result = trex.select();

    // Structural check only: the run completes with the expected shapes.
    // (Whether this small design selects anything at tFDR 0.1 with K = 3 is a
    // statistical question, not an install question.)
    const bool ok = trex.getTStop() >= 1 &&
                    result.Phi_prime.size() == X.cols() &&
                    result.selected_var.size() == X.cols();
    const std::size_t bytes = trex::utils::checked_arith::checked_bytes(80, 20, sizeof(double));
    std::printf("install smoke: %s — T_stop %zu, selected %zu of %ld (buffer %zu bytes)\n",
                ok ? "consumer ran" : "SHAPE MISMATCH", trex.getTStop(),
                trex.getNumSelected(), static_cast<long>(X.cols()), bytes);
    return ok ? 0 : 1;
}
