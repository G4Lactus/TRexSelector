with open("../../../R/TRexSelector/src/rcpp_trex_wrappers.h", "r") as f:
    content = f.read()

da_class = """
#include <trex_selector_methods/trex_da/trex_da.hpp>
using namespace trex::trex_selector_methods::trex_da;

/**
 * @brief Rcpp wrapper class for TRexDASelector leveraging zero-copy references.
 */
class RTRexDASelector {
private:
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_;
    std::unique_ptr<TRexDASelector> selector_;

public:
    RTRexDASelector(
        Eigen::Map<Eigen::MatrixXd> X,
        Eigen::Map<Eigen::VectorXd> y,
        double tFDR,
        const TRexDAControlParameter& da_control,
        const TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y);

        selector_ = std::make_unique<TRexDASelector>(
            *X_map_, *y_map_, tFDR, da_control, trex_control, seed, verbose
        );
    }
    
    TRexDASelector* get() const { return selector_.get(); }
    const TRexDASelector::DASelectionResult& select() { 
        selector_->select(); 
        return selector_->getDAResult();
    }
};

#endif // TREX_RCPP_TREX_WRAPPERS_H
"""
content = content.replace("#endif // TREX_RCPP_TREX_WRAPPERS_H", da_class)
with open("../../../R/TRexSelector/src/rcpp_trex_wrappers.h", "w") as f:
    f.write(content)
