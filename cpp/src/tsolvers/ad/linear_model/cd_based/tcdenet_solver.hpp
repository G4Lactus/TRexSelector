// =============================================================================
//  tcdenet_solver.hpp
// =============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_CD_BASED_TCDENET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_CD_BASED_TCDENET_SOLVER_HPP
// =============================================================================
/**
 * @file tcdenet_solver.hpp
 *
 * @brief Header file for the Terminating CCD Elastic Net (T-CDENET) solver
 * class, extending the T-CDLASSO solver.
 */
// =============================================================================

// std includes
#include <string>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>

// tsolvers includes
#include <tsolvers/ad/linear_model/cd_based/tcdlasso_solver.hpp>

// =============================================================================

// Embed into trex::tsolvers::ad::linear_model::cd_based namespace
namespace trex::tsolvers::ad::linear_model::cd_based {

// =============================================================================

/**
 * @brief Terminating CCD Elastic Net (T-CDENET) solver with dummy variable
 *        support for FDR control.
 *
 * @details
 *  Configures the matrix-passing T-CDLASSO core with the diagonal penalty K = I:
 *
 *      F(beta) = 1/2 ||y - X beta||^2 + lambda1 ||beta||_1
 *                + (lambda2/2) ||beta||_2^2
 *
 *  The dummy block receives kappa_dummy * I (default 1.0 = identical to the
 *  real block, preserving dummy/null exchangeability). The lambda1 path is
 *  traversed with the inherited crossing-localization machinery; the
 *  coordinate update reduces to the glmnet-style naive elastic net update
 *  at identical cost to the plain lasso.
 *
 *  Coefficients are reported as raw minimizers of the objective above (no
 *  Zou-Hastie (1 + lambda2) rescaling; apply it downstream if the corrected
 *  elastic net convention is required).
 *
 *  References:
 *  -----------
 *   H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R. Stat. Soc. Ser.
 *     B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
 *   J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate Optimization," Ann. Appl.
 *     Stat., vol. 1, no. 2, pp. 302-332, 2007, doi: 10.1214/07-AOAS131.
 *   J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear Models via
 *     Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010, doi: 10.18637/jss.v033.i01.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
class TCDENET_Solver: public TCDLASSO_Solver {

protected:

    // ==================================================================================
    // Constructor protected for inheritance
    // ==================================================================================
    /**
     * @brief Protected constructor for inheritance by derived classes.
     *
     * @param solver_type Algorithm variant as SolverTypeCdBased enum.
     */
    explicit TCDENET_Solver(SolverTypeCdBased solver_type) :
                            TCDLASSO_Solver(solver_type) {}

    /**
     * @brief Protected data constructor for inheritance (algorithm type
     * passthrough, mirroring the TLARS_Solver pattern). Configures the
     * diagonal elastic-net penalty; descendants may override the penalty
     * configuration afterwards.
     */
    TCDENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  const SolverOptions& options,
                  SolverTypeCdBased solver_type)
        : TCDLASSO_Solver(X, D, y, options, solver_type) {
        lambda2_ = options.lambda2;
        kappa_dummy_ = options.kappa_dummy;
        penalty_kind_ = PenaltyKind::DIAGONAL;
        penalty_dvec_ = Eigen::VectorXd::Ones(
            static_cast<Eigen::Index>(p_original_));
    }

public:
    // ==========================================================================
    // Constructors/Destructors
    // ==========================================================================

    /**
     * @brief Construct a new T-CDENET solver with data and configuration.
     *
     * @param X Predictor matrix (n x p).
     * @param D Dummy matrix (n x num_dummies).
     * @param y Response vector (size n).
     * @param lambda2 Quadratic (ridge) penalty weight (>= 0).
     * @param normalize If true, columns are scaled to unit L2-norm. Default true.
     * @param intercept If true, variables are centered. Default true.
     * @param verbose If true, print detailed status and diagnostics. Default false.
     * @param scaling_mode Column-scaling convention (L2 or z-score). Default L2.
     * @param kappa_dummy Ridge weight of the dummy block. Default 1.0.
     */
    TCDENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  const SolverOptions& options = SolverOptions{})
        : TCDENET_Solver(X, D, y, options, SolverTypeCdBased::TCDENET) {}

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TCDENET_Solver() : TCDLASSO_Solver(SolverTypeCdBased::TCDENET) {}

    /** @brief Deleted copy constructor. */
    TCDENET_Solver(const TCDENET_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TCDENET_Solver& operator=(const TCDENET_Solver&) = delete;

    /** @brief Move constructor (defaulted). */
    TCDENET_Solver(TCDENET_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TCDENET_Solver& operator=(TCDENET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TCDENET_Solver() override = default;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-CDENET solver state except for input
     * matrices (the penalty configuration lives in the T-CDLASSO base state).
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TCDLASSO_Solver>(this)
        );
    }

    /**
     * @brief Save T-CDENET solver/model state to file (binary serialization).
     *
     * @param filename Output file path for checkpoint.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load T-CDENET solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TCDENET_Solver object connected to X and ready for
     *         warm-start.
     */
    static TCDENET_Solver load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D);

};

// =============================================================================
} // End of namespace trex::tsolvers::ad::linear_model::cd_based

// =============================================================================
// Cereal polymorphic registration
// =============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::ad::linear_model::cd_based::TCDENET_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_CD_BASED_TCDENET_SOLVER_HPP */
