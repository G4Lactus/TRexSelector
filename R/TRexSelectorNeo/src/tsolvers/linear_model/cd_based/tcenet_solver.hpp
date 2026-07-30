// =============================================================================
//  tcenet_solver.hpp
// =============================================================================
#ifndef TSOLVERS_LINEAR_MODEL_CD_BASED_TCENET_SOLVER_HPP
#define TSOLVERS_LINEAR_MODEL_CD_BASED_TCENET_SOLVER_HPP
// =============================================================================
/**
 * @file tcenet_solver.hpp
 *
 * @brief Header file for the Terminating CCD Elastic Net (T-CENET) solver
 * class, extending the T-CCD solver.
 */
// =============================================================================

// std includes
#include <string>

// Cereal includes
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>

// tsolvers includes
#include <tsolvers/linear_model/cd_based/tccd_solver.hpp>

// =============================================================================

// Embed into trex::tsolvers::linear_model::cd_based namespace
namespace trex::tsolvers::linear_model::cd_based {

// =============================================================================

/**
 * @brief Terminating CCD Elastic Net (T-CENET) solver with dummy variable
 *        support for FDR control.
 *
 * @details
 *  Configures the matrix-passing T-CCD core with the diagonal penalty K = I:
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
 */
class TCENET_Solver: public TCCD_Solver {

protected:

    // ==================================================================================
    // Constructor protected for inheritance
    // ==================================================================================
    /**
     * @brief Protected constructor for inheritance by derived classes.
     *
     * @param solver_type Algorithm variant as SolverTypeCdBased enum.
     */
    explicit TCENET_Solver(SolverTypeCdBased solver_type) :
                            TCCD_Solver(solver_type) {}

    /**
     * @brief Protected data constructor for inheritance (algorithm type
     * passthrough, mirroring the TLARS_Solver pattern). Configures the
     * diagonal elastic-net penalty; descendants may override the penalty
     * configuration afterwards.
     */
    TCENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  double lambda2,
                  bool normalize,
                  bool intercept,
                  bool verbose,
                  ScalingMode scaling_mode,
                  double kappa_dummy,
                  SolverTypeCdBased solver_type)
        : TCCD_Solver(X, D, y, normalize, intercept, verbose,
                      solver_type, scaling_mode) {
        lambda2_ = lambda2;
        kappa_dummy_ = kappa_dummy;
        penalty_kind_ = PenaltyKind::DIAGONAL;
        penalty_dvec_ = Eigen::VectorXd::Ones(
            static_cast<Eigen::Index>(p_original_));
    }

public:
    // ==========================================================================
    // Constructors/Destructors
    // ==========================================================================

    /**
     * @brief Construct a new T-CENET solver with data and configuration.
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
    TCENET_Solver(Eigen::Map<Eigen::MatrixXd>& X,
                  Eigen::Map<Eigen::MatrixXd>& D,
                  Eigen::Map<Eigen::VectorXd>& y,
                  double lambda2,
                  bool normalize = true,
                  bool intercept = true,
                  bool verbose = false,
                  ScalingMode scaling_mode = ScalingMode::L2,
                  double kappa_dummy = 1.0)
        : TCENET_Solver(X, D, y, lambda2, normalize, intercept, verbose,
                        scaling_mode, kappa_dummy,
                        SolverTypeCdBased::TCENET) {}

    /**
     * @brief Default constructor for serialization or deferred initialization.
     */
    TCENET_Solver() : TCCD_Solver(SolverTypeCdBased::TCENET) {}

    /** @brief Deleted copy constructor. */
    TCENET_Solver(const TCENET_Solver&) = delete;

    /** @brief Deleted copy assignment operator. */
    TCENET_Solver& operator=(const TCENET_Solver&) = delete;

    /** @brief Move constructor (defaulted). */
    TCENET_Solver(TCENET_Solver&&) = default;

    /** @brief Deleted move assignment operator. */
    TCENET_Solver& operator=(TCENET_Solver&&) = delete;

    /** @brief Virtual destructor. */
    ~TCENET_Solver() override = default;

    // ============================================================================
    // Serialization & State Management
    // ============================================================================

    friend class cereal::access;

    /**
     * @brief Serialize all internal T-CENET solver state except for input
     * matrices (the penalty configuration lives in the T-CCD base state).
     *
     * @tparam Archive Cereal archive type.
     *
     * @param archive Output/input archive object.
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::base_class<TCCD_Solver>(this)
        );
    }

    /**
     * @brief Save T-CENET solver/model state to file (binary serialization).
     *
     * @param filename Output file path for checkpoint.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load T-CENET solver state from file (binary serialization).
     *
     * @param filename Path to input file.
     * @param X Map to feature matrix.
     * @param D Map to dummy matrix.
     *
     * @return Deserialized TCENET_Solver object connected to X and ready for
     *         warm-start.
     */
    static TCENET_Solver load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D);

};

// =============================================================================
} // End of namespace trex::tsolvers::linear_model::cd_based

// =============================================================================
// Cereal polymorphic registration
// =============================================================================
#include <cereal/types/polymorphic.hpp>
CEREAL_REGISTER_TYPE(trex::tsolvers::linear_model::cd_based::TCENET_Solver)

#endif /* End of TSOLVERS_LINEAR_MODEL_CD_BASED_TCENET_SOLVER_HPP */
