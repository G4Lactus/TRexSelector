// ===================================================================================
// trex_biobank_screening.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_TREX_BIOBANK_SCREENING_HPP
#define TREX_SELECTOR_METHODS_TREX_BIOBANK_SCREENING_HPP
// ===================================================================================
/**
 * @file trex_biobank_screening.hpp
 *
 * @brief Genomics biobank screening implementing Algorithm 1 from the Screen-TRex paper.
 *
 * @details Coordinates Screen-TRex and T-Rex selectors for efficient large-scale
 *          phenotype screening of genomics biobanks.
 *
 * Reference: Machkour J, Muma M, Palomar D
 * "False Discovery Rate Control for Fast Screening of Large-Scale Genomics Biobanks"
 * (IEEE SSP 2023)
 */
// ===================================================================================

// std includes
#include <cstddef>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <trex_selector_methods/trex_core/trex.hpp>
#include <trex_selector_methods/trex_screening/trex_screening.hpp>

// ===================================================================================

namespace trex::trex_selector_methods::trex_biobank_screening {

// Namespace aliases
namespace tc = trex::trex_selector_methods::trex_core;
namespace ts = trex::trex_selector_methods::trex_screening;

// ===================================================================================


// =========================================================
// Structures & Enums
// =========================================================

/** @brief Result of screening a single phenotype. */
struct BiobankScreenTRexResult {

    /** @brief Index of the screened phenotype in matrix Y. */
    std::size_t phenotype_index;

    /** @brief Final selected variable indices (from chosen method). */
    std::vector<std::size_t> selected_indices;

    /** @brief Estimated FDR of final selection. */
    double estimated_FDR;

    /** @brief Method used for selection */
    std::string method_used;

    /** @brief Estimated FDR (ordinary). */
    double estimated_FDR_screen_ordinary;

    /** @brief Estimated FDR (bootstrap). */
    double estimated_FDR_screen_bootstrap;

    /** @brief Selected indices from ordinary Screen-TRex
     *
     * @details Empty if not used in final selection.
     */
    std::vector<std::size_t> selected_indices_screen_ordinary;

    /** @brief Selected indices from bootstrap Screen-TRex
     *
     * @details Empty if not used in final selection.
     */
    std::vector<std::size_t> selected_indices_screen_bootstrap;

    /** @brief True if standard T-Rex was used */
    bool used_fallback_trex;
};


/**
 * @brief Control parameters for biobank screening (Algorithm 1).
 */
struct BiobankScreenTRexControl {

    /** @brief Target FDR level α for fallback T-Rex selector */
    double target_FDR_trex = 0.1;

    /**
     * @brief Lower bound α_l for acceptable estimated FDR.
     *
     * @details If estimated FDR < α_l, power might be too low.
     *          Algorithm 1 uses this to determine whether to accept Screen-TRex result.
     */
    double lower_bound_FDR = 0.05;

    /**
     * @brief Upper bound α_u for acceptable estimated FDR.
     *
     * @details If estimated FDR > α_u, false discoveries might be too many.
     *          Algorithm 1 uses this to determine whether to accept Screen-TRex result.
     */
    double upper_bound_FDR = 0.15;

    /** @brief Control parameters forwarded to the Screen-TRex selector. */
    ts::ScreenTRexControlParameter screen_ctrl;

    /** @brief Algorithmic control forwarded to both Screen-TRex and the fallback T-Rex. */
    tc::TRexControlParameter trex_ctrl;
};


/**
 * @brief Biobank screening orchestrator implementing Algorithm 1 from Screen-TRex paper.
 *
 * @details The algorithm is proposed as solution to efficiently screen thousands of phenotypes
 *          by:
 *          1. Running fast Screen-TRex
 *          2. Only using full T-Rex when Screen-TRex estimate is outside [α_l, α_u].
 *
 * @note Algorithm 1 requires BOTH ordinary (α̂) and bootstrap-CI (α̂_C)  FDR estimates for its
 *       decision logic. Therefore, Screen-TRex is run twice per phenotype (once in each mode).
 *
 * Reference: Machkour J, Muma M, Palomar D
 * "False Discovery Rate Control for Fast Screening of Large-Scale Genomics Biobanks"
 * (IEEE SSP 2023)
 */
class BiobankScreenTRex {
protected:
    // ======================================================================
    // Protected Members
    // ======================================================================

    /** @brief Pointer to design matrix (n x p) */
    Eigen::Map<Eigen::MatrixXd>* X_;

    /** @brief Phenotype vector for single phenotype mode (n x 1) */
    Eigen::VectorXd y_;

    /** @brief Pointer to phenotype matrix for multiple phenotype mode (n x q) */
    Eigen::Map<Eigen::MatrixXd>* Y_;

    /** @brief Control parameters for Biobank Screen-TRex */
    BiobankScreenTRexControl biosctrex_ctrl_;

    /** @brief Random seed for reproducibility */
    int seed_;

    /** @brief Verbosity flag */
    bool verbose_;


public:
    // ======================================================================
    // Constructors
    // ======================================================================

    /**
     * @brief Constructor for single phenotype screening.
     *
     * @param X Design matrix (n x p)
     * @param y Phenotype vector (n x 1)
     * @param biosctrex_ctrl Control parameters for Biobank Screen-TRex (moved)
     * @param seed Random seed for reproducibility (default: -1, random)
     * @param verbose If true, print progress messages (default: false)
     */
    BiobankScreenTRex(
        Eigen::Map<Eigen::MatrixXd>& X,
        Eigen::Map<Eigen::VectorXd>& y,
        BiobankScreenTRexControl biosctrex_ctrl,
        int seed = -1,
        bool verbose = false
    );


    /**
     * @brief Constructor for multiple phenotype screening.
     *
     * @param X Design matrix (n x p)
     * @param Y Phenotype matrix (n x q)
     * @param biosctrex_ctrl Control parameters for Biobank Screen-TRex (moved)
     * @param seed Random seed for reproducibility (default: -1, random)
     * @param verbose If true, print progress messages (default: false)
     */
    BiobankScreenTRex(
        Eigen::Map<Eigen::MatrixXd>& X,
        Eigen::Map<Eigen::MatrixXd>& Y,
        BiobankScreenTRexControl biosctrex_ctrl,
        int seed = -1,
        bool verbose = false
    );


    /** @brief Virtual destructor */
    virtual ~BiobankScreenTRex() = default;


    // Delete copy and move constructors and assignment operators
    /** @brief Deleted copy constructor */
    BiobankScreenTRex(const BiobankScreenTRex&) = delete;


    /** @brief Deleted copy assignment operator */
    BiobankScreenTRex& operator=(const BiobankScreenTRex&) = delete;


    /** @brief Deleted move constructor */
    BiobankScreenTRex(BiobankScreenTRex&&) = delete;


    /** @brief Deleted move assignment operator */
    BiobankScreenTRex& operator=(BiobankScreenTRex&&) = delete;


    // ======================================================================
    // Main Methods
    // ======================================================================

    /**
     * @brief Screen a single phenotype using Biobank Screen-TRex (Algorithm 1).
     *
     * @details Procedure:
     *          1. Run bootstrap-CI Screen-TRex: get α̂_C and A_G^(C)(γ, 1)
     *          2. Run ordinary Screen-TRex: get α̂ and A_p(0.5, 1)
     *          3. Apply Algorithm 1 decision logic
     *          4. Return selected set (Screen-TRex) or run fallback T-REx
     *
     * @return Screening result with selected variables and metadata.
     *
     * @throws std::runtime_error if called with multiple phenotypes
     *         (use screenPhenotypes() instead).
     */
    BiobankScreenTRexResult screenPhenotype();


    /**
     * @brief Screen all phenotypes in matrix Y using Biobank Screen-TRex (Algorithm 1).
     *
     * @return Screening results for all phenotypes with selected variables and metadata.
     *
     * @throws std::runtime_error if called without Y matrix (use screenPhenotype() instead).
     */
    std::vector<BiobankScreenTRexResult> screenPhenotypes();


    // ======================================================================
    // Public Getters and Structures
    // ======================================================================

    /**
     * @brief Get summary statistics across all phenotypes.
     */
    struct BiobankScreenTRexStatistics {
        /** @brief Total number of phenotypes. */
        std::size_t num_phenotypes;

        /** @brief Number using ordinary Screen-TRex result. */
        std::size_t num_used_screen_ordinary = 0;

        /** @brief Number using bootstrap-CI Screen-TRex result. */
        std::size_t num_used_screen_bootstrap = 0;

        /** @brief Number of phenotypes using fallback T-REx result. */
        std::size_t num_used_trex = 0;

        /** @brief Average estimated FDR across all phenotypes. */
        double avg_estimated_FDR;

        /** @brief Average number of selected variables across all phenotypes. */
        double avg_selected_count;
    };


    /**
     * @brief Get the Biobank Screen T-REx Result object.
     *
     * @param results Vector of individual phenotype screening results.
     *
     * @return BiobankScreenTRexStatistics object containing summary statistics.
     */
    BiobankScreenTRexStatistics getBiobankScreenTRexResult(
        const std::vector<BiobankScreenTRexResult>& results
    ) const;


protected:
    // ======================================================================
    // Protected Methods - Algorithm 1
    // ======================================================================

    /** @brief Screen a single phenotype column.
     *
     * @param y Phenotype vector.
     * @param phenotype_index Index of the phenotype being screened.
     *
     * @return Result from Screening after applying Algorithm 1.
     */
    BiobankScreenTRexResult screenSinglePhenotype(
        const Eigen::VectorXd& y,
        std::size_t phenotype_index
    );

    /**
     * @brief Convert binary selection vector to index vector.
     *
     * @param mask Binary vector (0/1) indicating selected variables.
     *
     * @return Vector of indices where mask == 1.
     */
    std::vector<std::size_t> selectedVarsToIndices(const Eigen::VectorXi& mask) const;


    /**
     * @brief Apply Algorithm 1's decision logic (Step 2.2).
     *
     * @details Choses between:
     *      - Bootstrap-CI Screen-TRex result A_G^(C)(γ, 1) with estimate α̂_C
     *      - Ordinary Screen-TRex result A_p(0.5, 1) with estimate α̂
     *      - Fallback T-REx selector result
     *
     * @param result_ordinary Result from ordinary Screen-TRex.
     * @param result_bootstrap Result from boostrap-CI Screen-TRex.
     * @param y Phenotype vector.
     * @param phenotype_index Index of the phenotype being screened.
     *
     * @return Final screening result after applying decision logic.
     */
    BiobankScreenTRexResult applyDecisionLogic(
        const ts::ScreenTRexSelectionResult& sctrex_result_bootstrap,
        const ts::ScreenTRexSelectionResult& sctrex_result_ordinary,
        const Eigen::VectorXd& y,
        std::size_t phenotype_index
    );


    /**
     * @brief Run fallback T-REx selector selector (Algorithm 1, Step 2.3).
     *
     * @param y Phenotype vector.
     * @param phenotype_idx Index of the phenotype being screened.
     *
     * @return Result from T-REx selector.
     */
    BiobankScreenTRexResult runFallbackTRexSelector(
        const Eigen::VectorXd& y,
        std::size_t phenotype_idx
    );


    /** @brief Print progress message if verbosity is enabled.
     *
     * @param msg Message to print
     */
    void printProgressMessage(const std::string& msg) const;

};

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::trex_biobank_screening */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_TREX_BIOBANK_SCREENING_HPP */
