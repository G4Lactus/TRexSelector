// ===================================================================================
// svd.hpp
// ===================================================================================
#ifndef TREX_ML_METHODS_SVD_HPP
#define TREX_ML_METHODS_SVD_HPP
// ===================================================================================
/**
 * @file svd.hpp
 *
 * @brief Singular Value Decomposition (SVD).
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <random>
#include <stdexcept>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// ===================================================================================

// Namespace declarations
namespace trex {
namespace ml_methods {
namespace svd {

// ===================================================================================

/**
 * @brief Container for final SVD components.
 */
struct SVDResult {
    /** @brief Left singular vectors */
    Eigen::MatrixXd U;

    /** @brief Singular values */
    Eigen::VectorXd S;

    /** @brief Right singular vectors */
    Eigen::MatrixXd V;
};


/**
 * @class SVDSolver
 *
 * @brief Dimensionality-aware SVD dispatcher.
 *
 * @details Selects among three computation strategies based on matrix shape:
 *  - Randomized SVD (RedSVD) for massive n and p with small M.
 *  - Gram / kernel trick for the wide (p >> n) regime.
 *  - Thin Jacobi SVD as the general fallback.
 */
class SVDSolver {
public:

    /**
     * @brief Construct an SVDSolver with tunable dispatch and cache parameters.
     *
     * @param min_dim_thresh Minimum dimension size (for both n and p) at which the
     *                           Randomized SVD path is considered. Default: 10,000.
     * @param block_size         Column-block width used in cache-friendly matrix products
     *                           inside the RedSVD and Gram SVD paths. Default: 1,024.
     */
    explicit SVDSolver(Eigen::Index min_dim_thresh = 10'000,
                       Eigen::Index block_size     = 1'024)
        : min_dim_thresh_(min_dim_thresh)
        , block_size_(block_size)
    {}


    /** @brief Default destructor */
    ~SVDSolver() = default;


    /**
     * @brief Compute the Singular Value Decomposition (SVD) of a matrix.
     *
     * @param X Input matrix.
     * @param M Number of singular values and vectors to compute.
     *
     * @return SVDResult containing the left singular vectors (U),
     *         singular values (S), and right singular vectors (V).
     */
    SVDResult compute(const Eigen::Ref<const Eigen::MatrixXd>& X,
                      Eigen::Index M) {
        Eigen::Index n = X.rows();
        Eigen::Index p = X.cols();

        if (M < 1) {
            throw std::invalid_argument("M must be at least 1");
        }
        if (M > std::min(n, p)) {
            throw std::invalid_argument("M must not exceed min(n, p)");
        }

        // Dispatcher: dimensionality-based SVD computation
        if (p > 2 * n) {
            return compute_gram_svd(X, M);            // exact, O(n²p + n³)
        } else if (std::min(n, p) > min_dim_thresh_ && M < std::min(n, p) / 10) {
            return compute_redsvd(X, M);              // approximate, O(npM)
        } else {
            return compute_direct_svd(X, M);
        }
    }

private:

    /**
     * @brief Compute the SVD using the Randomized SVD (red-SVD) for massive n and p.
     *
     * @param X Input matrix.
     * @param M Number of singular values and vectors to compute.
     * @param oversampling Oversampling parameter for the sketching matrix (default: 10)
     *
     * @return SVDResult containing the approximated left singular vectors (U),
     *         singular values (S), and right singular vectors (V).
     */
    SVDResult compute_redsvd(const Eigen::Ref<const Eigen::MatrixXd>& X,
                             Eigen::Index M,
                             Eigen::Index oversampling = 10) {

        Eigen::Index n = X.rows();
        Eigen::Index p = X.cols();
        Eigen::Index l = std::min(M + oversampling, std::min(n, p));

        // 1. Generating a random Gaussian sketching matrix
        Eigen::MatrixXd Omega(p, l);

        // Deterministic seed for reproducibility
        std::mt19937 gen(42);
        std::normal_distribution<double> ndist(0.0,1.0);
        Omega = Eigen::MatrixXd::NullaryExpr(p, l,
            [&](Eigen::Index, Eigen::Index) { return ndist(gen); });

        // 2. Computing the sketch Y = X * Omega.
        // Block-wise to preserve the CPU cache and avoid full out-of-core reads
        Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(n, l);
        for (Eigen::Index j = 0; j < p; j += block_size_) {
            Eigen::Index current_block = std::min(block_size_, p -j);
            Y += X.middleCols(j, current_block) *
                 Omega.middleRows(j, current_block);
        }

        // 3. Orthonormalizing Y to get Q.
        // Y is small (n x l), QR fits RAM usually
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, l);

        // 4. Computing B^T = Q^T * X of dim (p x l)
        // B^T is constructed block-by-block to maintain contiguous reads of X
        Eigen::MatrixXd Bt(p, l);
        for (Eigen::Index j = 0; j < p; j += block_size_) {
            Eigen::Index current_block = std::min(block_size_, p - j);
            Bt.middleRows(j, current_block) = X.middleCols(
                j, current_block).transpose() * Q;
        }

        // 5. Deterministic SVD of B^T
        // B^T = V * Sigma * U_tilde^T
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Bt,
            Eigen::ComputeThinU | Eigen::ComputeThinV);

        SVDResult result;
        // Truncate to top M components
        result.S = svd.singularValues().head(M);

        // The left singular vectors of B^T are the right singular vectors of X
        result.V = svd.matrixU().leftCols(M);

        // 6. Reconstruct true left singular vectors: U = Q * U_tilde
        Eigen::MatrixXd U_tilde = svd.matrixV().leftCols(M);
        result.U = Q * U_tilde;

        return result;
    }


    /**
    * @brief Compute the SVD using the Gram matrix method (kernel trick).
    *
    * @param X Input matrix.
    * @param M Number of singular values and vectors to compute.
    *
    * @return SVDResult containing the left singular vectors (U), singular values (S), and
    *         right singular vectors (V).
    */
    SVDResult compute_gram_svd(
        const Eigen::Ref<const Eigen::MatrixXd>& X,
        Eigen::Index M
    ) {

        Eigen::Index n = X.rows();
        Eigen::Index p = X.cols();

        // 1. Initialize the n x n Gram matrix with zeros
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);

        // 2. Loop-based Blocked Matrix Multiplication
        for (Eigen::Index j = 0; j < p; j += block_size_) {
            Eigen::Index current_block_size = std::min(block_size_, p - j);

            // Contiguous block of columns (zero-copy map)
            auto X_block = X.middleCols(j, current_block_size);

            // Accumulate outer product: K += X_block * X_block^T
            K.selfadjointView<Eigen::Lower>().rankUpdate(X_block);
        }

        // 3. Eigendecomposition of symmetric positive semi-definite K
        // Note: SelfAdjointEigenSolver defaults to reading only the lower triangular part
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(K);
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error("Eigendecomposition failed.");
        }

        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().reverse();
        Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors().rowwise().reverse();

        SVDResult result;
        // 4. Extract the top M components
        result.S = eigenvalues.head(M).cwiseMax(0.0).cwiseSqrt();
        result.U = eigenvectors.leftCols(M);

        // 5. Recover right singular vectors V = X^T * U * Sigma^{-1}
        // for singular values close to zero we add a tolerance to avoid division by zero
        Eigen::VectorXd S_inv = result.S.array().unaryExpr(
            [](double s) { return s > 1e-12 ? 1.0 / s : 0.0;});
        Eigen::MatrixXd U_scaled = result.U * S_inv.asDiagonal();


        // Note: Safe, out-of-core block-wise construction of V
        result.V.resize(p, M);
        for (Eigen::Index j = 0; j < p; j += block_size_) {

            Eigen::Index current_block_size = std::min(block_size_, p - j);
            auto X_block = X.middleCols(j, current_block_size);

            // X_block is (n x block_size), X_block^T is (block_size x n)
            // U_scaled is (n x M)
            // Resulting block is (block_size x M) and fits into V's rows
            result.V.middleRows(j, current_block_size) = X_block.transpose() * U_scaled;
        }

        return result;
    }


    /**
     * @brief Compute the SVD using the Thin Jacobi method.
     *
     * @param X Input matrix.
     * @param M Number of singular values and vectors to compute.
     *
     * @return SVDResult containing the left singular vectors (U), singular values (S), and
     *         right singular vectors (V).
     */
    SVDResult compute_direct_svd(
        const Eigen::Ref<const Eigen::MatrixXd>& X, Eigen::Index M
    ) {
        // Compute Thin SVD: Jacobi is immune to --fast-math IEEE relaxations
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(X,
            Eigen::ComputeThinU | Eigen::ComputeThinV);

        SVDResult result;
        result.U = svd.matrixU().leftCols(M);
        result.S = svd.singularValues().head(M);
        result.V = svd.matrixV().leftCols(M);

        return result;
    }

    // ===================================================================================
    // Private member variables
    // ===================================================================================

    /** @brief Minimum value of both n and p that triggers the Randomized SVD path. */
    Eigen::Index min_dim_thresh_;

    /** @brief Column-block width used in cache-friendly matrix products. */
    Eigen::Index block_size_;
};


// ===================================================================================
} /* End of namespace svd */
} /* End of namespace ml_methods */
} /* End of namespace trex */
// ===================================================================================
#endif /* End of TREX_ML_METHODS_SVD_HPP */
