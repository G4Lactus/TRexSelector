// ===================================================================================
// utils_eigen_threads.hpp
// ===================================================================================
#ifndef UTILS_EIGEN_THREADS_HPP
#define UTILS_EIGEN_THREADS_HPP
// ===================================================================================
/**
 * @file utils_eigen_threads.hpp
 *
 * @brief Scoped thread budget for Eigen's own parallel kernels.
 *
 * @details Eigen parallelizes its large products through OpenMP in builds
 * without a BLAS backend and sizes the team from its global thread count,
 * which defaults to the OpenMP runtime's team. The guard below sets that
 * count to the caller's budget for the scope of a computation and restores
 * the previous value on exit, so a default call stays single-threaded and
 * nothing leaks into the host process.
 */
// ===================================================================================

#include <Eigen/Core>

#include <utils/openmp/utils_openmp.hpp>

// ===================================================================================

namespace trex::utils::openmp {

/**
 * @brief RAII guard that sets Eigen's thread count to a budget for its scope.
 *
 * Inside a caller's region (an experiment runner's team) it does nothing:
 * that caller owns the global setting, and a write from inside a team would
 * race. Restores the previous value on exit so nothing leaks into the host.
 */
class EigenThreadBudgetGuard {
public:
    explicit EigenThreadBudgetGuard(int n_threads)
        : active_(omp_in_parallel() == 0), previous_(Eigen::nbThreads()) {
        if (active_) { Eigen::setNbThreads(n_threads > 0 ? n_threads : 1); }
    }
    ~EigenThreadBudgetGuard() { if (active_) { Eigen::setNbThreads(previous_); } }

    EigenThreadBudgetGuard(const EigenThreadBudgetGuard&) = delete;
    EigenThreadBudgetGuard& operator=(const EigenThreadBudgetGuard&) = delete;

private:
    bool active_;
    int previous_;
};

} /* End of namespace trex::utils::openmp */

// ===================================================================================
#endif /* UTILS_EIGEN_THREADS_HPP */
// ===================================================================================
