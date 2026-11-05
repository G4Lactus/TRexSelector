// ====================================================================
// vd_common.hpp
// ====================================================================
#ifndef TSOLVERS_VD_TSOLVERS_VD_COMMON_HPP
#define TSOLVERS_VD_TSOLVERS_VD_COMMON_HPP
// ====================================================================
/**
 * @file vd_common.hpp
 *
 * @brief Shared types, options, and helpers for the VD-* family of
 *        virtual-dummy forward selectors.
 *
 * @details Ported from the virtual-dummies project (Koka, Machkour, Muma).
 *          Virtual dummies are never materialized as columns; each dummy is
 *          represented only by its sufficient statistics (projection onto a
 *          growing orthonormal basis plus remaining "stick" mass under the
 *          spherical law), and gets realized on demand inside the solver.
 *
 *          Deviation from the original: the pread/mmap streaming path was
 *          removed. Callers with out-of-core data pass a column-major view
 *          backed by trex::utils::memmap::MemoryMappedMatrix instead, so the
 *          solvers stay agnostic of the storage backend.
 */
// ====================================================================

// std includes
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// ====================================================================

// Embedded into namespace trex::tsolvers::vd
namespace trex::tsolvers::vd {

using Vec     = Eigen::VectorXd;
using MatC    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatR    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMatC = Eigen::Map<const MatC>;
using MapVec  = Eigen::Map<const Vec>;

/** @brief Law of the virtual-dummy null features. */
enum class VDDummyLaw : uint8_t { Spherical = 0, Gaussian = 1 };

/** @brief Per-solver options for the VD-* forward selectors. */
struct VDOptions {
    int    T_stop      = 100;   ///< max dummies to realize (ceiling)
    int    max_vd_proj = 100;   ///< max basis vectors kept for VD projection
    double eps         = 1e-12; ///< numerical zero
    bool   standardize = false; ///< center + unit-L2 normalize X internally
    bool   debug       = false;
    unsigned long long seed = 0ULL;
    VDDummyLaw dummy_law = VDDummyLaw::Spherical;
    double rho = 1.0;           ///< AFS blending parameter (ignored by LARS/OMP)
};

static constexpr int VD_Y_SENTINEL     = -1;
static constexpr int VD_DUMMY_SENTINEL = -2;

/** @brief Entry of the active set: a real feature index or a realized-dummy slot. */
struct ActiveFeature {
    enum class Kind : uint8_t { Real, Dummy };
    Kind kind;
    int  index;
};

namespace vd_detail {

inline int colblock() {
    static int B = []() {
        const char* s = std::getenv("VD_COLBLOCK");
        long v = s ? std::strtol(s, nullptr, 10) : 0;
        if (v <= 0) v = 8192;
        return int(v);
    }();
    return B;
}

/** @brief out = X^T v, blocked over columns for cache friendliness. */
inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out) {
    const int n = (int)X.rows(), p = (int)X.cols();
    out.setZero(p);
    const int B = colblock();
    for (int j0 = 0; j0 < p; j0 += B) {
        const int jb = std::min(B, p - j0);
        const double* Xblk = X.col(j0).data();
        for (int j = 0; j < jb; ++j) {
            const double* x = Xblk + std::ptrdiff_t(j) * n;
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += x[i] * v[i];
            out[j0 + j] = s;
        }
    }
}

/** @brief y = X v, blocked over columns for cache friendliness. */
inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y) {
    const int n = (int)X.rows(), p = (int)X.cols();
    y.setZero(n);
    const int B = colblock();
    for (int j0 = 0; j0 < p; j0 += B) {
        const int jb = std::min(B, p - j0);
        const double* Xblk = X.col(j0).data();
        for (int j = 0; j < jb; ++j) {
            const double* x = Xblk + std::ptrdiff_t(j) * n;
            const double w = v[j0 + j];
            if (w == 0.0) continue;
            for (int i = 0; i < n; ++i) y[i] += x[i] * w;
        }
    }
}

/** @brief Marsaglia-Tsang gamma sampler (shape m, scale 1). */
inline double gamma_mt(double m, std::mt19937_64& rng) {
    auto uniform01 = [&]() {
        constexpr double scale = 1.0 / (1ULL << 53);
        return (rng() >> (64 - 53)) * scale;
    };
    std::normal_distribution<double> normal(0.0, 1.0);
    if (m < 1.0) {
        double g = gamma_mt(m + 1.0, rng);
        double u = std::max(uniform01(), std::numeric_limits<double>::min());
        return g * std::pow(u, 1.0 / m);
    }
    const double d = m - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double z = normal(rng), v = 1.0 + c * z;
        if (v <= 0.0) continue;
        double v3 = v * v * v;
        if (uniform01() < 1.0 - 0.0331 * (z * z) * (z * z)) return d * v3;
        double u = std::max(uniform01(), std::numeric_limits<double>::min());
        if (std::log(u) < 0.5 * z * z + d * (1.0 - v3 + std::log(v3))) return d * v3;
    }
}

}  // namespace vd_detail

}  // namespace trex::tsolvers::vd

// ====================================================================
#endif /* TSOLVERS_VD_TSOLVERS_VD_COMMON_HPP */
// ====================================================================
