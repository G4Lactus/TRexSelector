// ============================================================================
// buffer_registry.hpp
// ============================================================================
#ifndef TSOLVERS_AD_SOLVER_UTILS_BUFFER_REGISTRY_HPP
#define TSOLVERS_AD_SOLVER_UTILS_BUFFER_REGISTRY_HPP
// ============================================================================
/**
 * @file buffer_registry.hpp
 *
 * @brief Process-wide registry of data buffers currently preprocessed in
 *        place by a live T-solver or T-Rex selector (the single-owner
 *        invariant for zero-copy inputs).
 *
 * @details The T-solvers and the T-Rex selectors work on the caller's
 *          buffers through Eigen maps and, when asked to center or
 *          normalize, transform those buffers IN PLACE; the transformation
 *          is undone by an explicit restore() (the solvers) or by the
 *          selector at the end of its run. A second in-place preprocessing
 *          of a buffer that is already transformed is not idempotent: it
 *          moves the data by rounding, and downstream near-ties resolve
 *          differently (found in the T-Stagewise checkpoint
 *          demo). The registry makes the invariant executable at the cost
 *          of a set of pointers: a writer claims each buffer it transforms
 *          for as long as it holds it, and a second writer on a claimed
 *          buffer throws with an explanation. Readers — solvers constructed
 *          with normalize = false and intercept = false, and solvers
 *          resumed from a checkpoint — never claim.
 *
 *          The registry is keyed by the buffer's data pointer and holds a
 *          claim only for the lifetime of the claiming object (RAII
 *          BufferClaim): a claim that outlived its owner could collide with
 *          a new matrix allocated at the same address in an R or Python
 *          session.
 */
// ============================================================================

#include <string>

// ============================================================================

namespace trex::tsolvers::ad::solver_utils::buffer_registry {

/** @brief Whether `ptr` is currently claimed by a live writer. */
bool isClaimed(const void* ptr);

/**
 * @brief Claim `ptr` for in-place preprocessing.
 *
 * @throws std::runtime_error with `message` if the buffer is already claimed.
 */
void claimOrThrow(const void* ptr, const std::string& message);

/** @brief Release a claim (no-op for a null or unclaimed pointer). */
void release(const void* ptr) noexcept;


/**
 * @brief RAII claim on one buffer: claims on construction, releases on
 *        destruction or release(); movable (the claim travels with the
 *        moved-to object), not copyable.
 */
class BufferClaim {
public:
    BufferClaim() = default;

    /** @brief Claims `ptr`; throws (via claimOrThrow) if it is held. */
    BufferClaim(const void* ptr, const std::string& message) {
        claimOrThrow(ptr, message);
        ptr_ = ptr;
    }

    ~BufferClaim() { release(); }

    BufferClaim(BufferClaim&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    BufferClaim& operator=(BufferClaim&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    BufferClaim(const BufferClaim&) = delete;
    BufferClaim& operator=(const BufferClaim&) = delete;

    /** @brief Release the claim early (idempotent). */
    void release() noexcept {
        if (ptr_ != nullptr) {
            buffer_registry::release(ptr_);
            ptr_ = nullptr;
        }
    }

    /** @brief Whether this object currently holds a claim. */
    bool active() const noexcept { return ptr_ != nullptr; }

private:
    const void* ptr_ = nullptr;
};

} // namespace trex::tsolvers::ad::solver_utils::buffer_registry

// ============================================================================
#endif /* TSOLVERS_AD_SOLVER_UTILS_BUFFER_REGISTRY_HPP */
