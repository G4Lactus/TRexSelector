// ============================================================================
// buffer_registry.cpp
// ============================================================================
/**
 * @file buffer_registry.cpp
 *
 * @brief Implementation of the process-wide claim registry for buffers
 *        preprocessed in place (see buffer_registry.hpp).
 */
// ============================================================================

#include <tsolvers/ad/solver_utils/buffer_registry.hpp>

#include <mutex>
#include <set>
#include <stdexcept>

// ============================================================================

namespace trex::tsolvers::ad::solver_utils::buffer_registry {

namespace {

std::mutex& registryMutex() {
    static std::mutex m;
    return m;
}

std::set<const void*>& registry() {
    static std::set<const void*> reg;
    return reg;
}

} // namespace


bool isClaimed(const void* ptr) {
    if (ptr == nullptr) { return false; }
    std::lock_guard<std::mutex> lock(registryMutex());
    return registry().count(ptr) > 0;
}


void claimOrThrow(const void* ptr, const std::string& message) {
    if (ptr == nullptr) { return; }
    std::lock_guard<std::mutex> lock(registryMutex());
    if (!registry().insert(ptr).second) {
        throw std::runtime_error(message);
    }
}


void release(const void* ptr) noexcept {
    if (ptr == nullptr) { return; }
    std::lock_guard<std::mutex> lock(registryMutex());
    registry().erase(ptr);
}

} // namespace trex::tsolvers::ad::solver_utils::buffer_registry

// ============================================================================
