#include <iostream>
#include <Eigen/Core>

void checkEigenBackend() {
    std::cout << "==============================================\n";
    std::cout << "Eigen Configuration Check\n";
    std::cout << "==============================================\n";

    // Check SIMD
    std::cout << "SIMD Instructions: "
              << Eigen::SimdInstructionSetsInUse() << "\n";

    // Check BLAS
    #ifdef EIGEN_USE_BLAS
        std::cout << "BLAS Backend: ENABLED ✓\n";
    #else
        std::cout << "BLAS Backend: DISABLED ✗ (SLOW!)\n";
    #endif

    // Check LAPACK
    #ifdef EIGEN_USE_LAPACKE
        std::cout << "LAPACK Backend: ENABLED ✓\n";
    #else
        std::cout << "LAPACK Backend: DISABLED\n";
    #endif

    // Check MKL (if available)
    #ifdef EIGEN_USE_MKL_ALL
        std::cout << "Intel MKL: ENABLED\n";
    #endif

    std::cout << "==============================================\n\n";
}

int main() {
    checkEigenBackend();
    return 0;
}
