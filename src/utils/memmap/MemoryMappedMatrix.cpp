// =============================================================================
// MemoryMappedMatrix.cpp
// =============================================================================
/**
 * @file MemoryMappedMatrix.cpp
 *
 * @brief Implementation of a memory-mapped matrix using Eigen and Boost.
 */
// =============================================================================

#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

#include "utils/memmap/MemoryMappedMatrix.hpp"

// OS-Specific headers for optimization hints
#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#else // POSIX: Linux, macOS, etc.
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

// ================================================================================

namespace trex{
namespace utils{
namespace memmap {

// ===============================================
// Constructors
// ===============================================

template<typename Scalar>
MemoryMappedMatrix<Scalar>::MemoryMappedMatrix(
    const std::string& filename,
    std::size_t rows,
    std::size_t cols,
    AccessMode mode
) : filename_(filename), rows_(rows), cols_(cols), mode_(mode)
{
    initialize();
    applyAccessHints();
}


template<typename Scalar>
MemoryMappedMatrix<Scalar>::~MemoryMappedMatrix() {
    if (file_.is_open()) { file_.close(); }
}


// ===============================================
// Main Methods
// ===============================================

template<typename Scalar>
void MemoryMappedMatrix<Scalar>::initialize() {
    std::size_t required_bytes = rows_ * cols_ * sizeof(Scalar);
    namespace fs = std::filesystem;

    // 1. Handle File Creation / Resizing Explicitly
    if (mode_ == AccessMode::ReadWrite) {
        if (!fs::exists(filename_)) {
            // Create empty file if it doesn't exist
            std::ofstream ofs(filename_, std::ios::binary | std::ios::out);
            if (!ofs) throw std::runtime_error("Could not create file: " + filename_);
            ofs.close();
        }

        std::error_code ec;
        std::uintmax_t current_size = fs::file_size(filename_, ec);

        if (ec || current_size != required_bytes) {
            // Explicitly resize. On POSIX, this calls 'truncate', creating a sparse file.
            try {
                fs::resize_file(filename_, required_bytes);
            } catch (const fs::filesystem_error& e) {
                throw std::runtime_error("Failed to resize file (disk full?): " +
                                         std::string(e.what()));
            }
        }
    }
    else { // ReadOnly
        if (!fs::exists(filename_)) {
            throw std::runtime_error("File not found for ReadOnly access: " + filename_);
        }
        if (fs::file_size(filename_) != required_bytes) {
             throw std::runtime_error("File size mismatch for ReadOnly open.");
        }
    }

    // 2. Open the Mapping
    boost::iostreams::mapped_file_params params;
    params.path = filename_;
    params.length = required_bytes;

    if (mode_ == AccessMode::ReadOnly) {
        params.flags = boost::iostreams::mapped_file::readonly;
    } else {
        params.flags = boost::iostreams::mapped_file::readwrite;
    }

    try {
        file_.open(params);
    } catch (const std::exception& e) {
        throw std::runtime_error("Boost map failed: " + std::string(e.what()));
    }

    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open memory map: " + filename_);
    }
}


template<typename Scalar>
void MemoryMappedMatrix<Scalar>::applyAccessHints() {
    if (!file_.is_open()) return;

#if !defined(_WIN32) && !defined(_WIN64)
    // MADV_SEQUENTIAL tells the OS to aggressively read-ahead
    // MADV_RANDOM would disable read-ahead
    madvise(file_.data(), file_.size(), MADV_SEQUENTIAL);
#endif
}


// ===============================================
// Public Getters
// ===============================================

template<typename Scalar>
typename MemoryMappedMatrix<Scalar>::MapType MemoryMappedMatrix<Scalar>::getMap() {
    if (mode_ == AccessMode::ReadOnly) {
        throw std::runtime_error("Cannot return mutable Eigen::Map in ReadOnly mode");
    }
    return MapType(reinterpret_cast<Scalar*>(file_.data()), rows_, cols_);
}

template<typename Scalar>
typename MemoryMappedMatrix<Scalar>::ConstMapType MemoryMappedMatrix<Scalar>::getMap() const {
    return ConstMapType(reinterpret_cast<const Scalar*>(file_.const_data()), rows_, cols_);
}

template<typename Scalar>
Scalar* MemoryMappedMatrix<Scalar>::data() {
    if (mode_ == AccessMode::ReadOnly) {
        throw std::runtime_error("Cannot return mutable pointer in ReadOnly mode");
    }
    return reinterpret_cast<Scalar*>(file_.data());
}

template<typename Scalar>
const Scalar* MemoryMappedMatrix<Scalar>::data() const {
    return reinterpret_cast<const Scalar*>(file_.const_data());
}

// Explicit Instantiations
template class MemoryMappedMatrix<char>;
template class MemoryMappedMatrix<unsigned char>;
template class MemoryMappedMatrix<short>;
template class MemoryMappedMatrix<unsigned short>;
template class MemoryMappedMatrix<int>;
template class MemoryMappedMatrix<float>;
template class MemoryMappedMatrix<double>;
template class MemoryMappedMatrix<std::complex<float>>;
template class MemoryMappedMatrix<std::complex<double>>;

// ================================================================================

} /* End of namespace memmap */
} /* End of namespace utils */
} /* End of namespace trex */
