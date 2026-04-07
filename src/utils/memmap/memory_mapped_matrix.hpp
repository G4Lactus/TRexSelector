// =============================================================================
// memory_mapped_matrix.hpp
// =============================================================================
#ifndef UTILS_MEMORY_MAPPED_MATRIX_HPP
#define UTILS_MEMORY_MAPPED_MATRIX_HPP
// =============================================================================
/**
 * @file memory_mapped_matrix.hpp
 *
 * @brief Declaration of a memory-mapped matrix using Eigen and Boost.
 */
// =============================================================================

// std includes
#include <string>

// boost includes
#include <boost/iostreams/device/mapped_file.hpp>

// Eigen includes
#include <Eigen/Dense>

// ================================================================================

// Embedded into namespace trex::utils::memmap
namespace trex::utils::memmap {

// ================================================================================

/**
 * @brief Access mode for memory-mapped files
 *
 * @enum ReadOnly: Open the file in read-only mode
 * @enum ReadWrite: Open the file in read-write mode
 */
enum class AccessMode {
    ReadOnly,
    ReadWrite
};


/**
 * @brief Memory-mapped matrix using Boost.Iostreams
 *
 * @details This class provides a memory-mapped matrix interface using Boost.Iostreams.
 * @tparam Scalar The scalar type of the matrix elements
 *         (e.g., float, double, std::complex<double>).
 *
 */
template<typename Scalar>
class MemoryMappedMatrix {
public:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using MapType = Eigen::Map<MatrixType>;
    using ConstMapType = Eigen::Map<const MatrixType>;

    // =================================================
    // Constructors
    // =================================================
    /**
     * @brief Constructor to create or open a memory-mapped matrix.
     *
     * @param filename The path to the memory-mapped file.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param mode Access mode (ReadOnly or ReadWrite).
     *
     */
    MemoryMappedMatrix(
        const std::string& filename,
        std::size_t rows,
        std::size_t cols,
        AccessMode mode = AccessMode::ReadWrite
    );

    /** @brief Destructor: closes the memory-mapped file */
    ~MemoryMappedMatrix();

    // Prevent copying
    /** @brief Delete copy constructor */
    MemoryMappedMatrix(const MemoryMappedMatrix&) = delete;

    /** @brief Delete copy assignment operator */
    MemoryMappedMatrix& operator=(const MemoryMappedMatrix&) = delete;


    // =================================================
    // Public Methods
    // =================================================

    /** @brief Get Eigen::Map view (column-major) */
    MapType getMap();

    /** @brief Get const Eigen::Map view (column-major) */
    ConstMapType getMap() const;

    // Raw Pointers
    /** @brief Get raw pointer to data */
    Scalar* data();

    /** @brief Get const raw pointer to data */
    const Scalar* data() const;

    // Accessors
    /** @brief Get number of rows */
    std::size_t rows() const { return rows_; }

    /** @brief Get number of columns */
    std::size_t cols() const { return cols_; }

    /** @brief Get total size (rows * columns) */
    std::size_t size() const { return rows_ * cols_; }


private:
    // =================================================
    // Private Members
    // =================================================

    /** @brief Path to the memory-mapped file */
    std::string filename_;

    /** @brief Number of rows */
    std::size_t rows_;

    /** @brief Number of columns */
    std::size_t cols_;

    /** @brief Access mode */
    AccessMode mode_;

    /** @brief Boost mapped file object */
    boost::iostreams::mapped_file file_;


    // =================================================
    // Private Methods
    // =================================================

    /** @brief Initialize the memory-mapped matrix */
    void initialize();

    /** @brief Apply access hints to the memory-mapped file */
    void applyAccessHints();

};

// ================================================================================

} /* End of namespace trex::utils::memmap */

#endif /* UTILS_MEMORY_MAPPED_MATRIX_HPP */
