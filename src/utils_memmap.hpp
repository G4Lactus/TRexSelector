#ifndef UTILS_MEMMAP_HPP
#define UTILS_MEMMAP_HPP

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <unistd.h>
#endif


namespace utils_memmap {

// =====================================================================
// RAII MappedFile Structure (Templated)
// =====================================================================
/**
 * @brief Holds a memory-mapped file (RAII, templated for data type)
 *
 * This structure manages a memory-mapped file, ensuring proper cleanup of
 * resources when the object goes out of scope. It supports both POSIX and
 * Windows platforms.
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 */
template <typename T>
struct MappedFile
{
    T *data;
    std::size_t bytes;
#if defined(_WIN32) || defined(_WIN64)
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int fd;
#endif

    // POSIX constructor/destructor
#if !defined(_WIN32) && !defined(_WIN64)
    MappedFile(T *d, std::size_t b, int f) : data(d), bytes(b), fd(f){}
    ~MappedFile() {
        if (data && data != MAP_FAILED) munmap(data, bytes);
        if (fd >= 0) close(fd);
    }
#else
    // Windows constructor/destructor
    MappedFile(T *d, std::size_t b, HANDLE file, HANDLE mapping)
        : data(d), bytes(b), file_handle(file), mapping_handle(mapping) {}
    ~MappedFile() {
        if (data != nullptr) UnmapViewOfFile(data);
        if (mapping_handle) CloseHandle(mapping_handle);
        if (file_handle) CloseHandle(file_handle);
    }
#endif

    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    // Move constructor
    MappedFile(MappedFile &&other) noexcept
#if !defined(_WIN32) && !defined(_WIN64)
        : data(other.data), bytes(other.bytes), fd(other.fd)
#else
        : data(other.data), bytes(other.bytes), file_handle(other.file_handle),
             mapping_handle(other.mapping_handle)
#endif
    {
        other.data = nullptr;
#if !defined(_WIN32) && !defined(_WIN64)
        other.fd = -1;
#else
        other.file_handle = nullptr;
        other.mapping_handle = nullptr;
#endif
    }

    // Move assignment
    MappedFile &operator=(MappedFile &&other) noexcept {
        if (this != &other) {
            // Clean up current mapping
#if !defined(_WIN32) && !defined(_WIN64)
            if (data && data != MAP_FAILED) munmap(data, bytes);
            if (fd >= 0) close(fd);
#else
            if (data != nullptr) UnmapViewOfFile(data);
            if (mapping_handle) CloseHandle(mapping_handle);
            if (file_handle) CloseHandle(file_handle);
#endif
            data = other.data;
            bytes = other.bytes;
#if !defined(_WIN32) && !defined(_WIN64)
            fd = other.fd;
            other.fd = -1;
#else
            file_handle = other.file_handle;
            mapping_handle = other.mapping_handle;
            other.file_handle = nullptr;
            other.mapping_handle = nullptr;
#endif
            other.data = nullptr;
        }
        return *this;
    }
};

// =====================================================================
// Map File Functions (Templated)
// =====================================================================
#if !defined(_WIN32) && !defined(_WIN64) // POSIX Systems

/**
 * @brief Memory-map an existing file for read/write access (POSIX only)
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 *
 * @param filename Path to the file to map.
 * @param n_elements Number of elements to map.
 *
 * @return A `MappedFile<T>` object managing the memory-mapped file.
 *
 * @throws std::runtime_error if the file cannot be opened or mapped.
 */
template <typename T>
inline MappedFile<T> map_file_rw(const char *filename, std::size_t n_elements)
{
    std::size_t bytes = n_elements * sizeof(T);
    int fd = open(filename, O_RDWR);
    if (fd < 0) {
        throw std::runtime_error(std::string("Could not open file: ") + filename);
    }

    T *data = static_cast<T *>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0));
    if (data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error(std::string("mmap failed for: ") + filename);
    }
    return MappedFile<T>{data, bytes, fd};
}


/**
 * @brief Create and memory-map a new file for read/write access (POSIX only)
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 *
 * @param filename Path to the file to create and map.
 * @param n_elements Number of elements to map.
 * @return A `MappedFile<T>` object managing the memory-mapped file.
 *
 * @throws std::runtime_error if the file cannot be created or mapped.
 */
template <typename T>
inline MappedFile<T> create_empty_map(const char *filename, std::size_t n_elements) {
    std::size_t bytes = n_elements * sizeof(T);

    // Create and size the file
    int fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        throw std::runtime_error(std::string("Could not create file: ") + filename);
    }

    if (ftruncate(fd, bytes) != 0) {
        close(fd);
        throw std::runtime_error(std::string("Could not resize file: ") + filename);
    }
    close(fd);

    // Memory-map the newly created file
    return map_file_rw<T>(filename, n_elements);
}


/**
 * @brief Apply madvise hints to a memory-mapped region
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 *
 * @param data Pointer to the start of the memory-mapped region.
 * @param bytes Size of the memory-mapped region in bytes.
 * @param advice The madvise advice to apply (e.g., MADV_SEQUENTIAL).
 */
template <typename T>
inline void apply_madvise_hints(T* data, std::size_t bytes, int advice) {
    if (madvise(data, bytes, advice) != 0) {
        std::cerr << "Warning: madvise failed (non-fatal)\n";
    }
}


/**
 * @brief Flush changes to the memory-mapped file to disk
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 *
 * @param m The `MappedFile<T>` object to flush.
 * @param verbose If true, print warnings on failure.
 */
template <typename T>
inline void flush_mapping(const MappedFile<T>& m, bool verbose = true) {
    if (msync(m.data, m.bytes, MS_SYNC) != 0) {
        if (verbose) std::cerr << "Warning: msync failed (non-fatal)\n";
    }
}


#endif

#if defined(_WIN32) || defined(_WIN64) // Windows Systems

/**
 * @brief Memory-map an existing file for read/write access (Windows only)
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 * @param filename Path to the file to map.
 * @param n_elements Number of elements to map.
 * @return A `MappedFile<T>` object managing the memory-mapped file.
 * @throws std::runtime_error if the file cannot be opened or mapped.
 */
template <typename T>
inline MappedFile<T> map_file_rw(const char *filename, std::size_t n_elements)
{
    std::size_t bytes = n_elements * sizeof(T);

    HANDLE file =
        CreateFileA(filename, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        throw std::runtime_error("Could not open file: " +
              std::string(filename));
    }

    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READWRITE, 0,
         0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(file);
        throw std::runtime_error("Could not create file mapping: " +
              std::string(filename));
    }

    T *data = static_cast<T *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS,
                                             0, 0, bytes));
    if (data == nullptr) {
        CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("Could not map view of file: "
            + std::string(filename));
    }

    return MappedFile<T>{data, bytes, file, mapping};
}

/**
 * @brief Create and memory-map a new file for read/write access (Windows only)
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 *
 * @param filename Path to the file to create and map.
 * @param n_elements Number of elements to map.
 *
 * @return A `MappedFile<T>` object managing the memory-mapped file.
 *
 * @throws std::runtime_error if the file cannot be created or mapped.
 */
template <typename T>
inline MappedFile<T> create_empty_map(const char *filename, std::size_t n_elements)
{
    std::size_t bytes = n_elements * sizeof(T);

    // Create and size the file
    HANDLE file = CreateFileA(filename, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(std::string("Could not create file: ") + filename);
    }

    LARGE_INTEGER size;
    size.QuadPart = bytes;
    if (!SetFilePointerEx(file, size, nullptr, FILE_BEGIN)) {
        CloseHandle(file);
        throw std::runtime_error(std::string("Could not set file size: ") + filename);
    }
    if (!SetEndOfFile(file)) {
        CloseHandle(file);
        throw std::runtime_error(std::string("Could not resize file: ") + filename);
    }
    CloseHandle(file);

    // Memory-map the newly created file
    return map_file_rw<T>(filename, n_elements);
}

/**
 * @brief Flush changes to the memory-mapped file to disk (Windows only)
 *
 * @tparam T The data type of the elements in the memory-mapped file.
 *
 * @param m The `MappedFile<T>` object to flush.
 * @param verbose If true, print warnings on failure.
 */
template <typename T>
inline void flush_mapping(const MappedFile<T>& m, bool verbose = true) {
    if (!FlushViewOfFile(m.data, m.bytes)) {
        if (verbose) std::cerr << "Warning: FlushViewOfFile failed (non-fatal)\n";
    }
}

#endif

} /* End of namespace utils_memmap */

#endif /* End of UTILS_MEMMAP_HPP */
