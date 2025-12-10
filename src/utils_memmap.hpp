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


// =====================================================================
// RAII MappedFile Structure (Templated)
// =====================================================================
/**
 * @brief Holds a memory-mapped file (RAII, templated for data type)
 *
 * This structure manages a memory-mapped file, ensuring proper cleanup of
 * resources when the object goes out of scope.
 * It supports both POSIX and Windows platforms.
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
    MappedFile(T *d, std::size_t b, int f) : data(d), bytes(b), fd(f)
    {
    }
    ~MappedFile()
    {
        if (data && data != MAP_FAILED)
            munmap(data, bytes);
        if (fd >= 0)
            close(fd);
    }
#else
    // Windows constructor/destructor
    MappedFile(T *d, std::size_t b, HANDLE file, HANDLE mapping)
        : data(d), bytes(b), file_handle(file), mapping_handle(mapping)
    {
    }
    ~MappedFile()
    {
        if (data != nullptr)
            UnmapViewOfFile(data);
        if (mapping_handle)
            CloseHandle(mapping_handle);
        if (file_handle)
            CloseHandle(file_handle);
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
    MappedFile &operator=(MappedFile &&other) noexcept
    {
        if (this != &other)
        {
#if !defined(_WIN32) && !defined(_WIN64)
            if (data && data != MAP_FAILED)
                munmap(data, bytes);
            if (fd >= 0)
                close(fd);
#else
            if (data != nullptr)
                UnmapViewOfFile(data);
            if (mapping_handle)
                CloseHandle(mapping_handle);
            if (file_handle)
                CloseHandle(file_handle);
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
#endif


// =====================================================================
// Map File Functions (Templated)
// =====================================================================
#if !defined(_WIN32) && !defined(_WIN64)


/**
 * @brief Memory-map an existing file for read/write access (POSIX only,
 *        templated)
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
    if (fd < 0)
        throw std::runtime_error(std::string("Could not open file: ")
         + filename);
    T *data = static_cast<T *>(mmap(nullptr, bytes,
                                PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error(std::string("mmap failed for: ") + filename);
    }
    return MappedFile<T>{data, bytes, fd};
}


/**
 * @brief Create and memory-map a new file for read/write access (POSIX only,
 *        templated)
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
inline MappedFile<T> map_file_create(const char *filename,
                                     std::size_t n_elements) {
    std::size_t bytes = n_elements * sizeof(T);
    create_empty_file(filename, bytes);
    return map_file_rw<T>(filename, n_elements);
}

#endif

#if defined(_WIN32) || defined(_WIN64)

/**
 * @brief Memory-map an existing file for read/write access
 *        (Windows only, templated)
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
 * @brief Create and memory-map a new file for read/write access
 * (Windows only, templated)
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
inline MappedFile<T> map_file_create(const char *filename,
                                     std::size_t n_elements)
{
    std::size_t bytes = n_elements * sizeof(T);
    create_empty_file(filename, bytes);
    return map_file_rw<T>(filename, n_elements);
}

#endif /* End of UTILS_MEMMAP_HPP */
