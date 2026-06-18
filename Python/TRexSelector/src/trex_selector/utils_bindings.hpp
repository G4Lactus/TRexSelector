// =====================================================================================
// utils_bindings.hpp - Pybind11 bindings for utility classes and functions
// =====================================================================================
#ifndef TREX_PYTHON_UTILS_BINDINGS_HPP
#define TREX_PYTHON_UTILS_BINDINGS_HPP
// =====================================================================================
/**
 * @file utils_bindings.hpp
 * @brief Pybind11 bindings for utility classes and functions.
 */
// =====================================================================================

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

// utils includes
#include <utils/memmap/memory_mapped_matrix.hpp>
#include <utils/eval_metrics/utils_eval_counts.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>
#include <utils/openmp/utils_openmp.hpp>

// =====================================================================================

namespace py = pybind11;

// =====================================================================================

/**
 * @brief Binds the utility structures such as memory-mapped matrices.
 *
 * @param m The Python module to bind against.
 */
inline void bind_utils_module(py::module& m) {
    using trex::utils::memmap::AccessMode;
    using trex::utils::memmap::MemoryMappedMatrix;

    py::enum_<AccessMode>(m, "AccessMode")
        .value("ReadOnly", AccessMode::ReadOnly)
        .value("ReadWrite", AccessMode::ReadWrite)
        .export_values();

    py::class_<MemoryMappedMatrix<double>>(m, "MemoryMappedMatrix")

        .def(py::init<const std::string&, std::size_t, std::size_t, AccessMode>(),
             py::arg("filename"), py::arg("rows"), py::arg("cols"),
             py::arg("mode") = AccessMode::ReadWrite)

        .def("rows", &MemoryMappedMatrix<double>::rows)
        .def("cols", &MemoryMappedMatrix<double>::cols)
        .def("size", &MemoryMappedMatrix<double>::size)

        // Expose zero-copy numpy array mapped to the memory
        .def("to_numpy", [](MemoryMappedMatrix<double>& mmap_mat) -> py::array_t<double> {
            auto map = mmap_mat.getMap();
            // Create a NumPy array sharing the buffer, with Fortran (column-major) strides
            return py::array_t<double>(
                {map.rows(), map.cols()},                        // shape
                {sizeof(double), map.rows() * sizeof(double)}, // strides (column-major)
                map.data(),                                                // pointer to data
                py::cast(&mmap_mat)                                // base object to tied lifecycle
            );
        }, "Returns a zero-copy NumPy array view of the memory-mapped matrix.")

        // Element access (Eigen-like interface): mmap[row, col]
        .def("__getitem__",
             [](const MemoryMappedMatrix<double>& self,
                std::pair<std::size_t, std::size_t> idx) {
                 return self(idx.first, idx.second);
             },
             py::arg("idx"),
             "Read single element at (row, col) — 0-based indices.")

        .def("__setitem__",
             [](MemoryMappedMatrix<double>& self,
                std::pair<std::size_t, std::size_t> idx,
                double value) {
                 self(idx.first, idx.second) = value;
             },
             py::arg("idx"), py::arg("value"),
             "Write single element at (row, col) — 0-based indices. Requires ReadWrite mode.");

    m.def("convert_to_memory_mapped",
          [](py::array_t<double>& arr, const std::string& filename) {

        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Only 2D arrays are supported for memory mapping conversion.");
        }
        std::size_t rows = buf.shape[0];
        std::size_t cols = buf.shape[1];

        // Ensure Fortran contiguous (column-major) layout for Eigen compatibility
        py::array_t<double, py::array::f_style | py::array::forcecast> f_arr(arr);
        py::buffer_info f_buf = f_arr.request();

        MemoryMappedMatrix<double> mmap(filename, rows, cols, AccessMode::ReadWrite);
        auto map = mmap.getMap();
        std::memcpy(map.data(), f_buf.ptr, rows * cols * sizeof(double));
    }, py::arg("array"),
       py::arg("filename"),
       "Converts a NumPy array to a memory-mapped binary file.");

    // =================================================================================
    // Eval Metrics
    // =================================================================================

    // Using unambiguous casts to resolve templates and overloads
    m.def("compute_fdp",
          static_cast<double(*)(const std::vector<std::size_t>&,
                                  const std::vector<std::size_t>&)>(
                                    &trex::utils::eval::rates::compute_fdp),
          py::arg("selected_indices"), py::arg("true_support"),
          "Compute FDP from indices");

    m.def("compute_tpp",
          static_cast<double(*)(const std::vector<std::size_t>&,
                                  const std::vector<std::size_t>&)>(
                                    &trex::utils::eval::rates::compute_tpp),
          py::arg("selected_indices"), py::arg("true_support"),
          "Compute TPP from indices");

    m.def("compute_precision",
          &trex::utils::eval::rates::compute_precision,
          py::arg("selected_indices"), py::arg("true_support"),
          "Compute precision from indices");

    m.def("compute_recall",
          &trex::utils::eval::rates::compute_recall,
          py::arg("selected_indices"), py::arg("true_support"),
          "Compute recall from indices");

    m.def("compute_fdp_dense", [](const Eigen::VectorXd& beta_hat,
        const Eigen::VectorXd& beta, double eps) {
        return trex::utils::eval::rates::compute_fdp(beta_hat, beta, eps);
    }, py::arg("beta_hat"),
    py::arg("beta"),
    py::arg("eps") = 1e-15, "Compute FDP from dense coefficient vectors");

    m.def("compute_tpp_dense", [](const Eigen::VectorXd& beta_hat,
        const Eigen::VectorXd& beta, double eps) {
        return trex::utils::eval::rates::compute_tpp(beta_hat, beta, eps);
    }, py::arg("beta_hat"),
    py::arg("beta"),
    py::arg("eps") = 1e-15,
    "Compute TPP from dense coefficient vectors");

    // =================================================================================
    // OpenMP wrappers
    // =================================================================================
    m.def("get_max_threads",
          &trex::utils::openmp::get_max_threads,
          "Get maximum number of threads used by OpenMP");

    m.def("set_num_threads",
          &trex::utils::openmp::set_num_threads,
          py::arg("num_threads"),
          "Set number of threads for OpenMP");
}

// =====================================================================================
#endif /* End of TREX_PYTHON_UTILS_BINDINGS_HPP */
// =====================================================================================
