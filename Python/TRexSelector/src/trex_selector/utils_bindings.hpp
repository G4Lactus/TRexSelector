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

        // Block write — write_block(row_start, row_count, col_start, col_count, values)
        .def("write_block",
             [](MemoryMappedMatrix<double>& self,
                std::size_t row_start, std::size_t row_count,
                std::size_t col_start, std::size_t col_count,
                py::array_t<double, py::array::f_style | py::array::forcecast> values) {

                 py::buffer_info buf = values.request();
                 if (buf.ndim != 2)
                     throw py::value_error("values must be a 2D array");
                 if (static_cast<std::size_t>(buf.shape[0]) != row_count ||
                     static_cast<std::size_t>(buf.shape[1]) != col_count)
                     throw py::value_error(
                         "values shape (" + std::to_string(buf.shape[0]) + ", " +
                         std::to_string(buf.shape[1]) + ") does not match block dimensions (" +
                         std::to_string(row_count) + ", " + std::to_string(col_count) + ")");
                 if (row_start + row_count > self.rows() || col_start + col_count > self.cols())
                     throw py::index_error("Block exceeds matrix bounds");

                 Eigen::Map<const Eigen::MatrixXd> src(
                     static_cast<const double*>(buf.ptr),
                     static_cast<Eigen::Index>(row_count),
                     static_cast<Eigen::Index>(col_count));
                 self.getMap().block(
                     static_cast<Eigen::Index>(row_start),
                     static_cast<Eigen::Index>(col_start),
                     static_cast<Eigen::Index>(row_count),
                     static_cast<Eigen::Index>(col_count)) = src;
             },
             py::arg("row_start"), py::arg("row_count"),
             py::arg("col_start"), py::arg("col_count"),
             py::arg("values"),
             "Write a 2D NumPy block into the mmap at (row_start:row_start+row_count, "
             "col_start:col_start+col_count). Requires ReadWrite mode. Indices are 0-based.")

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
             "Write single element at (row, col) — 0-based indices. Requires ReadWrite mode.")

        // Block __setitem__: mmap[r1:r2, c1:c2] = arr  (also handles int/slice mix)
        .def("__setitem__",
             [](MemoryMappedMatrix<double>& self,
                py::tuple idx,
                py::array_t<double, py::array::f_style | py::array::forcecast> value) {

                 if (idx.size() != 2)
                     throw py::index_error("Index must be a 2-element tuple (row, col)");

                 // Parse a single int or a slice into (start, count)
                 auto parse_index = [](py::object obj, std::size_t dim_size)
                     -> std::pair<std::size_t, std::size_t> {
                     if (py::isinstance<py::int_>(obj)) {
                         auto n = py::cast<std::ptrdiff_t>(obj);
                         if (n < 0) n += static_cast<std::ptrdiff_t>(dim_size);
                         if (n < 0 || static_cast<std::size_t>(n) >= dim_size)
                             throw py::index_error("Index out of bounds");
                         return {static_cast<std::size_t>(n), 1};
                     } else if (py::isinstance<py::slice>(obj)) {
                         py::ssize_t start, stop, step, slicelen;
                         py::cast<py::slice>(obj).compute(
                             static_cast<py::ssize_t>(dim_size), &start, &stop, &step, &slicelen);
                         if (step != 1)
                             throw py::value_error("Only step=1 slices are supported");
                         return {static_cast<std::size_t>(start),
                                 static_cast<std::size_t>(slicelen)};
                     } else {
                         throw py::type_error("Each index must be an int or a slice");
                     }
                 };

                 py::object row_obj = idx[0];
                 py::object col_obj = idx[1];
                 auto [row_start, row_count] = parse_index(row_obj, self.rows());
                 auto [col_start, col_count] = parse_index(col_obj, self.cols());

                 py::buffer_info buf = value.request();
                 if (buf.ndim != 2)
                     throw py::value_error("value must be a 2D array");
                 if (static_cast<std::size_t>(buf.shape[0]) != row_count ||
                     static_cast<std::size_t>(buf.shape[1]) != col_count)
                     throw py::value_error(
                         "value shape (" + std::to_string(buf.shape[0]) + ", " +
                         std::to_string(buf.shape[1]) + ") does not match slice dimensions (" +
                         std::to_string(row_count) + ", " + std::to_string(col_count) + ")");

                 Eigen::Map<const Eigen::MatrixXd> src(
                     static_cast<const double*>(buf.ptr),
                     static_cast<Eigen::Index>(row_count),
                     static_cast<Eigen::Index>(col_count));
                 self.getMap().block(
                     static_cast<Eigen::Index>(row_start),
                     static_cast<Eigen::Index>(col_start),
                     static_cast<Eigen::Index>(row_count),
                     static_cast<Eigen::Index>(col_count)) = src;
             },
             py::arg("idx"), py::arg("value"),
             "Write a 2D NumPy array into the block selected by int/slice indices. "
             "Requires ReadWrite mode. Indices are 0-based.");

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
