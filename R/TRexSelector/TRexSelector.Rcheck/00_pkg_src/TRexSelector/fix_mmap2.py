import os

path_hpp = "src/utils/memmap/memory_mapped_matrix.hpp"
path_cpp = "src/utils/memmap/memory_mapped_matrix.cpp"

with open(path_hpp, "r") as f:
    hpp_content = f.read()
if "CRAN_BUILD" not in hpp_content:
    hpp_content = "#ifndef CRAN_BUILD\n" + hpp_content + "\n#endif // CRAN_BUILD\n"
    with open(path_hpp, "w") as f:
        f.write(hpp_content)

with open(path_cpp, "r") as f:
    cpp_content = f.read()
if "CRAN_BUILD" not in cpp_content:
    cpp_content = "#ifndef CRAN_BUILD\n" + cpp_content + "\n#endif // CRAN_BUILD\n"
    with open(path_cpp, "w") as f:
        f.write(cpp_content)
