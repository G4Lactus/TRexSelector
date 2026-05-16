import os

path_hpp = "../../src/utils/memmap/memory_mapped_matrix.hpp"
path_cpp = "../../src/utils/memmap/memory_mapped_matrix.cpp"

with open(path_hpp, "r") as f:
    hpp_content = f.read()

# Add ifndef CRAN_BUILD
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

rcpp_path = "src/rcpp_trex_selector.cpp"
with open(rcpp_path, "r") as f:
    rcpp = f.read()

import re
# Wrap mmap endpoints in Rcpp with #ifndef CRAN_BUILD
rcpp = re.sub(r"(//' @title Create TRexSelector from mmap\n//' @noRd\n// \[\[Rcpp::export\]\]\n.*?return[^\}]+\})", r"#ifndef CRAN_BUILD\n\1\n#endif", rcpp, flags=re.DOTALL)
rcpp = re.sub(r"(//' @title Create TRexDASelector from mmap\n//' @noRd\n// \[\[Rcpp::export\]\]\nXPtr<RTRexDASelector> trex_da_mmap_create.*?return[^\}]+\})", r"#ifndef CRAN_BUILD\n\1\n#endif", rcpp, flags=re.DOTALL)

with open(rcpp_path, "w") as f:
    f.write(rcpp)

