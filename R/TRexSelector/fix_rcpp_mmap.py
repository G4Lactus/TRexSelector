import re
import glob

for rcpp_file in glob.glob("src/rcpp_*.cpp") + glob.glob("src/rcpp_*.h"):
    with open(rcpp_file, "r") as f:
        content = f.read()

    # Wrap the using namespace line if it's there
    content = content.replace("using namespace trex::utils::memmap;", "#ifndef CRAN_BUILD\nusing namespace trex::utils::memmap;\n#endif")

    # Wrap the specific mmap endpoints. 
    # For rcpp_tsolvers.cpp:
    # mmap solver endpoints:
    content = re.sub(r"(//' @title[^\n]*mmap[^\n]*\n//' @noRd\n// \[\[Rcpp::export\]\]\nXPtr.*?return[^\}]+\})", r"#ifndef CRAN_BUILD\n\1\n#endif", content, flags=re.DOTALL)
    
    with open(rcpp_file, "w") as f:
        f.write(content)

