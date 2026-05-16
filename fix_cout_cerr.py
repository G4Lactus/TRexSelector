import os
import glob

def replace_in_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r') as f:
        content = f.read()

    if "std::cout" in content or "std::cerr" in content:
        # Add include if not present
        if "#include <utils/logging/logger.hpp>" not in content and "#include \"utils/logging/logger.hpp\"" not in content:
            # find first include, insert after
            if "#include" in content:
                parts = content.split("#include", 1)
                content = parts[0] + "#include <utils/logging/logger.hpp>\n#include" + parts[1]
            else:
                content = "#include <utils/logging/logger.hpp>\n" + content

        # We will use simple regex for some known patterns but sed is safer.
        # Actually doing this with pure python:
        lines = content.split('\n')
        out_lines = []
        for line in lines:
            if "std::cout" in line or "std::cerr" in line:
                if "#include" in line or "using" in line:
                    out_lines.append(line)
                    continue
                # Replace simple stream usage
                line = line.replace("std::cout <<", "TREX_INFO(")
                line = line.replace("std::cerr <<", "TREX_WARN(")
                
                # Close the parenthesis before the semicolon
                if "TREX_INFO(" in line or "TREX_WARN(" in line:
                    if line.rstrip().endswith(";"):
                        line = line.rstrip()[:-1] + ");"
            out_lines.append(line)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(out_lines))

targets = [
    "src/trex_selector_methods/utils/trex_data_normalizer.cpp",
    "src/trex_selector_methods/trex_core/trex.cpp",
    "src/tsolvers/tsolver_base.cpp",
    "src/trex_selector_methods/trex_gvs/trex_gvs.cpp",
    "src/trex_selector_methods/trex_da/trex_da.cpp",
    "src/trex_selector_methods/trex_screening/trex_biobank_screening.cpp",
    "src/ml_methods/clustering/hierarchical/agglomerative/block_tiled_matrix_policy.hpp",
    "src/ml_methods/clustering/hierarchical/agglomerative/proj_geom_updates.hpp"
]

for t in targets:
    replace_in_file(t)

print("Done")
