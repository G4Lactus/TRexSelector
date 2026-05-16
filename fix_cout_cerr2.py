import os

def replace_in_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r') as f:
        content = f.read()

    if "std::cout" in content or "std::cerr" in content:
        if "#include <utils/logging/logger.hpp>" not in content and "#include \"utils/logging/logger.hpp\"" not in content:
            if "#include" in content:
                parts = content.split("#include", 1)
                content = parts[0] + "#include <utils/logging/logger.hpp>\n#include" + parts[1]
            else:
                content = "#include <utils/logging/logger.hpp>\n" + content

        lines = content.split('\n')
        out_lines = []
        for line in lines:
            if "std::cout" in line or "std::cerr" in line:
                if "#include" in line or "using" in line or "default_stream" in line:
                    # comment out default stream assignment
                    if "default_stream = &std::cout;" in line:
                        line = line.replace("&std::cout", "nullptr")
                    else:
                        out_lines.append(line)
                        continue
                line = line.replace("std::cout <<", "TREX_INFO(")
                line = line.replace("std::cerr <<", "TREX_WARN(")
                
                if "TREX_INFO(" in line or "TREX_WARN(" in line:
                    if line.rstrip().endswith(";"):
                        line = line.rstrip()[:-1] + ");"
            out_lines.append(line)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(out_lines))

targets = [
    "src/utils/eval_metrics/utils_eval_cdiagnostics.hpp",
    "src/utils/openmp/utils_openmp.hpp"
]
for t in targets:
    replace_in_file(t)
