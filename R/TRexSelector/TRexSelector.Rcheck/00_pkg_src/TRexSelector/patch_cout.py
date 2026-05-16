import os
import re

files_to_patch = [
    "src/tsolvers/tsolver_base.cpp",
    "src/trex_selector_methods/trex_core/trex.cpp",
    "src/trex_selector_methods/trex_screening/trex_biobank_screening.cpp"
]

for path in files_to_patch:
    with open(path, "r") as f:
        content = f.read()

    # Wrap std::cout in tsolver_base.cpp
    if path.endswith("tsolver_base.cpp"):
        content = content.replace(
            "void TSolver_Base::logMsg(const std::string& msg) const { if (verbose_) std::cout << msg << '\\n'; }",
            "void TSolver_Base::logMsg(const std::string& msg) const {\n#ifndef CRAN_BUILD\n    if (verbose_) std::cout << msg << '\\n';\n#endif // CRAN_BUILD\n}"
        )

    # Wrap basic logging in trex.cpp
    if path.endswith("trex.cpp"):
        content = content.replace(
            "        std::cout << \"[T-Rex Selector] \" << message << \"\\n\";",
            "#ifndef CRAN_BUILD\n        std::cout << \"[T-Rex Selector] \" << message << \"\\n\";\n#endif // CRAN_BUILD"
        )
        content = content.replace(
            "            std::cout << \"T-Rex Selector: [DEBUG]\" << (msg) << std::endl;",
            "#ifndef CRAN_BUILD\n            std::cout << \"T-Rex Selector: [DEBUG]\" << (msg) << std::endl;\n#endif // CRAN_BUILD"
        )
        
    if path.endswith("trex_biobank_screening.cpp"):
        content = re.sub(r'(std::cout[^\;]*;)', r'#ifndef CRAN_BUILD\n\1\n#endif // CRAN_BUILD', content)

    with open(path, "w") as f:
        f.write(content)
