with open("src/demos/trex_selector_methods/trex_gvs/sim_trex_gvs_13_hapgen.cpp", "r") as f:
    text = f.read()

import re

# Find multiline unescaped quotes with n= and fix.
text = re.sub(r'"\n\s*?n=', '"\\\\nn=', text)
text = re.sub(r'MC trials...\\n"\n\s*?<< std::flush', 'MC trials...\\n" << std::flush', text)
text = re.sub(r'MC trials...\n"\n\s*?<< std::flush', 'MC trials...\\n" << std::flush', text)

with open("src/demos/trex_selector_methods/trex_gvs/sim_trex_gvs_13_hapgen.cpp", "w") as f:
    f.write(text)
print("Done")
