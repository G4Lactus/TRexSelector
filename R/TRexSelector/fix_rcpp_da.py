import re

with open("src/rcpp_trex_selector.cpp", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "params.BT_grid_size = control" in line:
        continue
    if "params.BT_min_rho_index = control" in line:
        continue
    if "if (control.containsElementNamed(\"BT_grid_size\"))" in line:
        line = line.replace("BT_grid_size", "hc_grid_length")
        new_lines.append(line)
        continue
    new_lines.append(line)

with open("src/rcpp_trex_selector.cpp", "w") as f:
    f.writelines(new_lines)
