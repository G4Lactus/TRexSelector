#!/bin/bash
sed -i '' 's/R_useDynamicSymbols(dll, FALSE)/R_useDynamicSymbols(dll, (Rboolean)FALSE)/g' src/RcppExports.cpp
