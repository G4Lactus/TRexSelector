#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
lldb --batch -o "run" -o "bt" -o "quit" -- /opt/homebrew/Caskroom/miniconda/base/bin/python3 test_final.py
