#!/bin/bash

# Define the Python files to run
python_file1="Higher_Order_Finding_U_Matrices.py"
python_file2="Higher_Order_Finding_Xrecs.py"
python_file3="Higher_Order_Solving_for_Vrinf.py"

# Loop from 0 to 99
for i in $(seq 0 7); do
  echo "Running $python_file with input: $i"
  python3 "$python_file3" "$i"
  echo "--------------------"
done