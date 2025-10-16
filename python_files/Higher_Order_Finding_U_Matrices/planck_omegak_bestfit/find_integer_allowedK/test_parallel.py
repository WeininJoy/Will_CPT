#!/usr/bin/env python3
"""
Test script for the parallelized Higher_Order_Finding_U_Matrices.py

This script demonstrates how to use the parallelized version of compute_U_matrices.
Results are written to a log file instead of printing to console.
"""

from Higher_Order_Finding_U_Matrices.planck_omegak_bestfit.find_integer_allowedK.Higher_Order_Finding_U_Matrices import compute_U_matrices
import time
import os
from datetime import datetime

# Example parameters (adjust these to match your actual use case)
# Format: [mt, kt, omega_b_ratio, h]
params = [401.38626259929055, 1.4181566171960542, 0.16686454899542, 0.5635275092831583]
z_rec = 1089.8  # Recombination redshift
folder_path = './data/test_parallel_output/'  # Output directory

# Create output directory if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Open log file for writing
log_filename = os.path.join(folder_path, f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
with open(log_filename, 'w') as log_file:

    def log(message):
        """Write message to both console and file"""
        print(message)
        log_file.write(message + '\n')
        log_file.flush()  # Ensure immediate write

    log("=" * 60)
    log("Testing Parallelized U Matrices Computation")
    log(f"Log file: {log_filename}")
    log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # Test 1: Using all available cores (default)
    log("\nTest 1: Using all available CPU cores")
    log("-" * 60)
    start_time = time.time()
    try:
        result = compute_U_matrices(params, z_rec, folder_path, nu_spacing=4, n_processes=None)
        elapsed = time.time() - start_time
        log(f"Computation completed in {elapsed:.2f} seconds")
    except Exception as e:
        log(f"Error: {e}")

    # Test 2: Using specific number of processes
    log("\nTest 2: Using 4 processes")
    log("-" * 60)
    start_time = time.time()
    try:
        result = compute_U_matrices(params, z_rec, folder_path, nu_spacing=4, n_processes=4)
        elapsed = time.time() - start_time
        log(f"Computation completed in {elapsed:.2f} seconds")
    except Exception as e:
        log(f"Error: {e}")

    # Test 3: Single process (for comparison)
    log("\nTest 3: Using 1 process (sequential, for comparison)")
    log("-" * 60)
    start_time = time.time()
    try:
        result = compute_U_matrices(params, z_rec, folder_path, nu_spacing=4, n_processes=1)
        elapsed = time.time() - start_time
        log(f"Computation completed in {elapsed:.2f} seconds")
    except Exception as e:
        log(f"Error: {e}")

    log("\n" + "=" * 60)
    log("Testing completed")
    log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

print(f"\nResults written to: {log_filename}")
