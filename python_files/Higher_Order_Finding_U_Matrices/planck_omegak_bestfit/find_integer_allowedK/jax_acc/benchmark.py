import os
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

# Import the original and JAX functions
# Adjust the import names if your files are named differently
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_U_Matrices_Jax import compute_U_matrices_Jax

def check_difference(name, arr_scipy, arr_jax):
    """Helper to compute and print the maximum difference between two arrays."""
    abs_diff = np.abs(arr_scipy - arr_jax)
    max_err = np.max(abs_diff)
    
    # Avoid division by zero for relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / np.abs(arr_scipy)
        rel_diff = np.nan_to_num(rel_diff, nan=0.0, posinf=0.0, neginf=0.0)
    
    max_rel_err = np.max(rel_diff)
    
    print(f"  {name+':':<12} Max Abs Err: {max_err:.2e}  |  Max Rel Err: {max_rel_err:.2e}")
    return max_err

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # 1. Setup Parameters and Output Folders
    # -------------------------------------------------------------------------
    # Standard cosmological parameters from your earlier snippet
    params =[409.969398, 1.459351, 0.163514, 0.547313]
    z_rec = 1100.0
    
    scipy_dir = './out_scipy/'
    jax_dir = './out_jax/'
    
    os.makedirs(scipy_dir, exist_ok=True)
    os.makedirs(jax_dir, exist_ok=True)

    print("="*60)
    print("🚀 STARTING BENCHMARK: SCIPY (Original) vs JAX")
    print("="*60)

    # -------------------------------------------------------------------------
    # 2. Benchmark Original SciPy (Multiprocessing)
    # -------------------------------------------------------------------------
    print("\n[1/3] Running Original SciPy Implementation...")
    t0_scipy = time.perf_counter()
    compute_U_matrices(params, z_rec, scipy_dir, kvalues=np.linspace(10, 15, num=10))
    t_scipy = time.perf_counter() - t0_scipy
    print(f"✅ Original SciPy Time: {t_scipy:.3f} seconds")

    # -------------------------------------------------------------------------
    # 3. Benchmark JAX (First run includes JIT compilation)
    # -------------------------------------------------------------------------
    print("\n[2/3] Running JAX Implementation (Cold Start / Compiling)...")
    t0_jax_cold = time.perf_counter()
    compute_U_matrices_Jax(params, z_rec, jax_dir, kvalues=np.linspace(10, 15, num=10))
    t_jax_cold = time.perf_counter() - t0_jax_cold
    print(f"✅ JAX Cold Time (Execution + JIT Compile): {t_jax_cold:.3f} seconds")

    # -------------------------------------------------------------------------
    # 4. Benchmark JAX (Second run - pure execution speed)
    # -------------------------------------------------------------------------
    print("\n[3/3] Running JAX Implementation (Warm Start / Pure Execution)...")
    t0_jax_warm = time.perf_counter()
    compute_U_matrices_Jax(params, z_rec, jax_dir, kvalues=np.linspace(10, 15, num=10))
    t_jax_warm = time.perf_counter() - t0_jax_warm
    print(f"✅ JAX Warm Time (Pure Execution): {t_jax_warm:.3f} seconds")

    # -------------------------------------------------------------------------
    # 5. Accuracy / Correctness Comparison
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("📊 ACCURACY COMPARISON")
    print("="*60)
    
    files_to_check =[
        'L70_ABCmatrices.npy', 
        'L70_DEFmatrices.npy', 
        'L70_GHIvectors.npy', 
        'L70_X1matrices.npy', 
        'L70_X2matrices.npy'
    ]
    
    all_passed = True
    for fname in files_to_check:
        scipy_path = os.path.join(scipy_dir, fname)
        jax_path = os.path.join(jax_dir, fname)
        
        arr_scipy = np.load(scipy_path)
        arr_jax = np.load(jax_path)
        
        matrix_name = fname.replace('L70_', '').replace('.npy', '')
        max_err = check_difference(matrix_name, arr_scipy, arr_jax)
        
        # Check if tolerance is met (e.g., 1e-6 relative to integration drift)
        if max_err > 1e-5:
            all_passed = False
            
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS")
    print("="*60)
    print(f"SciPy total time : {t_scipy:.3f} s")
    print(f"JAX total time   : {t_jax_warm:.3f} s")
    print(f"Speedup          : {t_scipy / t_jax_warm:.2f}x faster")
    
    if all_passed:
        print("\n✅ ACCURACY TEST PASSED! The outputs match closely.")
    else:
        print("\n⚠️ WARNING: Noticeable differences in outputs. Check tolerance levels.")