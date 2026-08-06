My question: the code find_intK_baye_opt_planck_bounds.py, which want to find the parameter set, which allowedK_integer is most close to a integer sequence with spacing=nu_spacing (minimize integer_loss). Since calculating compute_U_matrices() require solving ODEs for all k modes indepently, which is really time consuming, I use openmp parallization and run it on HPC with one cpu node (with 56 cores). I am now considering to use MPI for the whole code, such that I can calculate compute_U_matrices() for different parameter sets parallely. Do you think BayesianOptimization is the best function to use? Or there are better option, which can help me find the exact parameter set (note that there might be several local minimum), and can easily speed up via mpi?


Answer from Gemini-3.1-pro:

Based on your code and the nature of your objective function, **Bayesian Optimization (specifically the `bayes_opt` package) is likely NOT the best choice for your current setup**, for several reasons.

Here is a breakdown of why, followed by the best alternatives for finding the exact parameter set and easily scaling via MPI on your HPC.

### Why standard Bayesian Optimization (`bayes_opt`) is struggling here:
1. **Inherently Sequential**: `bayes_opt` decides the *next* single point to evaluate based on all previous points. It does not natively support batching or asynchronous parallel evaluations. If you run it with MPI, all your MPI workers will end up waiting for the master node to compute the next point.
2. **Discontinuous Objective Landscape**: Your loss function returns `float('inf')` under certain conditions and trims array boundaries using thresholds. This creates sharp cliffs and discontinuities in the parameter space. Standard Bayesian Optimization relies on Gaussian Processes (GPs), which mathematically assume a smooth, continuous landscape. GPs perform poorly when encountering sudden `inf` values or step-like changes.
3. **Exploration vs. Fine-tuning**: BO is great for getting *close* to the global minimum, but it is notoriously slow at fine-tuning to find the *exact* local minimum compared to local derivative-free optimizers.

---

### The Best Alternatives

To evaluate multiple parameter sets in parallel via MPI, handle non-smooth surfaces, and find the *exact* minimum, I recommend **Differential Evolution (DE)** or **Optuna (TPE)**. 

#### Option 1: `scipy.optimize.differential_evolution` with `mpi4py` (Highly Recommended)
Differential Evolution is a genetic algorithm. It evaluates a "population" of parameters at once, meaning you can flawlessly distribute an entire generation across your HPC using MPI. 

* **Handles discontinuities:** It doesn't use derivatives or assume smoothness, so it handles `inf` returns beautifully.
* **Finds the exact minimum:** Scipy's DE has a `polish=True` argument. Once it finds the global basin, it automatically runs a local search (like Nelder-Mead) to "snap" to the exact local minimum.
* **Trivial MPI Integration:** You can drop in `mpi4py.futures.MPIPoolExecutor`.

Here is how you modify your script to use it:

```python
import numpy as np
from scipy.optimize import differential_evolution
# Make sure to install mpi4py: pip install mpi4py
from mpi4py.futures import MPIPoolExecutor

# ...[Keep your previous imports and function definitions here] ...

# 1. Wrap your loss function to accept a 1D array (required by scipy)
def objective_wrapper(x):
    mt, kt, Omegab_ratio, h = x
    params = [mt, kt, Omegab_ratio, h]
    
    # Optional: Create a specific folder for each worker based on MPI rank
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    folder_path = f'{data_folder}/worker_{rank}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    loss_integer = calculate_integer_loss(params, folder_path)
    
    # DE requires a high float instead of 'inf' to avoid math errors in scipy
    if loss_integer == float('inf') or np.isnan(loss_integer):
        return 1e10 
        
    return loss_integer

if __name__ == "__main__":
    # 2. Define bounds in Scipy format
    # pbounds = {'mt': (350, 500), 'kt': (1.2, 1.8), ...}
    bounds = [
        pbounds['mt'], 
        pbounds['kt'], 
        pbounds['Omegab_ratio'], 
        pbounds['h']
    ]

    print("Starting Differential Evolution via MPI...")
    
    # 3. Use MPIPoolExecutor to parallelize evaluations
    with MPIPoolExecutor() as executor:
        result = differential_evolution(
            objective_wrapper, 
            bounds, 
            strategy='best1bin',
            maxiter=50,
            popsize=15,          # Population size = popsize * len(bounds) = 15 * 4 = 60 points per generation
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.7,
            updating='deferred', # CRITICAL for MPI: Wait for all workers before updating the population
            workers=executor.map, # Passes the MPI workers to Scipy
            polish=True,         # CRITICAL: Runs a local optimizer at the end to find the exact minimum!
            disp=True            # Print progress
        )

    print("\nOPTIMIZATION COMPLETE")
    print(f"Best parameters: {result.x}")
    print(f"Minimum loss: {result.fun}")
```

#### Option 2: Optuna (Asynchronous Distributed Optimization)
If you really want to keep the probabilistic approach of Bayesian Optimization, switch from `bayes_opt` to **Optuna**. Optuna uses the Tree-structured Parzen Estimator (TPE) algorithm, which handles discrete jumps and `inf` values much better than GPs. 

More importantly, Optuna allows **asynchronous parallelism**. You don't use `mpi4py` directly; instead, you spin up your script using MPI (`mpirun -np N python script.py`), and all workers communicate via a shared database file (like SQLite or JournalStorage) to pick up the next parameters dynamically.

```python
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

def optuna_objective(trial):
    mt = trial.suggest_float('mt', pbounds['mt'][0], pbounds['mt'][1])
    kt = trial.suggest_float('kt', pbounds['kt'][0], pbounds['kt'][1])
    Omegab_ratio = trial.suggest_float('Omegab_ratio', pbounds['Omegab_ratio'][0], pbounds['Omegab_ratio'][1])
    h = trial.suggest_float('h', pbounds['h'][0], pbounds['h'][1])
    
    params = [mt, kt, Omegab_ratio, h]
    
    import multiprocessing
    folder_path = f'{data_folder}/try_{trial.number}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    loss = calculate_integer_loss(params, folder_path)
    if loss == float('inf'):
        raise optuna.TrialPruned() # Tells Optuna this point is invalid
        
    return loss

if __name__ == "__main__":
    # JournalStorage uses an append-only file, safe for parallel HPC workers without setting up a database server
    storage = JournalStorage(JournalFileStorage("./optuna_shared_journal.log"))
    
    # Create or load the study
    study = optuna.create_study(
        study_name="planck_integer_K",
        storage=storage,
        load_if_exists=True,
        direction="minimize"
    )
    
    # Every MPI worker will run this block independently and asynchronously
    study.optimize(optuna_objective, n_trials=50)
```

### HPC Execution Strategy (MPI + OpenMP Hybrid)

Since calculating `compute_U_matrices()` is already OpenMP parallelized (intra-node), you want a **Hybrid MPI + OpenMP** setup.

If you have 1 Node with 56 cores, you shouldn't launch 56 MPI tasks. Instead, map it effectively. For example:
* Launch **8 MPI tasks**.
* Give each MPI task **7 OpenMP threads** (`export OMP_NUM_THREADS=7`).
* $8 \times 7 = 56$ cores.

This allows you to evaluate 8 different parameter sets simultaneously, while the ODE solver for each parameter set finishes 7 times faster using OpenMP.

**Summary Recommendation:** Use **Option 1 (`scipy.optimize.differential_evolution`)**. It explicitly addresses your request to escape local minima, natively uses MPI without external database dependencies, and the `polish=True` flag specifically solves your need to hone in on the *exact* parameter set at the end of the run.