import os
import numpy as np
import time
from multiprocessing import Pool

start_time = time.time()

data_dir = "data/diff_m/"
T = np.pi
L = 2.0 * np.pi
eigenvalues_threshold = 0.05
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))  # controlled by --cpus-per-task in the SLURM script
kappa_list = np.linspace(0.01, 1, num=100)


def inner_product(f, g):
    return np.vdot(f, g)


def gram_schmidt(functions):
    N = len(functions)
    orthogonal_functions = [functions[0].copy()]
    for i in range(1, N):
        new_function = functions[i].copy()
        for j in range(i):
            proj = inner_product(orthogonal_functions[j], functions[i]) / inner_product(orthogonal_functions[j], orthogonal_functions[j])
            new_function = new_function - proj * orthogonal_functions[j]
        orthogonal_functions.append(new_function)

    transformation_matrix = np.zeros((N, N), dtype=complex)
    transformation_matrix[0, 0] = 1
    for i in range(1, N):
        transformation_matrix[i, i] = 1
        for m in range(i):
            for j in range(m, i):
                transformation_matrix[i, m] -= (
                    inner_product(orthogonal_functions[j], functions[i]) /
                    inner_product(orthogonal_functions[j], orthogonal_functions[j]) *
                    transformation_matrix[j, m]
                )

    return orthogonal_functions, transformation_matrix.real


def normalization(orthogonal_functions, transformation_matrix):
    N = len(orthogonal_functions)
    norms = [np.sqrt(inner_product(f, f).real) for f in orthogonal_functions]
    orthonormal_functions = [f / norms[i] for i, f in enumerate(orthogonal_functions)]
    for i in range(N):
        for j in range(N):
            transformation_matrix[i, j] /= norms[i]
    return orthonormal_functions, transformation_matrix


def compute_A_matrix(orthonormal_functions_1, orthonormal_functions_2):
    N = len(orthonormal_functions_1)
    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            M[i, j] = inner_product(orthonormal_functions_1[i], orthonormal_functions_2[j])
    M = np.asmatrix(M)
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(np.dot(M, M.getH()))
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(np.dot(M.getH(), M))
    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2


def choose_eigenvalues(eigenvalues, threshold):
    return [i for i in range(len(eigenvalues)) if np.abs(eigenvalues[i] - 1.0) < threshold]


def solve_for_kappa(kappa, N, Nt):
    t_list = np.linspace(0, T, Nt)
    x_list = np.linspace(0, L, Nt)
    t, x = np.meshgrid(t_list, x_list)

    basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j * k * x) for k in range(1, N + 1)]

    # start_n ensures (n*pi/T)^2 > kappa^2 so the x-wavenumber is real
    start_n = int(kappa) + 1
    basis_2 = [
        np.cos(n * np.pi / T * t) * np.exp(1j * np.sqrt((n * np.pi / T)**2 - kappa**2) * x)
        for n in range(start_n, N + start_n)
    ]

    orthogonal_functions_1, transformation_matrix_1 = gram_schmidt(basis_1)
    orthogonal_functions_2, transformation_matrix_2 = gram_schmidt(basis_2)

    orthonormal_functions_1, transformation_matrix_1 = normalization(orthogonal_functions_1, transformation_matrix_1)
    orthonormal_functions_2, transformation_matrix_2 = normalization(orthogonal_functions_2, transformation_matrix_2)

    eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2 = compute_A_matrix(
        orthonormal_functions_1, orthonormal_functions_2
    )

    eigen_valid_idx_1 = choose_eigenvalues(eigenvalues_1, eigenvalues_threshold)
    eigenvalues_valid_1 = eigenvalues_1[eigen_valid_idx_1]
    eigenvectors_valid_1 = eigenvectors_1[:, eigen_valid_idx_1]

    eigen_valid_idx_2 = choose_eigenvalues(eigenvalues_2, eigenvalues_threshold)
    eigenvalues_valid_2 = eigenvalues_2[eigen_valid_idx_2]
    eigenvectors_valid_2 = eigenvectors_2[:, eigen_valid_idx_2]

    coefficients_1 = np.dot(eigenvectors_valid_1.T, transformation_matrix_1)
    coefficients_2 = np.dot(eigenvectors_valid_2.T, transformation_matrix_2)

    tag = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.3f}"
    np.savetxt(os.path.join(data_dir, f"all_eigenvalues1_2d_{tag}.txt"), eigenvalues_1)
    np.savetxt(os.path.join(data_dir, f"all_eigenvalues2_2d_{tag}.txt"), eigenvalues_2)
    np.savetxt(os.path.join(data_dir, f"eigenvalues1_2d_{tag}.txt"), eigenvalues_valid_1)
    np.savetxt(os.path.join(data_dir, f"eigenvalues2_2d_{tag}.txt"), eigenvalues_valid_2)
    np.savetxt(os.path.join(data_dir, f"coefficients1_2d_{tag}.txt"), coefficients_1)
    np.savetxt(os.path.join(data_dir, f"coefficients2_2d_{tag}.txt"), coefficients_2)

    print(f"  saved: {tag}  valid_1={len(eigenvalues_valid_1)}  valid_2={len(eigenvalues_valid_2)}")


def _worker(kappa):
    N = max(20, int(kappa*200))
    Nt = 15*N

    tag = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.3f}"
    out_file = os.path.join(data_dir, f"coefficients1_2d_{tag}.txt")
    if os.path.exists(out_file):
        print(f"kappa={kappa:.3f}: already computed, skipping", flush=True)
        return
    print(f"kappa={kappa:.3f}: N={N}, Nt={Nt} ...", flush=True)
    t0 = time.time()
    solve_for_kappa(kappa, N, Nt)
    print(f"  kappa={kappa:.3f} elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    with Pool(processes=num_workers) as pool:
        pool.map(_worker, kappa_list)
    print(f"\nTotal elapsed: {time.time() - start_time:.1f}s")


