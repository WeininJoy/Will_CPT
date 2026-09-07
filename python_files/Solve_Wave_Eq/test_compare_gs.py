"""
Compare eigenvalues from the no-GS and with-GS algorithms using
identical kappa, N, Nt, and grid for each run.
"""
import numpy as np

T = np.pi
L = 2.0 * np.pi
threshold = 0.05   # match withGS threshold so valid sets are comparable

kappa_list = [0.1, 0.2, 0.3]
N  = 40
Nt = 600


# ── helpers shared by both algorithms ─────────────────────────────────────────

def make_grids(Nt):
    x_list = np.linspace(0, L, Nt, endpoint=False)
    dt     = T / Nt
    t_list = np.linspace(dt / 2, T - dt / 2, Nt)
    return np.meshgrid(t_list, x_list)   # t, x  (shape Nt×Nt)


def build_basis1(kappa, N, t, x):
    flat = t.size
    F = np.empty((N, flat), dtype=complex)
    inv_norm = np.empty(N)
    for i, k in enumerate(range(1, N + 1)):
        f = (np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j * k * x)).ravel()
        norm = np.sqrt(np.vdot(f, f).real)
        inv_norm[i] = 1.0 / norm
        F[i] = f / norm
    return F, inv_norm


def build_basis2_raw(kappa, N, t, x):
    flat    = t.size
    start_n = int(kappa) + 1
    F = np.empty((N, flat), dtype=complex)
    for i, n in enumerate(range(start_n, N + start_n)):
        kx   = np.sqrt((n * np.pi / T)**2 - kappa**2)
        F[i] = (np.cos(n * np.pi / T * t) * np.exp(1j * kx * x)).ravel()
    return F


def gram_schmidt_rows(F):
    N = F.shape[0]
    Q = np.empty_like(F)
    Tm = np.zeros((N, N), dtype=complex)
    for i in range(N):
        q = F[i].copy()
        Tm[i, i] = 1.0
        for j in range(i):
            coeff = np.vdot(Q[j], q)
            q -= coeff * Q[j]
            Tm[i] -= coeff * Tm[j]
        norm = np.sqrt(np.vdot(q, q).real)
        Q[i] = q / norm
        Tm[i] /= norm
    return Q, Tm


def choose_valid(eigenvalues, threshold):
    return [i for i in range(len(eigenvalues)) if abs(eigenvalues[i] - 1.0) < threshold]


# ── algorithm A: no GS on F2 (original broken version) ────────────────────────

def algo_noGS(kappa, N, Nt):
    t, x = make_grids(Nt)
    F1, inv_norm1 = build_basis1(kappa, N, t, x)

    F2_raw = build_basis2_raw(kappa, N, t, x)
    # individually normalize only (no GS)
    inv_norm2 = np.empty(N)
    F2 = np.empty_like(F2_raw)
    for i in range(N):
        norm = np.sqrt(np.vdot(F2_raw[i], F2_raw[i]).real)
        inv_norm2[i] = 1.0 / norm
        F2[i] = F2_raw[i] / norm

    M = F1.conj() @ F2.T
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    eigenvalues = S**2
    return eigenvalues[choose_valid(eigenvalues, threshold)]


# ── algorithm B: with GS on F2 (updated solve_pde_diff_kappa.py) ───────────────

def algo_withGS(kappa, N, Nt):
    t, x = make_grids(Nt)
    F1, inv_norm1 = build_basis1(kappa, N, t, x)

    F2_raw = build_basis2_raw(kappa, N, t, x)
    F2, T2 = gram_schmidt_rows(F2_raw)

    M = F1.conj() @ F2.T
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    eigenvalues = S**2
    return eigenvalues[choose_valid(eigenvalues, threshold)]


# ── algorithm C: withGS reference (solve_pde_diff_kappa_withGS.py logic) ───────
# Uses linspace(0, T, Nt) / linspace(0, L, Nt) grids and eig instead of SVD.

def algo_reference(kappa, N, Nt):
    t_list = np.linspace(0, T, Nt)
    x_list = np.linspace(0, L, Nt)
    t, x   = np.meshgrid(t_list, x_list)

    basis1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j * k * x)
              for k in range(1, N + 1)]
    start_n = int(kappa) + 1
    basis2 = [np.cos(n * np.pi / T * t) * np.exp(1j * np.sqrt((n * np.pi / T)**2 - kappa**2) * x)
              for n in range(start_n, N + start_n)]

    def gs_list(funcs):
        orth = [funcs[0].copy()]
        for i in range(1, len(funcs)):
            q = funcs[i].copy()
            for v in orth:
                q -= (np.vdot(v, funcs[i]) / np.vdot(v, v)) * v
            orth.append(q)
        norms = [np.sqrt(np.vdot(f, f).real) for f in orth]
        return [f / n for f, n in zip(orth, norms)]

    on1 = gs_list(basis1)
    on2 = gs_list(basis2)

    M = np.array([[np.vdot(on1[i], on2[j]) for j in range(N)] for i in range(N)])
    ev1 = np.linalg.eigvalsh(M @ M.conj().T)[::-1]   # eigvalsh: real, sorted
    return ev1[choose_valid(ev1, threshold)]


# ── run and compare ────────────────────────────────────────────────────────────

print(f"N={N}, Nt={Nt}\n")
print(f"{'kappa':>6}  {'noGS valid_ev':>35}  {'withGS valid_ev':>35}  {'reference valid_ev':>35}")
print("-" * 120)

for kappa in kappa_list:
    ev_no  = np.sort(algo_noGS   (kappa, N, Nt))[::-1]
    ev_gs  = np.sort(algo_withGS (kappa, N, Nt))[::-1]
    ev_ref = np.sort(algo_reference(kappa, N, Nt))[::-1]

    print(f"\nkappa={kappa}")
    print(f"  no-GS   ({len(ev_no):2d} valid): {np.array2string(ev_no,  precision=6, separator=', ')}")
    print(f"  with-GS ({len(ev_gs):2d} valid): {np.array2string(ev_gs,  precision=6, separator=', ')}")
    print(f"  ref     ({len(ev_ref):2d} valid): {np.array2string(ev_ref, precision=6, separator=', ')}")

    if len(ev_gs) == len(ev_ref) and len(ev_gs) > 0:
        diff = np.max(np.abs(ev_gs - ev_ref))
        print(f"  max |with-GS - ref| = {diff:.2e}")
    else:
        print(f"  count mismatch: with-GS {len(ev_gs)} vs ref {len(ev_ref)}")
