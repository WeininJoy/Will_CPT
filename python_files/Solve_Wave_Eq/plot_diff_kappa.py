import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

eigenvalues_threshold = 0.99
data_dir = "data/diff_m/"
T = np.pi
L = 2.0 * np.pi
kappa_list = np.linspace(0.025, 1, num=40)
N_plot = 4
N_t_plot = 300
N_x_plot = 300

os.makedirs("figures/diff_m", exist_ok=True)


def get_N_Nt(kappa):
    return 30, 1000


def load_data(kappa):
    N, Nt = get_N_Nt(kappa)
    tag = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.2f}"
    eig1  = np.loadtxt(f"{data_dir}eigenvalues1_2d_{tag}.txt",  dtype=np.complex128)
    coef1 = np.loadtxt(f"{data_dir}coefficients1_2d_{tag}.txt", dtype=np.complex128)
    eig2  = np.loadtxt(f"{data_dir}eigenvalues2_2d_{tag}.txt",  dtype=np.complex128)
    coef2 = np.loadtxt(f"{data_dir}coefficients2_2d_{tag}.txt", dtype=np.complex128)
    if eig1.ndim  == 0: eig1  = eig1[np.newaxis]
    if coef1.ndim == 1: coef1 = coef1[np.newaxis, :]
    if eig2.ndim  == 0: eig2  = eig2[np.newaxis]
    if coef2.ndim == 1: coef2 = coef2[np.newaxis, :]
    return N, Nt, eig1, coef1, eig2, coef2


def build_solutions(kappa, coef1, coef2, N):
    t_list = np.linspace(0, T, N_t_plot)
    x_list = np.linspace(0, L, N_x_plot)
    t_grid, x_grid = np.meshgrid(t_list, x_list)

    basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t_grid) * np.exp(1j * k * x_grid)
               for k in range(1, N + 1)]

    if (np.pi / T) ** 2 - kappa**2 < 0:
        start_n = round(np.sqrt(abs((np.pi / T) ** 2 - kappa**2))) + 1
    else:
        start_n = 1
    basis_2 = [np.cos(n * np.pi / T * t_grid) * np.exp(1j * np.sqrt((n * np.pi / T) ** 2 - kappa**2) * x_grid)
               for n in range(start_n, N + start_n)]

    n_sol = coef1.shape[0]
    sol1_list, sol2_list = [], []
    for i in range(n_sol):
        sol1 = sum(coef1[i, j] * basis_1[j] for j in range(N))
        sol2 = sum(coef2[i, j] * basis_2[j] for j in range(N))
        sol1_list.append(sol1)
        sol2_list.append(sol2)

    diff_array = np.zeros((n_sol, n_sol))
    for i in range(n_sol):
        for j in range(n_sol):
            norm1 = np.linalg.norm(sol1_list[i])
            diff_array[i, j] = np.linalg.norm(sol1_list[i] - sol2_list[j]) / (norm1 + 1e-30)
    small_diff_idx = [int(np.argmin(diff_array[i])) for i in range(n_sol)]

    max_indices  = np.argmax(coef1.real, axis=1)
    sorted_index = np.argsort(max_indices).tolist()

    return t_grid, x_grid, t_list, x_list, sol1_list, sol2_list, small_diff_idx, sorted_index, max_indices


# # ── Figure 1: Phi(t, x=L/2) overlay for m=0.1~1.0 ───────────────────────────

# kappa_list_fig1 = kappa_list[kappa_list <= 1]
# fig, ax = plt.subplots(figsize=(8, 5))
# cmap_v = plt.get_cmap("viridis")
# colors = cmap_v(np.linspace(0, 1, len(kappa_list_fig1)))

# for idx, kappa in enumerate(kappa_list_fig1):
#     N, Nt, eig1, coef1, eig2, coef2 = load_data(kappa)

#     mask      = eig1.real > eigenvalues_threshold
#     coef1_f   = coef1[mask]
#     if len(coef1_f) == 0:
#         print(f"kappa={kappa:.3f}: no eigenvalue > {eigenvalues_threshold}, skipping")
#         continue

#     dom_k_idx  = np.argmax(np.abs(coef1_f.real), axis=1)
#     chosen     = np.argmin(dom_k_idx)
#     chosen_c   = coef1_f[chosen]
#     chosen_dom = dom_k_idx[chosen] + 1

#     t_list  = np.linspace(0, T, N_t_plot)
#     x_half  = L / 2.0
#     phi_t   = np.zeros(N_t_plot, dtype=complex)
#     for j in range(N):
#         k     = j + 1
#         omega = np.sqrt(k**2 + kappa**2)
#         phi_t += chosen_c[j] * np.cos(omega * t_list) * np.exp(1j * k * x_half)

#     ax.plot(t_list, phi_t.real, color=colors[idx],
#             label=rf"$\mu={kappa:.2f}$, dom $k={chosen_dom}$")

# ax.set_xlabel(r"$t$", fontsize=12)
# ax.set_ylabel(r"$\Phi(t,\, x=L/2)$", fontsize=12)
# ax.set_title(rf"Solutions $\Phi(t,\,x=L/2)$ for each $\mu$ (eigenvalue $> {eigenvalues_threshold}$, lowest dominant $k$)")
# ax.legend(fontsize=7, ncol=2, loc="upper right")
# fig.tight_layout()
# plt.savefig("figures/phi_t_diff_m.pdf")
# plt.savefig("figures/phi_t_diff_m.png", dpi=150)
# print("Saved figures/phi_t_diff_m.pdf and .png")
# plt.close(fig)


# # ── Figure 2: 2×2 contour for selected kappas ────────────────────────────────

# kappa_contour = [0.1, 0.4, 0.7, 1.0]
# cmap_c  = plt.get_cmap("RdGy")
# fig2, axs2 = plt.subplots(2, 2, figsize=(9, 5.5))

# for ax_c, kappa in zip(axs2.flat, kappa_contour):
#     N, Nt, eig1, coef1, eig2, coef2 = load_data(kappa)

#     mask      = eig1.real > eigenvalues_threshold
#     coef1_f   = coef1[mask]
#     dom_k_idx = np.argmax(np.abs(coef1_f.real), axis=1)
#     chosen    = np.argmin(dom_k_idx)
#     chosen_c  = coef1_f[chosen]
#     chosen_dom = dom_k_idx[chosen] + 1

#     t_list   = np.linspace(0, T, N_t_plot)
#     x_list   = np.linspace(0, L, N_x_plot)
#     t_g, x_g = np.meshgrid(t_list, x_list)

#     solution = np.zeros_like(t_g, dtype=complex)
#     for j in range(N):
#         k     = j + 1
#         omega = np.sqrt(k**2 + kappa**2)
#         solution += chosen_c[j] * np.cos(omega * t_g) * np.exp(1j * k * x_g)

#     cf = ax_c.contourf(x_g, t_g, solution.real, 20, cmap=cmap_c)
#     fig2.colorbar(cf, ax=ax_c)
#     ax_c.set_title(rf"$\mu={kappa:.2f}$, dom $k={chosen_dom}$", fontsize=10)
#     ax_c.set_xlabel(r"$x$", fontsize=9)
#     ax_c.set_ylabel(r"$t$", fontsize=9)

# fig2.suptitle(rf"$\Phi(t,x)$ for selected $\mu$ (eigenvalue $> {eigenvalues_threshold}$, lowest dominant $k$)", fontsize=11)
# fig2.tight_layout()
# plt.savefig("figures/phi_contour_diff_m.pdf")
# plt.savefig("figures/phi_contour_diff_m.png", dpi=150)
# print("Saved figures/phi_contour_diff_m.pdf and .png")
# plt.close(fig2)


# ── Figure 3: Lowest dominant k vs mu ────────────────────────────────────────

kappa_vals, dom_k_vals = [], []
for kappa in kappa_list:
    N, Nt, eig1, coef1, eig2, coef2 = load_data(kappa)
    mask = eig1.real > eigenvalues_threshold
    coef1_f = coef1[mask]
    if len(coef1_f) == 0:
        print(f"kappa={kappa:.2f}: no eigenvalue > {eigenvalues_threshold}, skipping")
        continue
    dom_k_idx = np.argmax(np.abs(coef1_f.real), axis=1)
    chosen_dom = int(dom_k_idx[np.argmin(dom_k_idx)]) + 1
    kappa_vals.append(kappa)
    dom_k_vals.append(chosen_dom)

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(kappa_vals, dom_k_vals, "o-", color="steelblue", markersize=6)
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.set_xlabel(r"$\mu$", fontsize=13)
ax3.set_ylabel(r"Lowest dominant $k$", fontsize=13)
ax3.set_title(rf"Lowest dominant $k$ vs $\mu$ (eigenvalue $> {eigenvalues_threshold}$)", fontsize=11)
ax3.grid(True, linestyle="--", alpha=0.5)
fig3.tight_layout()
plt.savefig("figures/dom_k_vs_mu.pdf")
plt.savefig("figures/dom_k_vs_mu.png", dpi=150)
print("Saved figures/dom_k_vs_mu.pdf and .png")
plt.close(fig3)


# # ── Per-kappa figures (all figure types from plot_solution_wave_eq.py) ────────

# for kappa in kappa_list[kappa_list <= 1]:
#     N, Nt, eig1, coef1, eig2, coef2 = load_data(kappa)
#     tag    = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.2f}"
#     n_sol  = coef1.shape[0]
#     n_plot = min(N_plot, n_sol)

#     t_grid, x_grid, t_list, x_list, sol1_list, sol2_list, small_diff_idx, sorted_idx, max_indices = \
#         build_solutions(kappa, coef1, coef2, N)

#     cmap_rdgy = plt.get_cmap("RdGy")

#     # Figure A: Coefficients bar plot
#     # Determine xlim: rightmost k with normalized coefficient > threshold across all plotted solutions
#     coef_threshold = 0.05
#     k_max = 1
#     for i in range(n_plot):
#         coefs_norm = coef1[sorted_idx[i]].real
#         coefs_norm = coefs_norm / (np.max(np.abs(coefs_norm)) + 1e-30)
#         hits = np.where(np.abs(coefs_norm) > coef_threshold)[0]
#         if len(hits) > 0:
#             k_max = max(k_max, hits[-1] + 1)  # hits are 0-based; +1 gives 1-based k

#     fig, axs = plt.subplots(n_plot, figsize=(3.8, 0.7 * n_plot))
#     if n_plot == 1:
#         axs = [axs]
#     fig.suptitle(rf"Coefficients for linear combination, $\mu={kappa:.2f}$", fontsize=10)
#     for i in range(n_plot):
#         idx   = sorted_idx[i]
#         coefs = coef1[idx].real
#         coefs = coefs / (np.max(np.abs(coefs)) + 1e-30)
#         axs[i].bar(range(1, N + 1), coefs.tolist(), width=0.2)
#         axs[i].set_xlim(0, k_max + 1)
#         axs[i].set_ylim(-1, 1.0)
#         axs[i].xaxis.set_tick_params(labelsize=8)
#         axs[i].yaxis.set_tick_params(labelsize=8)
#         axs[i].label_outer()
#     axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
#     axs[-1].set_xlabel(r"$k$", fontsize=8)
#     fig.tight_layout()
#     plt.savefig(f"figures/diff_m/eigenvalue_2d_{tag}_coefficients.pdf")
#     plt.close(fig)

#     # Figure B: Phi(t, x=L/2) – basis_1 vs basis_2
#     fig, axs = plt.subplots(n_plot, figsize=(6, 0.8 * n_plot))
#     if n_plot == 1:
#         axs = [axs]
#     fig.suptitle(rf"$\Phi(t,\,x=L/2)$, $\mu={kappa:.2f}$")
#     for i in range(n_plot):
#         idx = sorted_idx[i]
#         s1  = sol1_list[idx]
#         s2  = sol2_list[small_diff_idx[idx]]
#         axs[i].plot(t_list, s1[N_x_plot // 2, :].real, color="r", label=r"basis: $\phi$")
#         axs[i].plot(t_list, s2[N_x_plot // 2, :].real, color="g", linestyle="dashed",
#                     label=r"basis: $\tilde\phi$")
#         axs[i].label_outer()
#     axs[-1].legend(fontsize=7)
#     axs[-1].set_xlabel(r"$t$")
#     fig.tight_layout()
#     plt.savefig(f"figures/diff_m/eigenvalue_2d_{tag}_xpi.pdf")
#     plt.close(fig)

#     # Figure C: Phi(x, t=T/2) – basis_1 vs basis_2
#     fig, axs = plt.subplots(n_plot, figsize=(6, 0.8 * n_plot))
#     if n_plot == 1:
#         axs = [axs]
#     fig.suptitle(rf"$\Phi(x,\,t=T/2)$, $\mu={kappa:.2f}$")
#     for i in range(n_plot):
#         idx = sorted_idx[i]
#         s1  = sol1_list[idx]
#         s2  = sol2_list[small_diff_idx[idx]]
#         axs[i].plot(x_list, s1[:, N_t_plot // 2].real, color="r", label=r"basis: $\phi$")
#         axs[i].plot(x_list, s2[:, N_t_plot // 2].real, color="g", linestyle="dashed",
#                     label=r"basis: $\tilde\phi$")
#         axs[i].label_outer()
#     axs[-1].legend(fontsize=7)
#     axs[-1].set_xlabel(r"$x$")
#     fig.tight_layout()
#     plt.savefig(f"figures/diff_m/eigenvalue_2d_{tag}_t0.5pi.pdf")
#     plt.close(fig)

#     # Figure D: Contour sequence (n_plot//3 rows × 3 cols)
#     n_rows = n_plot // 2
#     if n_rows > 0:
#         fig, axs2d = plt.subplots(n_rows, 2, figsize=(7, 5))
#         if n_rows == 1:
#             axs2d = axs2d[np.newaxis, :]
#         for i in range(n_rows):
#             for j in range(2):
#                 idx      = sorted_idx[2 * i + j]
#                 solution = sol1_list[idx].real
#                 axs2d[i, j].contourf(x_grid, t_grid, solution, 20, cmap=cmap_rdgy)
#                 axs2d[i, j].set_title(f"dom k: {max_indices[idx] + 1:d}", size=8)
#         for ax in axs2d.flat:
#             ax.set_xlabel(r"$x$", fontsize=8)
#             ax.set_ylabel(r"$t$", fontsize=8)
#             ax.label_outer()
#         # fig.suptitle(rf"$\Phi(t,x)$ contour sequence, $\mu={kappa:.2f}$", fontsize=10)
#         fig.tight_layout()
#         plt.savefig(f"figures/diff_m/eigenvalue_2d_{tag}_contour_sequences.pdf")
#         plt.close(fig)

#     # Figure E: Single contour (lowest-dominant-k solution)
#     idx      = sorted_idx[0]
#     solution = sol1_list[idx].real
#     fig = plt.figure(figsize=(4.5, 3))
#     plt.contourf(x_grid, t_grid, solution, 20, cmap=cmap_rdgy)
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$t$")
#     plt.colorbar()
#     plt.title(rf"$\Phi(t,x)$, $\mu={kappa:.2f}$, dom $k={max_indices[idx] + 1}$", fontsize=10)
#     plt.subplots_adjust(bottom=0.15)
#     plt.savefig(f"figures/diff_m/eigenvalue_2d_{tag}_contour1.pdf")
#     plt.close(fig)

#     # Figure F: Cylinder plot – prefer solution with k_dom=5, else 6, else 4
#     k_dom_all = max_indices + 1  # 1-based dominant k for each solution
#     idx_cyl = None
#     for target_k in [5, 6, 4]:
#         hits = np.where(k_dom_all == target_k)[0]
#         if len(hits) > 0:
#             idx_cyl = hits[0]
#             break
#     if idx_cyl is None:
#         idx_cyl = sorted_idx[0]
#     solution_cyl = sol1_list[idx_cyl].real

#     y_cyl = np.cos(x_grid)
#     z_cyl = np.sin(x_grid)
#     vmin, vmax = solution_cyl.min(), solution_cyl.max()
#     facecolors = cmap_rdgy((solution_cyl - vmin) / (vmax - vmin + 1e-30))
#     fig = plt.figure(figsize=(3.2, 3.2))
#     ax3d = fig.add_subplot(1, 1, 1, projection="3d")
#     ax3d.set_axis_off()
#     ax3d.plot_surface(y_cyl, z_cyl, t_grid, rstride=1, cstride=1,
#                       facecolors=facecolors, linewidth=0, antialiased=False, alpha=0.9)
#     # plt.title(rf"$\Phi(t,x)$, $\mu={kappa:.2f}$, dom $k={k_dom_all[idx_cyl]}$", fontsize=10)
#     fig.tight_layout()
#     plt.savefig(f"figures/diff_m/eigenvalue_2d_{tag}_cylinder_kdom{k_dom_all[idx_cyl]}.pdf")
#     plt.close(fig)

#     print(rf"$\mu={kappa:.2f}$: saved 6 figures to figures/diff_m/")
