import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

start_time = time.time()

if __name__=="__main__":

    # Parameters
    N = 25
    N_t = 500
    N_x = 500
    kappa = 1.5
    T = np.pi 
    L = 2. * np.pi
    eigenvalues_threshold = 1.e-1
    N_plot = 6

    # Interval t=[0, T], x=[0, L]
    t_list = np.linspace(0, T, N_t)
    x_list = np.linspace(0, L, N_x)
    t, x = np.meshgrid(t_list, x_list)

    
    # List of basis to apply the Gram-Schmidt process
    basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j*k*x) for k in range(1, N+1)]
    if (np.pi/T)**2 - kappa**2 < 0: # make sure wave vector k is real
        start_n = round(np.sqrt(abs((np.pi/T)**2 - kappa**2))) + 1
    else: start_n = 1
    basis_2 = [np.cos(n*np.pi/T * t) * np.exp(1j*np.sqrt((n*np.pi/T)**2 - kappa**2)*x) for n in range(start_n, N+start_n)]

    # Load eigenvalues and coefficients from txt files
    eigenvalues_valid_1 = np.loadtxt(f"./data/linear_combination/eigenvalues1_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}.txt", dtype=np.complex_)
    coefficients_1 = np.loadtxt(f"./data/linear_combination/coefficients1_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}.txt", dtype=np.complex_)
    # sorted_index = np.array(eigenvalues_valid_1.real).argsort().tolist()[::-1]  # Sort the eigenvalues in descending order
    max_indices = np.argmax(coefficients_1.real, axis=1) # Find the index of the maximum value in each row (coefficients)
    sorted_index = np.argsort(max_indices)  # Sort the coefficients based on the dominant k mode (from small to large)

    eigenvalues_valid_2 = np.loadtxt(f"./data/linear_combination/eigenvalues2_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}.txt", dtype=np.complex_)
    coefficients_2 = np.loadtxt(f"./data/linear_combination/coefficients2_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}.txt", dtype=np.complex_)

    # Plot the coefficients of the solutions
    if len(eigenvalues_valid_1) < N_plot:
        N_plot = len(eigenvalues_valid_1)

    fig, axs = plt.subplots(N_plot, figsize=(3.8,0.7*N_plot))
    fig.suptitle("coefficients for linear combination", fontsize=10)

    for i in range(N_plot):
        idx = sorted_index[i]
        coefficients = coefficients_1[idx, :].real
        axs[i].bar([k for k in range(1,N+1)], coefficients.tolist(),width=0.2)
        axs[i].set_xlim(0, 20)
        axs[i].set_ylim(-1, 1.)

    for ax in axs.flat:
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"$k$", fontsize=8)
    fig.tight_layout()
    plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_coefficients.pdf")


    # # Construct the solutions
    # solution_1_list = []
    # solution_2_list = []
    # for i in range(len(eigenvalues_valid_1)):
    #     solution_1 = np.zeros_like(basis_1[0], dtype=complex)
    #     solution_2 = np.zeros_like(basis_2[0], dtype=complex)
    #     for j in range(N):
    #         solution_1 += coefficients_1[i, j]*basis_1[j]
    #         solution_2 += coefficients_2[i, j]*basis_2[j]  
    #     solution_1_list.append(solution_1)
    #     solution_2_list.append(solution_2)

    # diff_array = np.zeros((len(eigenvalues_valid_1), len(eigenvalues_valid_2)))
    # small_diff_index = []
    # for i in range(len(eigenvalues_valid_1)):
    #     solution_1 = solution_1_list[i]
    #     for j in range(len(eigenvalues_valid_2)):
    #         solution_2 = solution_2_list[j]
    #         diff_array[i, j] += np.linalg.norm(solution_1 - solution_2) / np.linalg.norm(solution_1)
    #     small_diff_index.append(np.argmin(diff_array[i, :]))

    # # Plot the solution Phi(t) at x=L/2
    # if len(eigenvalues_valid_1) < N_plot:
    #     N_plot = len(eigenvalues_valid_1)

    # fig, axs = plt.subplots(N_plot, figsize=(6,0.8*N_plot))
    # fig.suptitle(r"$\Phi(t,x=L/2)$")
    # for i in range(N_plot):
    #     idx = sorted_index[i]
    #     solution_1 = solution_1_list[idx]
    #     solution_2 = solution_2_list[small_diff_index[idx]]
    #     axs[i].plot(t_list, solution_1[N_x//2, :].real, color='r', label=r"basis: $\phi$")
    #     axs[i].plot(t_list, solution_2[N_x//2, :].real, color='g', linestyle='dashed', label=r"basis: $\tilde\phi$")
    # plt.legend()
    # plt.xlabel(r"$t$")
    # plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_xpi_sorted_by_eigenvalue.pdf")

    # # Plot the solution Phi(x) at t=T/2
    # if len(eigenvalues_valid_1) < N_plot:
    #     N_plot = len(eigenvalues_valid_1)
    # fig, axs = plt.subplots(N_plot, figsize=(6,0.8*N_plot))
    # fig.suptitle(r"$\Phi(t,x=L/2)$")
    # for i in range(N_plot):
    #     idx = sorted_index[i]
    #     solution_1 = solution_1_list[idx]
    #     solution_2 = solution_2_list[small_diff_index[idx]]
    #     axs[i].plot(x_list, solution_1[:, N_t//2].real, color='r', label=r"basis: $\phi$")
    #     axs[i].plot(x_list, solution_2[:, N_t//2].real, color='g', linestyle='dashed', label=r"basis: $\tilde\phi$")  
    # plt.legend()
    # plt.xlabel(r"$x$")
    # plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_t0.5pi.pdf")


    # ## 3D plot of the solution Phi(t,x) 
    # # Contour plot (sequence)
    # fig, axs = plt.subplots(len(eigenvalues_valid_1)//3, 3, figsize=(7,7))
    # cmap = plt.get_cmap('RdGy')
    # for i in range(len(eigenvalues_valid_1)//3):
    #     for j in range(3):
    #         idx = sorted_index[3*i+j]
    #         solution = solution_1_list[idx].real
    #         axs[i, j].contourf(x, t, solution, 20, cmap=cmap)
    #         axs[i, j].set_title(f"dominate n: {max_indices[idx]:d}", size=8)
    # for ax in axs.flat:
    #     ax.set_xlabel(r'$x$', fontsize=8)
    #     ax.set_ylabel(r'$t$', fontsize=8)
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # fig.tight_layout()
    # plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_contour_sequences.pdf")

    # # Plot just one contour plot (solution[0])
    # cmap = plt.get_cmap('RdGy')
    # idx = sorted_index[0]
    # solution = solution_1_list[idx].real
    # fig = plt.figure(figsize=(4.5,3))
    # plt.contourf(x, t, solution, 20, cmap=cmap)
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$t$')
    # plt.colorbar()
    # plt.subplots_adjust(bottom=0.15) 
    # plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_contour1.pdf")

    # # Cylinder Plot 
    # y = np.cos(x)
    # z = np.sin(x)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.set_axis_off()

    # facecolors = cmap((solution-solution.min())/(solution.max()-solution.min()))
    # plot = ax.plot_surface(y, z, t, rstride=1, cstride=1, facecolors=facecolors, linewidth=0, antialiased=False, alpha=0.9)
    # fig.tight_layout()
    # plt.savefig(f"eigenvalue_2d_N{N:d}_Nt{N_t:d}_T{T:.2f}_m{kappa:.2f}_cylinder1.pdf")
    

print("--- %s seconds ---" % (time.time() - start_time))