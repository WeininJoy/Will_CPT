import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})

##############
# make wave plots in a tunnel
##############

t_fcb = np.pi
N = 4
N_x = 1000
N_t = 1000
x = np.linspace(0, np.pi, N_x)
t = np.linspace(0, t_fcb, N_t)


X, T = np.meshgrid(x, t)
Y = np.cos(X)
Z = np.sin(X)

def wave(x, t, k):
    return np.cos(np.sqrt(k**2 ) * t) * np.exp(1j*k*x)

basis_1 = [wave(X, T, k) for k in range(1, N+1)]

# plot it
fig, (a0, a1) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1.5]}, figsize=(7, 3)) # 3, 4.5, 7.5
cmap = plt.get_cmap('RdGy')


# for T=pi (m=1,2,n=1,2) and T=2*pi (m=2,4,n=1,2)
a0[0].set_title(r"$m=1,n=1$", fontsize=17) 
a0[1].set_title(r"$m=2,n=2$", fontsize=17) 
a0[0].plot(x[:N_x//2], wave(x, 0, 1).real[:N_x//2], color='r')
a0[0].plot(x[N_x//2:], wave(x, 0, 1).real[N_x//2:], color='black')
a1[0].contourf(X, T, basis_1[0].real, 20, cmap=cmap)
a0[1].plot(x[:N_x//4], wave(x, 0, 2).real[:N_x//4],color='r')
a0[1].plot(x[N_x//4:3*N_x//4], wave(x, 0, 2).real[N_x//4:3*N_x//4],color='black')
a0[1].plot(x[3*N_x//4:], wave(x, 0, 2).real[3*N_x//4:],color='r')

# # for T=0.5*pi (m=1,2,n=2,4)
# a0[0].set_title(r"$m=1,n=2$", fontsize=17)
# a0[1].set_title(r"$m=2,n=4$", fontsize=17)
# a0[0].plot(x[:N_x//4], wave(x, 0, 2).real[:N_x//4],color='r')
# a0[0].plot(x[N_x//4:3*N_x//4], wave(x, 0, 2).real[N_x//4:3*N_x//4],color='black')
# a0[0].plot(x[3*N_x//4:], wave(x, 0, 2).real[3*N_x//4:],color='r')
# a1[0].contourf(X, T, basis_1[1].real, 20, cmap=cmap)
# a0[1].plot(x[:N_x//8], wave(x, 0, 4).real[:N_x//8],color='r')
# a0[1].plot(x[N_x//8:3*N_x//8], wave(x, 0, 4).real[N_x//8:3*N_x//8],color='black')
# a0[1].plot(x[3*N_x//8:5*N_x//8], wave(x, 0, 4).real[3*N_x//8:5*N_x//8],color='r')
# a0[1].plot(x[5*N_x//8:7*N_x//8], wave(x, 0, 4).real[5*N_x//8:7*N_x//8],color='black')
# a0[1].plot(x[7*N_x//8:], wave(x, 0, 4).real[7*N_x//8:],color='r')
# a1[1].contourf(X, T, basis_1[3].real, 20, cmap=cmap)


# Labels 
a0[0].set_ylabel(r'$\Re[\Phi]$', fontsize=17)
a1[0].set_xlabel(r'$x$',fontsize=15)
a1[0].set_ylabel(r'$t$',fontsize=15)
a1[1].set_xlabel(r'$x$',fontsize=15)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in a0.flat:
    ax.label_outer()
for ax in a1.flat:
    ax.label_outer()

fig.tight_layout()
plt.savefig("standing_waves_x-t_Tpi.pdf")


##############
# make n-m plot
##############

# color_list = ['lightsteelblue', 'royalblue', 'darkblue']
# plt.figure(figsize=(5, 5))
# plt.text(2.5, 0.5, r'Slope$=\frac{\Delta m}{\Delta n}=\frac{1}{N}$ or $N$', fontsize=15,  bbox=dict(facecolor='none', edgecolor='red'))
# plt.plot([2*n for n in range(8)],[m for m in range(8)],color=color_list[0], label=r"$T=\frac{1}{2}\pi$, slope=1/2")
# plt.plot([n for n in range(8)],[m for m in range(8)],color=color_list[1], label=r"$T=\pi$, slope=1")
# plt.plot([n for n in range(8)],[2*m for m in range(8)],color=color_list[2], label=r"$T=2\pi$,slope=2")
# plt.xlim(0, 7)
# plt.ylim(0, 7)
# plt.xlabel(r"$k$, should be $n\in\mathbb{N}$", fontsize=15)
# plt.ylabel(r"$\frac{2}{\pi}\theta$, should be $m\in\mathbb{N}$", fontsize=15)
# plt.legend(fontsize=15, loc='upper left')
# plt.savefig("n-m_plot.pdf")