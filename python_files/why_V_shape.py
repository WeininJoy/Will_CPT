import numpy as np
import matplotlib.pyplot as plt

m = 1
t_fcb = np.pi
N = 20
N_x = 500
N_t = 500
x = np.linspace(0, 2*np.pi, N_x)
t = np.linspace(0, t_fcb, N_t)


X, T = np.meshgrid(x, t)
Y = np.cos(X)
Z = np.sin(X)

basis_1 = [np.cos(np.sqrt(k**2 + m**2) * T) * np.exp(1j*k*X) for k in range(1, N+1)]

amplitude = [ 1.28665642e-02+0.j, -3.97038545e-02+0.j,  8.44426750e-02+0.j,\
 -1.54021788e-01+0.j,  2.50769937e-01+0.j, -3.52780661e-01+0.j,\
  3.71777802e-01+0.j, -1.22813108e-01+0.j, -4.86091292e-01+0.j,\
  6.95594339e-01+0.j,  8.65103553e-01+0.j,  3.74232056e-01+0.j,\
  7.74463775e-02+0.j,  1.39894436e-02+0.j, -1.51377289e-03+0.j,\
  2.10914820e-03+0.j, -1.62421258e-03+0.j,  1.37747346e-03+0.j,\
 -1.14518381e-03+0.j,  1.04066299e-03+0.j,  2.99768041e-10+0.j,\
 -1.63478140e-09+0.j,  3.07410015e-09+0.j, -4.92770520e-09+0.j,\
  1.77932400e-08+0.j, -6.05397259e-09+0.j, -1.19617012e-09+0.j,\
  1.08540928e-08+0.j, -7.90003824e-09+0.j,  1.16104912e-08+0.j,\
 -1.45675293e-13+0.j,  1.70309312e-08+0.j,  7.69669259e-09+0.j,\
 -2.31615369e-08+0.j, -1.56759639e-08+0.j,  1.12581813e-08+0.j,\
  9.52383345e-09+0.j, -1.01868436e-09+0.j,  1.67020324e-08+0.j,\
 -1.17667053e-09+0.j]

phi = np.zeros_like(basis_1[0], dtype=complex)
for i in range(len(basis_1)):
    phi += amplitude[i]* basis_1[i]

fig, ax = plt.subplots()
cmap = plt.get_cmap('RdGy')
ax.contourf(X, T, phi.real, 20, cmap=cmap)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_axis_off()

facecolors = cmap((phi-phi.min())/(phi.max()-phi.min()))
plot = ax.plot_surface(Y, Z, T, rstride=1, cstride=1, facecolors=facecolors, linewidth=0, antialiased=False, alpha=0.9)
plt.show()