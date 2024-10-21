import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

mt_list = np.linspace(300, 450, 10)
kt_list = np.linspace(1.e-4, 1, 10)

deltaKint_arr = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        mt = mt_list[i]
        kt = kt_list[j]
        allowedK = np.load(f'./data/allowedK_{10*i+j}.npy')
        deltaK_list = [allowedK[n+1] - allowedK[n] for n in range(len(allowedK)-1)]
        deltaK = sum(deltaK_list[len(deltaK_list)//2:]) / len(deltaK_list[len(deltaK_list)//2:])
        deltaKint = deltaK / np.sqrt(kt)
        print('Delta k_int = ', 1/deltaKint)
        deltaKint_arr[i,j] = 1/deltaKint

# Create the figure and axes object
fig, ax = plt.subplots()

x, y = np.meshgrid(kt_list, mt_list)
Z = deltaKint_arr

levels = [1./N for N in range(6,100)]
levels = levels[::-1]

# slope contour
def fmt(x):
    N = round(1./x)
    return rf"$\Delta kint=${N:d}"

CS = ax.contour(x, y, Z, levels=levels)
ax.clabel(CS, CS.levels[-10:-1], fmt=fmt, inline=True, fontsize=8)

# lev_exp = np.arange(np.floor(np.log10(Z.min())),
#                    np.ceil(np.log10(Z.max())+1),0.3)
# levs = np.power(10, lev_exp)
# CS = ax.contourf(x, y, Z, levs, locator=ticker.LogLocator(), cmap=cm.PuBu_r
# ax.clabel(CS, CS.levels[:6], inline=True, fmt=fmt, fontsize=16)
cbar = fig.colorbar(CS)

plt.xlabel(r'$\tilde \kappa$')
plt.ylabel(r'$\tilde m$')
plt.title(r'$1/\Delta k_{int}$')
plt.savefig("1_kint_contour.pdf")