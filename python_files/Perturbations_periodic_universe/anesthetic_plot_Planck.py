from matplotlib import pyplot as plt
import numpy as np
from anesthetic import read_chains, make_2d_axes
import pandas as pd

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

omegarh2 = 2.47e-5 

ns = read_chains('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE')
column_names_raw = np.loadtxt('/home/wnd22/Documents/Research/PlanckData_2018/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE.paramnames', dtype=str, usecols=[0])
column_names = [x.replace("b'",'').replace("'",'') for x in column_names_raw]

ns['omegar'] = omegarh2 / ns['H0']**2 * 100**2
ns.set_label('omegar', r'$\Omega_r$')
ns.set_label('omegak', r'$\Omega_{K}$')
ns.set_label('omegal', r'$\Omega_{\Lambda}$')

ns['kappa'] = - ns.omegak / np.sqrt(ns.omegal*ns.omegar) / 2.
ns.set_label('kappa', r'$\tilde\kappa$')

# ns.kappa.plot.kde_1d()
# plt.show()

column_names = ['H0', 'omegal', 'omegak', 'kappa']
fig, ax = make_2d_axes(column_names, figsize=(3.375,3.375))

kwargs = dict(upper_kwargs=dict(levels=[0.95, 0.68], bins=20, alpha=0.9),
               lower_kwargs=dict(zorder=-10, ms=0.5),
               diagonal_kwargs=dict(bins=20, alpha=0.9),
               kinds=dict(upper='hist_2d', diagonal='hist_1d', lower='scatter_2d'))

ns.plot_2d(ax, **kwargs, color='orange', label="Planck 2018")

# ns = ns[ns.kappa<=1]


discrete_kappa = [0.113284, 0.238231, 0.671658, 0.98607, 0.99969] # slope = [1/3, 1/2, 1, 2, 3] respectively 
color_list = ['aliceblue', 'lightcyan', 'lightblue', 'lightskyblue', 'deepskyblue', 'blue', 'darkblue'] # slope = [1/5, 1/4, 1/3, 1/2, 1, 2, 3] respectively 

for n in range(len(discrete_kappa)):
    kappa = discrete_kappa[n]
    # s = ns[np.isclose(ns.kappa, kappa, rtol=0.5)]
    s = ns[(ns.kappa>=kappa-0.025) & (ns.kappa<=kappa+0.025)]
    if s.empty:
        print('no samples close to kappa =', kappa)
        continue
    s.plot_2d(ax,**kwargs,color=color_list[n+2], label=rf'$\tilde\kappa = {round(kappa,3)}$')

ax.iloc[-1,  0].legend(loc='lower center', bbox_to_anchor=(len(ax)/2, len(ax)), ncol=3)
ax['kappa']['kappa'].set_xlim(0, 1)
fig.autofmt_xdate()
plt.savefig('Planck_kappa.pdf', bbox_inches="tight")