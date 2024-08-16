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

# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


omegarh2 = 4.53e-5

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
color_list = [lighten_color('b', 0.25* (n+1)) for n in range(7)] # slope = [1/5, 1/4, 1/3, 1/2, 1, 2, 3] respectively 

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