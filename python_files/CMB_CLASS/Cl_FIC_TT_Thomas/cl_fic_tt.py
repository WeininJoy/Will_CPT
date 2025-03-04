##### Script to Generate Figure of the TT Cls for FIC

#### Imports
import matplotlib
matplotlib.use('PDF') 
import classy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.special as scp
import numpy as np
from accuracies import high_acc

##REVTEK Params
plt.rcParams['axes.labelsize']=10
plt.rcParams['legend.fontsize']=8
plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8
plt.rcParams['text.usetex']=True
plt.rcParams['font.family']="serif"
plt.rcParams['font.serif']="cm"


#### Parameters
## Function
T_cmb = 2.725
ell_max= 2508
D_A = 13878

## Display
Cherry_Red = '#900001'
Bright_Red = '#fe3533'
Warm_Green = '#8db046'
Ultramarine = '#4166f5'
Deep_Blue = '#1d3e6e'
Black = 'k'
Light_Grey = '#A0A0A0BF'

width_default = 1

## Models
models = ('lcdm','fic','fic','fic','fic')
dks = (1.0e-4,2.0e-4,3.0e-4,5.0e-4,10.0e-4)

## CLASS Parameters
ell_max = 2508
class_params = { #The invariant parameters between models
        'H0': 67.32117,
        'omega_b': 0.02238280,
        'N_ur': 2.03066666667,
        'omega_cdm': 0.1201075,
        'N_ncdm': 1,
        'omega_ncdm': 0.0006451439,
        'YHe': 0.2454006,
        'tau_reio': 0.05430842,
        'non linear': 'halofit',
        'output': 'tCl lCl pCl',
        'lensing': 'yes',
        'l_max_scalars':ell_max,
        'write warnings' : 'yes'}

params = []
for model, dk in zip(models,dks):
    model_params = {
        'P_k_ini type': 'external_Pk',
        'command': 'python3 /home/thomas/Documents/Research/ICBI/9-Project_To_Paper/Figures/Cl_FIC_TT/PPS/' + model + '_ps.py', #Change for your file structures
        'custom1': 0.9660499,
        'custom2': 2.100549e-09,
        'custom3': dk}

    model_params.update(class_params)
    model_params.update(high_acc)
    params.append(model_params)




#### Calculate Cls

# Initalising class objects
print("Initalising Class Object:")
cosmo = classy.Class()

# Calculate Cls
cls = []
for model, param in zip(models,params):
    cosmo.set(param)
    cosmo.compute()
    cl = cosmo.raw_cl(ell_max)
    
    for spectra in cl:
       if ((spectra == 'bb') or (spectra == 'ell')):
            cl[spectra] = cl[spectra] 
       elif (spectra == 'pp'):
            norm = cl['ell']*cl['ell']*(cl['ell']+1)*(cl['ell']+1)/(2.0*np.pi)
            cl[spectra] = cl[spectra] * norm * (T_cmb*10**6)**2
       else:
            norm = cl['ell']*(cl['ell']+1)/(2.0*np.pi)
            cl[spectra] = cl[spectra] * norm * (T_cmb*10**6)**2

        
    cls.append(cl.copy())
    cosmo.struct_cleanup()
    cosmo.empty()
    
(ls,lcdm_TT) = (cls[0]['ell'][2:],cls[0]['tt'][2:])
(ls,fic2_TT) = (cls[1]['ell'][2:],cls[1]['tt'][2:])
(ls,fic3_TT) = (cls[2]['ell'][2:],cls[2]['tt'][2:])
(ls,fic5_TT) = (cls[3]['ell'][2:],cls[3]['tt'][2:])
(ls,fic10_TT) = (cls[4]['ell'][2:],cls[4]['tt'][2:])


## Planck Results
filename = "COM_PowerSpect_CMB-TT-full_R3.01.txt"

file = open(filename,"r")
    
#Remove the Header
dump = file.readline()
    
#Read in the actual data
exp = np.empty(2507)
errors = np.empty((2,2507))
count = 0
for line in file:
    exp[count] = float(line.split()[1])
    errors[0,count] =  float(line.split()[2])
    errors[1,count] = float(line.split()[3])

    count += 1




### Plotting

## Lines to plot
TTs = (lcdm_TT,fic2_TT,fic3_TT,fic5_TT,fic10_TT)
cols = (Black,'C0','C1','C2','C3')
labels = (r'$\Lambda$CDM',r'$ \Delta k =2.0 \times 10^{-4}$ Mpc$^{-1}$',r'$ \Delta k =3.0 \times 10^{-4}$ Mpc$^{-1}$',r'$ \Delta k =5.0 \times 10^{-4}$ Mpc$^{-1}$',r'$ \Delta k =10.0 \times 10^{-4}$ Mpc$^{-1}$')
zorders = (1.4,1.3,1.2,1.1,1.0)


## Form Grid
gs = gridspec.GridSpec(3,4)
gs.update(wspace=0.0, hspace=0.0)
ax4 = plt.subplot(gs[1:,2:])
ax3 = plt.subplot(gs[1:,0:2])
ax1 = plt.subplot(gs[0, 0:2])
ax2 = plt.subplot(gs[0,2:])
fig = plt.gcf()
fig.set_size_inches(7.20, 3.75)

## Plot Upper Left
ax1.errorbar(ls[0:29],exp[0:29],errors[:,0:29],fmt = 'none',elinewidth = 1.0,color = Light_Grey,ecolor=Light_Grey,label = 'Planck 2018',zorder = 0.5)
for TT,col,zo,lbl in zip(TTs,cols,zorders,labels):
    ax1.plot(ls[0:29],TT[0:29],col,linewidth=width_default,zorder = zo,label=lbl)
ax1.set_ylim([-200,6200])
ax1.set_xlim([1.5,30])
ax1.set_xscale("log")
ax1.xaxis.set_ticklabels([])
ax1.tick_params(axis="x", direction='in',which = 'both')
ax1.yaxis.set_ticks(range(200,6000,200),minor=True)
ax1.tick_params(axis="y", direction='in',which = 'both')
ax1.yaxis.set_ticks([0,1000,2000,3000,4000,5000,6000])
ax1.yaxis.set_ticklabels([0,1000,2000,3000,4000,5000,6000])


## Plot Upper Right
ax2.errorbar(ls[30:2507],exp[30:2507],errors[:,30:2507],fmt = 'none',elinewidth = 0.2,color = Light_Grey,ecolor=Light_Grey,label = 'Planck 2018',zorder = 0.5)
for TT,col,lbl,zo in zip(TTs,cols,labels,zorders):
    ax2.plot(ls[30:2507],TT[30:2507],col,linewidth=width_default,label = lbl,zorder = zo)
ax2.set_ylim([-200,6200])
ax2.set_xlim([30,2600])
ax2.xaxis.set_ticks(range(100,2600,100),minor=True)
ax2.set_yticks([],)
ax2.xaxis.set_ticks([500,1000,1500,2000,2500])
ax2.xaxis.set_ticklabels([])
ax2.tick_params(axis="x", direction='in',which = 'both')
ax2.tick_params(axis="y", direction='in',which = 'both',right=True,left=False)
ax2.set_ylim([-200,6200])
ax2.yaxis.set_ticks(range(200,6000,200),minor=True)
ax2.yaxis.set_ticks(range(0,7000,1000))
ax2.yaxis.set_ticklabels([])

## Plot Lower Left
ax3.errorbar(ls[0:29],exp[0:29]-lcdm_TT[0:29],errors[:,0:29],fmt = 'none',elinewidth = 1.0,color = Light_Grey,label = 'Planck 2018',ecolor=Light_Grey,zorder = 0.5)
for TT,col,zo,lbl in zip(TTs,cols,zorders,labels):
    ax3.plot(ls[0:29],TT[0:29]-lcdm_TT[0:29],col,linewidth=width_default,zorder = zo,label = lbl)
ax3.set_ylim([-1200,1200])
ax3.set_xlim([1.5,30])
ax3.set_xscale("log")
ax3.xaxis.set_ticks([2,10,30])
ax3.xaxis.set_ticklabels([2,10,30])
ax3.tick_params(axis="x", direction='in',which = 'both',top=True)
ax3.tick_params(axis="y", direction='in',which = 'both')
ax3.yaxis.set_ticks([-1000,-500,0,500,1000])
ax3.yaxis.set_ticklabels([-1000,-500,0,500,1000])
ax3.yaxis.set_ticks(range(-1200,1200,100),minor=True)


##Plot Lower Right
ax4.errorbar(ls[30:2507],exp[30:2507]-lcdm_TT[30:2507],errors[:,30:2507],fmt = 'none',elinewidth = 0.2,color = Light_Grey,ecolor=Light_Grey,label = 'Planck 2018',zorder = 0.5)
for TT,col,zo,lbl in zip(TTs,cols,zorders,labels):
    ax4.plot(ls[30:2507],TT[30:2507]-lcdm_TT[30:2507],col,linewidth=width_default, zorder = zo,label = lbl)
ax4.set_ylim([-1200,1200])
ax4.set_xlim([30,2600])
ax4.set_yticks([])
ax4.xaxis.set_ticks(range(100,2600,100),minor=True)
ax4.xaxis.set_ticks([500,1000,1500,2000,2500])
ax4.xaxis.set_ticklabels(['',1000,'',2000,''])
ax4.tick_params(axis="x", direction='in',which = 'both',top=True)
ax4.yaxis.set_ticks([-1000,-500,0,500,1000])
ax4.yaxis.set_ticklabels([])
ax4.tick_params(axis="y", direction='in',which = 'both',right=True,left=False)
ax4.yaxis.set_ticks(range(-1200,1200,100),minor=True)
ax4.legend(prop={'size': 7},loc = 'upper right')

##Limber Axis
ax5 = ax1.twiny()
ax6 = ax2.twiny()

ax5.set_xlim([1.5/D_A,30/D_A])
ax5.set_xscale("log")
ax5.tick_params(axis="x", direction='in',which = 'both')

ax6.set_xlim([30/D_A,2600/D_A])
ax6.xaxis.set_ticks(np.linspace(0.01,0.19,19),minor=True)
ax6.xaxis.set_ticks(np.linspace(0.05,0.15,3))
ax6.tick_params(axis="x", direction='in',which = 'both')


##Axis titles
fig.text(0.04, 0.75, (r'$\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]'),ha = 'center', va='center', rotation='vertical',fontsize=12)
fig.text(0.04, 0.36, (r'$\Delta\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]'),ha = 'center',  va='center', rotation='vertical',fontsize=12)
fig.text(0.50, 0.02, (r'$\ell$'), va='center',fontsize=12)
fig.text(0.46, 0.96, (r'$k$ [Mpc$^{-1}$]'), va='center',fontsize=12)

plt.subplots_adjust(left=0.10, right=0.90, top=0.87, bottom=0.10)

plt.savefig('cl_fic_tt.pdf')