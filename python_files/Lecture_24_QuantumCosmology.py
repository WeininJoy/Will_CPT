import sys
import lecture_style
import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import sigma, c, hbar,G, m_p
from scipy.constants import k as kB
from scipy.integrate import solve_ivp

def V(phi):
    return phi**2/2
def dV(phi):
    return phi

def f(t, y):
    phi, dphi, N, R, dR = y
    H = np.sqrt( 1/3 * (dphi**2/2 + V(phi)))
    ddphi = -3*H*dphi - dV(phi)
    dH = -dphi**2/2

    ddR = -(3*H+2*ddphi/dphi-2*dH/H)*dR - k**2*np.exp(-2*N)*R
    return dphi, -3*H*dphi - dV(phi), H, dR, ddR


t0 = 1e-2
t1 = 1e2
phip = 6

y0 = phip - np.sqrt(2/3)*np.log(t0)
dy0 =  - np.sqrt(2/3)/t0
N0 = 0
k = 0

t_eval = np.logspace(np.log10(t0),np.log10(t1), 10000)
sol = solve_ivp(f, [t0,t1], [y0,dy0,N0,0,0], t_eval=t_eval)

t = sol.t
phi, dphi, N, R, dR = sol.y
H = np.sqrt( 1/3 * (dphi**2/2 + V(phi))) 

fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_xlabel('log evolution')
ax.set_ylabel('Horizon')
ax.set_yticks([])
ax.plot(np.log(t), -np.log(H)-N)

for k in np.logspace(2,5,5):
    sol = solve_ivp(f, [t0,t1], [y0,dy0,N0,1,0], t_eval=t_eval)
    t = sol.t
    phi, dphi, N, R, dR = sol.y
    ax.plot(np.log(t),R/abs(R).max()-np.log(k))

fig.tight_layout()
fig.savefig('Lecture_24/mukhanov.pdf')



T0 = 2.7255
Mpc = 3.086e22
rhogamma = np.pi**2*kB**4/(15*hbar**3*c**3) * T0**4/c**2
H0 = 67.4 * 1e3 / Mpc
Or = 8*np.pi*G*rhogamma /(3*H0**2) 
Om = 0.3
Ok = 0
Ol = 1-Om-Ok-Or


def g(t, y):
    N, R, dR = y

    H = (Or * np.exp(-4*N) + Om * np.exp(-3*N)  + Ok * np.exp(-2*N) + Ol)**0.5

    ddR = -3*H*dR - w * k**2*np.exp(-2*N)*R
    
    return H, dR, ddR


def stop(t, y):
    N, Phi, dPhi = y
    return N + np.log(1+zstop)


stop.terminal = True

Ni = -18
ti = np.exp(2*Ni)/Or**0.5/2
y0=[Ni,0,0]
#w = 0
w = 1/3
k = 0

zstop = 1000
sol = solve_ivp(g, [ti,np.inf], y0, events=stop)
trec = sol.t_events[0][-1] 
yrec = sol.y_events[0][-1]
N, Phi, dPhi = sol.y
H = (Or * np.exp(-4*N) + Om * np.exp(-3*N)  + Ok * np.exp(-2*N) + Ol)**0.5

zstop = 0
sol = solve_ivp(g, [ti,np.inf], y0, events=stop)
t0 = sol.t_events[0][-1]
t_eval =np.logspace(np.log10(ti),np.log10(t0),1000)[1:-1]
sol = solve_ivp(g, [ti,np.inf], y0, events=stop,t_eval=t_eval,rtol=1e-8)

N, R, dR = sol.y
H = (Or * np.exp(-4*N) + Om * np.exp(-3*N)  + Ok * np.exp(-2*N) + Ol)**0.5

fig, ax = plt.subplots()

ax.plot(np.log(sol.t),-np.log(H)-N)  # plot time - Horizon: log(1/(aH))
ax.axvline(np.log(trec), color='k', ls='--')  # CMB time

y0=[Ni,1,0]
for k in np.logspace(2,5,5):
    t_eval =np.logspace(np.log10(ti),np.log10(trec),1000)[1:-1]
    sol = solve_ivp(g, [ti,trec], y0, t_eval=t_eval)
    t = sol.t
    N, R, dR = sol.y
    t = np.concatenate([t,[trec,t0]])
    R = np.concatenate([R,[R[-1],R[-1]]])
    ax.plot(np.log(t),R/abs(R).max()-np.log(k))


ax.set_xticks([])
ax.set_xlabel('log evolution')
ax.set_ylabel('Horizon')
ax.set_yticks([])
fig.tight_layout()
fig.savefig('Lecture_24/radiation.pdf')



fig, axes = plt.subplots(5, sharex=True, gridspec_kw={'hspace':0}) 

y0=[Ni,0,0]
t_eval =np.logspace(np.log10(ti),np.log10(t0),1000)[1:-1]
sol = solve_ivp(g, [ti,np.inf], y0, events=stop,t_eval=t_eval,rtol=1e-8)

for ax in axes:
    ax.plot(np.log(sol.t),-np.log(H)-N)
    ax.axvline(np.log(trec), color='k', ls='--')

for i, (k, ax) in enumerate(zip([10,50,80,120,150], axes)):
    for start in [0,-0.5,-1,0.5,1]:
        y0=[Ni,start,0]
        t_eval =np.logspace(np.log10(ti),np.log10(trec),1000)[1:-1]
        sol = solve_ivp(g, [ti,trec], y0, t_eval=t_eval)
        t = sol.t
        N, R, dR = sol.y
        t = np.concatenate([t,[trec,t0]])
        R = np.concatenate([R,[R[-1],R[-1]]])
        ax.plot(np.log(t),R-np.log(k), color='C%i' %(i+1))
    ax.set_ylim(-np.log(k)-1.5,-np.log(k)+1.5)
    ax.set_yticks([])

ax.set_xlim(np.log(trec)-15,np.log(trec)+2)


ax.set_xticks([])
ax.set_xlabel('log evolution')
fig.tight_layout()
fig.savefig('Lecture_24/synch.pdf')
sys.exit(0);





sys.exit(0);

w=1/3
k = 1
yi=[Ni,0,1]
sol = solve_ivp(f, [ti, trec], yi, t_eval = np.logspace(np.log10(ti),np.log10(trec),10000)[1:-1])
N, Phi, dPhi = sol.y
H = (Or * np.exp(-4*N) + Om * np.exp(-3*N)  + Ok * np.exp(-2*N) + Ol)**0.5
drho = -(3*H*(dPhi + H*Phi) + k**2*np.exp(-2*N)*Phi)
rho = 3*H**2
delta = drho/rho
plt.plot(np.exp(N),drho)
plt.xscale('log')
plt.yscale('symlog')
#plt.ylim(-0.3,0.3)

plt.yscale('log')
#plt.plot(sol.t,sol.y[1])
plt.ylim(-5,5)

sol.y[1]


plt.plot(sol.t, np.exp(sol.y[0]))
