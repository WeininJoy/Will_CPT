"""
Benchmark: solve_ivp sequential vs JAX vmap
ODE: dX2_dt / dX3_dt from Higher_Order_Finding_U_Matrices.py
All 300 k values, column n=4 (vr=1 initial condition) of the ABC matrix.
"""
import os
# Prevent BLAS multi-threading (avoids fork deadlocks and improves benchmarking)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# ─────────────────────────────────────────────────────────────────────────────
# Cosmological parameters (best-fit nu_spacing=4)
# ─────────────────────────────────────────────────────────────────────────────
mt, kt, omega_b_ratio, h = 409.969398, 1.459351, 0.163514, 0.547313
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
rt = 1

Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3)) * Omega_gamma_h2 / h**2

def _f_a0(a0, Omega_r, mt, kt):
    return a0**4 - 3*kt*a0**2 + mt*a0 + (rt - 1./Omega_r)

a0_val = root_scalar(_f_a0, bracket=[1, 1e3], args=(Omega_r, mt, kt)).root
OmegaLambda = Omega_r * a0_val**4
OmegaM = mt * OmegaLambda**(1/4) * Omega_r**(3/4)
OmegaK = -3 * kt * np.sqrt(OmegaLambda * Omega_r)
OmegaR = Omega_r

atol_ode = 1e-13
rtol_ode = 1e-13
stol     = 1e-10
NV       = 75        # number of perturbation variables (state = NV+1 with background)
swaptime = 2.0
deltaeta = 6.6e-4
H0   = 1.0 / np.sqrt(3 * OmegaLambda)
Hinf = H0 * np.sqrt(OmegaLambda)

# ─────────────────────────────────────────────────────────────────────────────
# Background integration  (scipy, done once)
# ─────────────────────────────────────────────────────────────────────────────
t0 = 1e-8
smin1 = np.sqrt(3*OmegaLambda/OmegaR)
szero = -OmegaM/(4*OmegaR)
s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) \
     - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) \
     + OmegaK*OmegaM/(48*OmegaLambda*OmegaR)
s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3)
      - 80./3.*OmegaM**2*OmegaR*OmegaK
      + 224./9.*OmegaR**2*OmegaK**2) \
     / (3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)))
s4 = (-OmegaM**5 + 20./3.*OmegaM**3*OmegaR*OmegaK
      - 32./3.*OmegaM*OmegaR**2*OmegaK**2) \
     / (9216*(OmegaR**3)*(OmegaLambda**2))
s_IC = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4

def ds_dt(t, s):
    v = s[0]
    return [-H0*np.sqrt(OmegaLambda + OmegaK*abs(v**2)
                        + OmegaM*abs(v**3) + OmegaR*abs(v**4))]

def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

def at_fcb(t, X):
    if X[0] < stol: X[0] = 0
    return X[0]
at_fcb.terminal = True

print("Background integration ...")
sol_bg = solve_ivp(ds_dt, [t0, 12], [s_IC], max_step=0.25e-4,
                   events=reach_FCB, method='LSODA',
                   atol=atol_ode, rtol=rtol_ode)
fcb_time = sol_bg.t_events[0][0]
endtime  = fcb_time - deltaeta

z_rec = 1100.0
s_rec = 1 + z_rec
recConformalTime = sol_bg.t[np.abs(sol_bg.y[0] - s_rec).argmin()]

sol2 = solve_ivp(ds_dt, [t0, endtime], [s_IC], method='LSODA',
                 events=at_fcb, atol=atol_ode, rtol=rtol_ode)
s_init = sol2.y[0, -1]

print(f"  fcb_time={fcb_time:.6f}, endtime={endtime:.6f}, "
      f"recConformalTime={recConformalTime:.6f}, s_init={s_init:.4e}")

# k grid (same as in compute_U_matrices)
# kvalues = np.linspace(1e-5, 15, num=300)
kvalues = np.linspace(10, 15, num=10)

# Initial condition: column n=4 of ABC matrix  (vr = 1)
IC_COL = 4
x0_base = np.zeros(NV)
x0_base[IC_COL] = 1.0
inits_base = np.concatenate(([s_init], x0_base))   # shape (NV+1,)

# ─────────────────────────────────────────────────────────────────────────────
# scipy ODE functions (k as a closure variable)
# ─────────────────────────────────────────────────────────────────────────────
def make_scipy_odes(k):
    def dX2(t, X):
        s = X[0]
        sdot = -H0*np.sqrt(OmegaLambda + OmegaK*abs(s**2)
                            + OmegaM*abs(s**3) + OmegaR*abs(s**4))
        rho_m = 3*H0**2*OmegaM*abs(s)**3
        rho_r = 3*H0**2*OmegaR*abs(s)**4
        phi, psi, dr, dm, vr, vm, fr2 = X[1:8]
        phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
        fr2dot = -(8/15)*k**2*vr - 0.6*k*X[8]
        psidot = phidot - (1/k**2)*(6*H0**2*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
        drdot  = (4/3)*(3*phidot + k**2*vr)
        dmdot  = 3*phidot + vm*k**2
        vrdot  = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
        vmdot  = (sdot/s)*vm - psi
        derivs = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
        for j in range(8, NV):
            l = j - 5
            derivs.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
        derivs.append(k*X[NV-1] - ((NV-5+1)*X[NV])/t)
        return derivs

    def dX3(t, X):
        sig = X[0]
        sigmadot = -H0*np.sqrt(OmegaLambda*np.exp(-2*sig) + OmegaK
                               + OmegaM*np.exp(sig) + OmegaR*np.exp(2*sig))
        rho_m = 3*H0**2*OmegaM*np.exp(3*sig)
        rho_r = 3*H0**2*OmegaR*np.exp(4*sig)
        phi, psi, dr, dm, vr, vm, fr2 = X[1:8]
        phidot = sigmadot*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sig))
        fr2dot = -(8/15)*k**2*vr - (3/5)*k*X[8]
        psidot = phidot - (1/k**2)*(6*H0**2*OmegaR*np.exp(sig))*(
                     sigmadot*np.exp(sig)*fr2 + 0.5*np.exp(sig)*fr2dot)
        drdot  = (4/3)*(3*phidot + k**2*vr)
        dmdot  = 3*phidot + vm*k**2
        vrdot  = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
        vmdot  = sigmadot*vm - psi
        derivs = [sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]
        for j in range(8, NV):
            l = j - 5
            derivs.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))
        derivs.append(k*X[NV-1] - ((NV-5+1)*X[NV])/t)
        return derivs

    return dX2, dX3


def solve_single_k_scipy(k):
    dX2, dX3 = make_scipy_odes(k)
    sol1 = solve_ivp(dX2, [endtime, swaptime], inits_base,
                     method='LSODA', atol=atol_ode, rtol=rtol_ode)
    y1 = sol1.y[:, -1].copy()
    y1[0] = np.log(y1[0])
    sol2 = solve_ivp(dX3, [swaptime, recConformalTime], y1,
                     method='LSODA', atol=atol_ode, rtol=rtol_ode)
    return sol2.y[1:, -1]   # perturbation variables only


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 1 – scipy sequential loop
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK 1 — scipy sequential (300 k values)")
print("="*60)
t0_bench = time.perf_counter()
results_scipy_seq = [solve_single_k_scipy(k) for k in kvalues]
t_scipy_seq = time.perf_counter() - t0_bench
results_scipy_seq = np.array(results_scipy_seq)
print(f"  Time: {t_scipy_seq:.3f} s")
print(f"  First k result (6 vars): {results_scipy_seq[0, :6]}")
print(f"  Last  k result (6 vars): {results_scipy_seq[-1, :6]}")

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 2 – JAX (Fully Optimized)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK 2 — JAX Optimized (300 k values)")
print("="*60)

import jax
import jax.numpy as jnp
import diffrax

jax.config.update("jax_enable_x64", True)

# Pre-compute static slicing arrays to eliminate 'gather' indexing
_ls = jnp.arange(3, NV - 5)
_ls_plus_1 = _ls + 1
_ls_factor = 1.0 / (2 * _ls + 1)

_H0  = jnp.float64(H0)
_OL  = jnp.float64(OmegaLambda)
_OK  = jnp.float64(OmegaK)
_OM  = jnp.float64(OmegaM)
_OR  = jnp.float64(OmegaR)
_t_end = jnp.float64(endtime)
_t_sw  = jnp.float64(swaptime)
_t_rec = jnp.float64(recConformalTime)

def dX2_jax(t, X, args):
    k = args
    s = X[0]
    
    sdot = -_H0 * jnp.sqrt(_OL + _OK * s**2 + _OM * jnp.abs(s**3) + _OR * s**4)
    rho_m = 3 * _H0**2 * _OM * jnp.abs(s)**3
    rho_r = 3 * _H0**2 * _OR * s**4
    
    phi, psi, dr, dm, vr, vm, fr2 = X[1], X[2], X[3], X[4], X[5], X[6], X[7]
    
    phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
    fr2dot = -(8/15)*k**2*vr - 0.6*k*X[8]
    psidot = phidot - (1/k**2)*(6*_H0**2*_OR*s)*(sdot*fr2 + 0.5*s*fr2dot)
    drdot  = (4/3)*(3*phidot + k**2*vr)
    dmdot  = 3*phidot + vm*k**2
    vrdot  = -(psi + dr/4) + (1 + 3*_OK*_H0**2/k**2)*fr2/2
    vmdot  = (sdot/s)*vm - psi

    # Vectorized Boltzmann hierarchy (eliminates X_pad and gather ops!)
    hier = (k * _ls_factor) * (_ls * X[7:NV-1] - _ls_plus_1 * X[9:NV+1])
    lastderiv = jnp.array([k * X[NV-1] - ((NV - 4) * X[NV]) / t])

    return jnp.concatenate([
        jnp.array([sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]),
        hier,
        lastderiv
    ])

def dX3_jax(t, X, args):
    k = args
    sig = X[0]
    
    exp_sig = jnp.exp(sig)
    exp_2sig = jnp.exp(2*sig)
    exp_3sig = jnp.exp(3*sig)
    exp_4sig = jnp.exp(4*sig)
    
    sigmadot = -_H0 * jnp.sqrt(_OL / exp_2sig + _OK + _OM * exp_sig + _OR * exp_2sig)
    rho_m = 3 * _H0**2 * _OM * exp_3sig
    rho_r = 3 * _H0**2 * _OR * exp_4sig
    
    phi, psi, dr, dm, vr, vm, fr2 = X[1], X[2], X[3], X[4], X[5], X[6], X[7]
    
    phidot = sigmadot * psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*exp_2sig)
    fr2dot = -(8/15)*k**2*vr - (3/5)*k*X[8]
    psidot = phidot - (1/k**2)*(6*_H0**2*_OR*exp_sig)*(
                 sigmadot*exp_sig*fr2 + 0.5*exp_sig*fr2dot)
    drdot  = (4/3)*(3*phidot + k**2*vr)
    dmdot  = 3*phidot + vm*k**2
    vrdot  = -(psi + dr/4) + (1 + 3*_OK*_H0**2/k**2)*fr2/2
    vmdot  = sigmadot * vm - psi

    hier = (k * _ls_factor) * (_ls * X[7:NV-1] - _ls_plus_1 * X[9:NV+1])
    lastderiv = jnp.array([k * X[NV-1] - ((NV - 4) * X[NV]) / t])

    return jnp.concatenate([
        jnp.array([sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]),
        hier,
        lastderiv
    ])

@jax.jit
def solve_batch_jax(k_arr, y0):
    # Used high order explicit solver to combat your extremely tight 1e-13 tolerance!
    solver = diffrax.Dopri8() 
    
    def solve_one_k(k):
        sol1 = diffrax.diffeqsolve(
            diffrax.ODETerm(dX2_jax),
            solver,
            t0=_t_end, t1=_t_sw,
            dt0=-1e-5,
            y0=y0,
            args=k,
            stepsize_controller=diffrax.PIDController(rtol=rtol_ode, atol=atol_ode),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=int(1e5),
        )
        y1 = sol1.ys[0]
        y1 = y1.at[0].set(jnp.log(y1[0]))

        sol2 = diffrax.diffeqsolve(
            diffrax.ODETerm(dX3_jax),
            solver,
            t0=_t_sw, t1=_t_rec,
            dt0=-1e-5,
            y0=y1,
            args=k,
            stepsize_controller=diffrax.PIDController(rtol=rtol_ode, atol=atol_ode),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=int(1e5),
        )
        return sol2.ys[0][1:]
    
    # jax.lax.map sequentially loops over k without "lock-step dummy evaluations" but retains XLA C++ speeds. 
    # Try swapping to jax.vmap(solve_one_k)(k_arr) if executing on a powerful multi-core GPU
    return jax.lax.map(solve_one_k, k_arr) 

y0_jax = jnp.array(inits_base)
k_arr  = jnp.array(kvalues)

print("  JIT compile (warm-up) ...")
t0_compile = time.perf_counter()
_ = solve_batch_jax(k_arr, y0_jax)
jax.block_until_ready(_)
t_compile = time.perf_counter() - t0_compile
print(f"  JIT compile time: {t_compile:.3f} s")

print("  Timed run ...")
t0_bench = time.perf_counter()
results_jax = solve_batch_jax(k_arr, y0_jax)
jax.block_until_ready(results_jax)
t_jax = time.perf_counter() - t0_bench
results_jax_np = np.array(results_jax)

print(f"  Time (after JIT): {t_jax:.3f} s")
print(f"  First k result (6 vars): {results_jax_np[0, :6]}")
print(f"  Last  k result (6 vars): {results_jax_np[-1, :6]}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  scipy sequential  : {t_scipy_seq:.3f} s")
print(f"  JAX vmap (compile): {t_compile:.3f} s  (first call, includes JIT)")
print(f"  JAX vmap (timed)  : {t_jax:.3f} s  ({t_scipy_seq/t_jax:.1f}x vs sequential)")

print("\n--- Accuracy (JAX vmap vs scipy sequential) ---")
abs_err = np.abs(results_scipy_seq - results_jax_np)
rel_err = abs_err / (np.abs(results_scipy_seq) + 1e-30)
print(f"  Max absolute error : {abs_err.max():.3e}")
print(f"  Max relative error : {rel_err.max():.3e}")
print(f"  Mean relative error: {rel_err.mean():.3e}")