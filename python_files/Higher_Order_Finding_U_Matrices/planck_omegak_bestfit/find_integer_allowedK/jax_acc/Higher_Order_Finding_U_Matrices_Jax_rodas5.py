# -*- coding: utf-8 -*-
"""
Refactored to use JAX & custom Rodas5 Rosenbrock solver for stiff equations.
"""

import numpy as np
import time

import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Import the modified custom Rodas5 solver
import rodas5

# Ensure JAX uses 64-bit precision to maintain the 1e-13 tolerance required
jax.config.update("jax_enable_x64", True)

def compute_U_matrices_Jax(params, z_rec, folder_path, n_processes=None, kvalues=None):
    
    # Use default kvalues if none are provided
    if kvalues is None:
        kvalues = np.linspace(1.0, 15.0, num=300)

    mt, kt, omega_b_ratio, h = params
    Omega_gamma_h2 = 2.47e-5
    Neff = 3.046

    def cosmological_parameters(mt, kt, h):
        Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2
        def solve_a0(Omega_r, rt, mt, kt):
            def f(a0):
                return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
            sol = root_scalar(f, bracket=[1, 1.e3])
            return sol.root
        a0 = solve_a0(Omega_r, 1, mt, kt)
        s0 = 1/a0
        Omega_lambda = Omega_r * a0**4
        Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
        Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
        return s0, Omega_lambda, Omega_m, Omega_K

    s0, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
    OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2

    atol = 1e-13
    rtol = 1e-13
    stol = 1e-10
    num_variables = 75
    swaptime = 2.0
    deltaeta = 6.6e-4
    H0 = 1/np.sqrt(3*OmegaLambda)
    Hinf = H0*np.sqrt(OmegaLambda)

    def ds_dt(t, s):
        return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    t0 = 1e-8
    smin1 = np.sqrt(3*OmegaLambda/OmegaR)
    szero = - OmegaM/(4*OmegaR)
    s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
    s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) 
    s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)))
    s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

    s_IC = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4

    print('Performing Initial Background Integration...')
    def reach_FCB(t, s): return s[0]
    reach_FCB.terminal = True

    sol = solve_ivp(ds_dt,[t0,12],[s_IC], max_step=0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
    
    if sol.t_events and len(sol.t_events[0]) > 0:
        fcb_time = sol.t_events[0][0]
    else:
        fcb_time = None 

    endtime = fcb_time - deltaeta
    s_rec = 1 + z_rec  
    recScaleFactorDifference = abs(sol.y[0] - s_rec)
    recConformalTime = sol.t[recScaleFactorDifference.argmin()]

    def at_fcb(t,X):
        if X[0]<stol: X[0] = 0
        return X[0]
    at_fcb.terminal = True

    kvalues = np.linspace(10, 15, num=10)
    sol2 = solve_ivp(ds_dt, [t0,endtime], [s_IC], method='LSODA', events=at_fcb, atol=atol, rtol=rtol)
    s_init = sol2.y[0,-1]

    # -------------------------------------------------------------------------
    # RODAS5 SETUP (Signatures rearranged to (X, t, args))
    # -------------------------------------------------------------------------
    t_start = time.perf_counter()

    NV = num_variables
    _ls = jnp.arange(3, NV - 5)
    _ls_plus_1 = _ls + 1
    _ls_factor = 1.0 / (2 * _ls + 1)

    _H0  = jnp.float64(H0)
    _OL  = jnp.float64(OmegaLambda)
    _OK  = jnp.float64(OmegaK)
    _OM  = jnp.float64(OmegaM)
    _OR  = jnp.float64(OmegaR)

    # Rodas5 requires (y, t, params)
    def dX2_jax(X, t, k):
        s = X[0]
        sdot = -_H0 * jnp.sqrt(_OL + _OK * s**2 + _OM * jnp.abs(s**3) + _OR * s**4)
        rho_m = 3 * _H0**2 * _OM * jnp.abs(s)**3
        rho_r = 3 * _H0**2 * _OR * s**4
        
        phi, psi, dr, dm, vr, vm, fr2 = X[1:8]
        phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
        fr2dot = -(8/15)*k**2*vr - 0.6*k*X[8]
        psidot = phidot - (1/k**2)*(6*_H0**2*_OR*s)*(sdot*fr2 + 0.5*s*fr2dot)
        drdot  = (4/3)*(3*phidot + k**2*vr)
        dmdot  = 3*phidot + vm*k**2
        vrdot  = -(psi + dr/4) + (1 + 3*_OK*_H0**2/k**2)*fr2/2
        vmdot  = (sdot/s)*vm - psi

        hier = (k * _ls_factor) * (_ls * X[7:NV-1] - _ls_plus_1 * X[9:NV+1])
        lastderiv = jnp.array([k * X[NV-1] - ((NV - 4) * X[NV]) / t])

        return jnp.concatenate([
            jnp.array([sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]),
            hier, lastderiv
        ])

    def dX3_jax(X, t, k):
        sig = X[0]
        exp_sig = jnp.exp(sig); exp_2sig = jnp.exp(2*sig)
        exp_3sig = jnp.exp(3*sig); exp_4sig = jnp.exp(4*sig)
        
        sigmadot = -_H0 * jnp.sqrt(_OL / exp_2sig + _OK + _OM * exp_sig + _OR * exp_2sig)
        rho_m = 3 * _H0**2 * _OM * exp_3sig
        rho_r = 3 * _H0**2 * _OR * exp_4sig
        
        phi, psi, dr, dm, vr, vm, fr2 = X[1:8]
        phidot = sigmadot * psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*exp_2sig)
        fr2dot = -(8/15)*k**2*vr - (3/5)*k*X[8]
        psidot = phidot - (1/k**2)*(6*_H0**2*_OR*exp_sig)*(sigmadot*exp_sig*fr2 + 0.5*exp_sig*fr2dot)
        drdot  = (4/3)*(3*phidot + k**2*vr)
        dmdot  = 3*phidot + vm*k**2
        vrdot  = -(psi + dr/4) + (1 + 3*_OK*_H0**2/k**2)*fr2/2
        vmdot  = sigmadot * vm - psi

        hier = (k * _ls_factor) * (_ls * X[7:NV-1] - _ls_plus_1 * X[9:NV+1])
        lastderiv = jnp.array([k * X[NV-1] - ((NV - 4) * X[NV]) / t])

        return jnp.concatenate([
            jnp.array([sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]),
            hier, lastderiv
        ])

    @jax.jit
    def process_all_k(k_arr):
        n_k = k_arr.shape[0]

        # 1. GENERATE ALL ICS FOR ALL Ks SIMULTANEOUSLY (Shape: 300 x 9 x 76)
        def create_ics(k):
            abc_x0 = jnp.eye(NV)[:6]
            def_x0 = jnp.eye(NV)[6:8] 
            x3 = -(16.0 / 945.0) * (k**4) * (deltaeta**3)
            ghi_x0 = jnp.zeros((1, NV)).at[0, 8].set(x3) 
            all_x0 = jnp.concatenate([abc_x0, def_x0, ghi_x0], axis=0)  
            s_col = jnp.full((9, 1), s_init)
            return jnp.concatenate([s_col, all_x0], axis=1)  
        
        all_inits = jax.vmap(create_ics)(k_arr)

        # 2. FLATTEN TO BATCH 2700 TRAJECTORIES (N_k * 9)
        all_inits_flat = all_inits.reshape(-1, NV + 1)
        k_flat = jnp.repeat(k_arr, 9)

        # 3. SOLVE PHASE 1 (endtime -> swaptime)
        res_2 = rodas5.solve(
            dX2_jax,
            all_inits_flat,
            jnp.array([endtime, swaptime]),
            k_flat,
            rtol=rtol, atol=atol,
            max_steps=500000
        )
        # Extract state at 'swaptime'
        y_sw_flat = res_2[:, -1, :] 

        # 4. LOG TRANSFORM S TO SIGMA
        y_sw_flat = y_sw_flat.at[:, 0].set(jnp.log(y_sw_flat[:, 0]))

        # 5. SOLVE PHASE 2 (swaptime -> recConformalTime)
        res_3 = rodas5.solve(
            dX3_jax,
            y_sw_flat,
            jnp.array([swaptime, recConformalTime]),
            k_flat,
            rtol=rtol, atol=atol,
            max_steps=500000
        )
        
        # 6. EXTRACT RESULTS AND RESHAPE BACK
        y_final_flat = res_3[:, -1, 1:]  # Drop the background variable
        y_final = y_final_flat.reshape(n_k, 9, NV)

        def compute_matrices(k, y_final_k):
            ABC_matrix = y_final_k[0:6].T
            DEF_matrix = y_final_k[6:8].T
            GHI_vector = y_final_k[8]

            coeff1 = (Hinf**3)*(OmegaM/OmegaLambda)
            coeff2 = (Hinf**4)*(OmegaR/OmegaLambda)
            coeff3 = (Hinf**2)*(OmegaK/OmegaLambda)
            denom = k**2 + 3*coeff3

            de = deltaeta; de2 = de**2; de3 = de**3; de4 = de**4; de5 = de**5; de6 = de**6
            
            # X1 Row 0
            phi4_dr = (8*k**2*coeff2 + 24*coeff2*coeff3)/(80*denom)
            phi4_dm = -75*coeff1**2 / (80*denom)
            phi5_dm = -35*coeff1*(56*coeff2 + coeff3**2) / (2800*denom)
            phi4_vr = 300*coeff1*coeff2 / (80*denom)
            phi5_vr = -(260*k**2*coeff2 + 644*coeff2*coeff3 - 140*coeff2*(coeff3**2 + 56*coeff2)) / (2800*denom)
            x00 = phi4_dr*de4
            x01 = -(3*coeff1)/(2*denom)*de - (5*coeff1*coeff3)/(20*denom)*de3 + phi4_dm*de4 + phi5_dm*de5
            x02 = (6*coeff2)/denom*de + (4*k**2*coeff2 + 32*coeff2*coeff3)/(20*denom)*de3 + phi4_vr*de4 + phi5_vr*de5
            x03 = -(9*coeff1)/(2*denom)*de - (15*k**2*coeff1 + 60*coeff1*coeff3)/(20*denom)*de3 + 3*phi4_dm*de4 + (-(525*coeff1*coeff3)/(2800*denom) + 3*phi5_dm)*de5
            x10 = x00; x11 = x01; x12 = x02 + (24*k**2*coeff2 + 32*coeff2*coeff3)/(20*denom)*de3; x13 = x03

            dr5_dm = (210*k**2*coeff1*coeff3 - 630*k**4*coeff1 - 1575*coeff1*(56*coeff2 + coeff3**2)) / (31500*denom)
            dr6_dm = -(k**2*coeff1**2)/(12*denom)
            dr5_vr = -(150*k**8 + 858*k**6*coeff3 - 10080*coeff2*(-8*coeff3**2 + 35*coeff2) + 48*k**2*coeff3*(14*coeff3**2 + 1195*coeff2) + 4*k**4*(362*coeff3**2 + 1665*coeff2)) / (31500*denom)
            dr6_vr = -(4*k**2*coeff1*coeff2)/(12*denom)
            dr5_vm = (-13545*k**2*coeff1*coeff3 + 1260*k**4*coeff1 - 1575*coeff1*(168*coeff2 + 48*coeff3**2)) / (31500*denom)
            x20 = 1 - (1/6)*k**2*de2 + ((1/120)*k**4 + (k**2*coeff3)/90 + 0.4*coeff2)*de4 - (75*k**6 + 204*k**4*coeff3 + 112*k**2*coeff3**2)/378000*de6
            x21 = -(6*coeff1)/denom*de + (30*k**2*coeff1 - 45*coeff1*coeff3)/(45*denom)*de3 + dr5_dm*de5 + dr6_dm*de6
            x22 = -(4*k**4 + 12*k**2*coeff3 - 72*coeff2)/(3*denom)*de + (6*k**6 + 26*k**4*coeff3 + 288*coeff2*coeff3 + 12*k**2*(2*coeff3**2 - 7*coeff2))/(45*denom)*de3 + dr5_vr*de5 + dr6_vr*de6
            x23 = -(18*coeff1)/denom*de - (k**2*coeff1 + 12*coeff1*coeff3)/denom*de3 + dr5_vm*de5 + 3*dr6_dm*de6

            dm5_dm = -105*coeff1*(-4*k**2*coeff3 + 3*(coeff3**2 + 56*coeff2)) / (8400*denom)
            dm6_dm = -(k**2*coeff1**2)/(16*denom)
            x30 = (72*coeff2/240)*de4 + (3*k**2*coeff2/720)*de6
            x31 = 1 - (9*coeff1)/(2*denom)*de + (10*k**2*coeff1 - 15*coeff1*coeff3)/(20*denom)*de3 + dm5_dm*de5 + dm6_dm*de6
            x32 = (18*coeff2)/denom*de - (28*k**2*coeff2 - 96*coeff2*coeff3)/(20*denom)*de3 + (-(1556*k**2*coeff2 + 5796*coeff2*coeff3) + 420*coeff2*(-4*k**2*coeff3 + 3*(coeff3**2 + 56*coeff2))/denom)/8400*de5
            x33 = -(27*coeff1)/(2*denom)*de + 0.5*k**2*de2 - (15*k**2*coeff1 + 180*coeff1*coeff3)/(20*denom)*de3 + (k**2*coeff3/24)*de4 + ((630*k**2*coeff1 - 4725*coeff1*coeff3)/8400 + 3*dm5_dm)*de5 + (k**2*(coeff3**2 + 12*coeff2)/720 + 3*dm6_dm)*de6

            vr5_dm = 3*coeff1**2 / (8*denom)
            vr6_dm = -(7875*k**4*coeff1 + 10395*k**2*coeff1*coeff3 + 2205*coeff1*(200*coeff2 + 7*coeff3**2)) / (4410000*denom)
            vr6_vr_num = 1935*k**8 + 13605*k**6*coeff3 + 36*k**4*(902*coeff3**2 - 2485*coeff2) + 112*k**2*coeff3*(271*coeff3**2 + 2880*coeff2) + 4704*(2*coeff3**4 + 330*coeff2*coeff3**2 - 375*coeff2**2)
            vr6_vm = (9450*k**4*coeff1 + 163485*k**2*coeff1*coeff3 + 2205*coeff1*(600*coeff2 + 336*coeff3**2)) / (4410000*denom)
            x40 = 0.25*de - (3*k**2 + 4*coeff3)/120*de3 + (75*k**4 + 204*k**2*coeff3 + 112*coeff3**2)/84000*de5
            x41 = -(15*coeff1)/(10*denom)*de2 + (315*k**2*coeff1 - 105*coeff1*coeff3)/(4200*denom)*de4 + vr5_dm*de5 + vr6_dm*de6
            x42 = 1 - (3*k**4 + 13*k**2*coeff3 + 12*(coeff3**2 - 5*coeff2))/(10*denom)*de2 + (75*k**6 + 429*k**4*coeff3 + 4*k**2*(181*coeff3**2 - 630*coeff2) + 336*coeff3*(coeff3**2 - 10*coeff2))/(4200*denom)*de4 + (12*coeff1*coeff2)/(8*denom)*de5 - vr6_vr_num/(4410000*denom)*de6
            x43 = -(45*coeff1)/(10*denom)*de2 - (630*k**2*coeff1 + 5040*coeff1*coeff3)/(4200*denom)*de4 + 3*vr5_dm*de5 + vr6_vm*de6

            vm5_dm = 45*coeff1**2 / (120*denom)
            x50 = -(3*coeff2/120)*de5
            x51 = -(3*coeff1)/(2*denom)*de2 - (30*coeff1*coeff3)/(120*denom)*de4 + vm5_dm*de5
            x52 = (6*coeff2)/denom*de2 - (56*k**2*coeff2 + 48*coeff2*coeff3)/(120*denom)*de4 + (180*coeff1*coeff2)/(120*denom)*de5
            x53 = -de - (9*coeff1)/(2*denom)*de2 - (coeff3/6)*de3 - (45*k**2*coeff1 + 225*coeff1*coeff3)/(120*denom)*de4 + (-(coeff3**2 + 12*coeff2)/120 + 3*vm5_dm)*de5
            X1 = jnp.array([[x00, x01, x02, x03],[x10, x11, x12, x13],[x20, x21, x22, x23],[x30, x31, x32, x33],[x40, x41, x42, x43],[x50, x51, x52, x53]])
            
            fr2_5_dm = (3150*k**2*coeff1 - 735*coeff1*coeff3) / (275625*denom)
            fr2_5_vr = (795*k**6 + 4065*k**4*coeff3 + 28*k**2*(208*coeff3**2 - 765*coeff2) + 2352*coeff3*(coeff3**2 - 10*coeff2)) / (275625*denom)
            fr2_5_vm = (-1575*k**2*coeff1 - 735*48*coeff1*coeff3) / (275625*denom)
            fr2_6_dm = k**2*coeff1**2 / (30*denom)
            x00 = (1/15)*k**2*de2 + k**2*(-15*k**2 - 14*coeff3)/3150*de4 + (795*k**6 + 1680*k**4*coeff3 + 784*k**2*coeff3**2)/6615000*de6
            x01 = -(4*k**2*coeff1)/(15*denom)*de3 + fr2_5_dm*de5 + fr2_6_dm*de6
            x02 = (8/15)*k**2*de - 4*k**2*(30*k**4 + 118*k**2*coeff3 + 84*(coeff3**2 - 5*coeff2))/(1575*denom)*de3 + fr2_5_vr*de5 + (4*k**2*coeff1*coeff2)/(30*denom)*de6
            x03 = -(12*k**2*coeff1)/(15*denom)*de3 + fr2_5_vm*de5 + 3*fr2_6_dm*de6
            
            fr3_6_dm = k**3*(3150*k**2*coeff1 - 735*coeff1*coeff3 + 2205*coeff1*(200*coeff2 + 7*coeff3**2)) / (3858750*denom)
            fr3_6_vr_num = 795*k**6 + 4065*k**4*coeff3 + 28*k**2*(208*coeff3**2 - 765*coeff2) + 2352*coeff3*(coeff3**2 - 10*coeff2)
            fr3_6_vm = k**3*(1575*k**2*coeff1 + 735*48*coeff1*coeff3 - 2205*coeff1*(600*coeff2 + 336*coeff3**2)) / (3858750*denom)
            x10 = -(1/105)*k**3*de3 + k**3*(15*k**2 + 14*coeff3)/36750*de5
            x11 = (k**3/35)*coeff1/denom*de4 + fr3_6_dm*de6
            x12 = -(4/35)*k**3*de2 + k**3*(30*k**4 + 118*k**2*coeff3 + 84*(coeff3**2 - 5*coeff2))/(3675*denom)*de4 - k**3*fr3_6_vr_num/(3858750*denom)*de6
            x13 = (3*k**3*coeff1)/(35*denom)*de4 + fr3_6_vm*de6
            X2 = jnp.array([[x00, x01, x02, x03],[x10, x11, x12, x13]])

            return ABC_matrix, DEF_matrix, GHI_vector, X1, X2

        return jax.vmap(compute_matrices)(k_arr, y_final)

    # FIRE THE PIPELINE!
    k_arr = jnp.array(kvalues)
    results = process_all_k(k_arr)

    ABCmatrices = np.array(results[0])
    DEFmatrices = np.array(results[1])
    GHIvectors  = np.array(results[2])
    X1matrices  = np.array(results[3])
    X2matrices  = np.array(results[4])

    print(f"Parallel computation completed in {time.perf_counter() - t_start:.3f} seconds.")
    
    np.save(folder_path + 'L70_kvalues', kvalues)
    np.save(folder_path + 'L70_ABCmatrices', ABCmatrices)
    np.save(folder_path + 'L70_DEFmatrices', DEFmatrices)
    np.save(folder_path + 'L70_GHIvectors', GHIvectors)
    np.save(folder_path + 'L70_X1matrices', X1matrices)
    np.save(folder_path + 'L70_X2matrices', X2matrices)

    return {
        'kvalues': kvalues, 'ABCmatrices': ABCmatrices, 'DEFmatrices': DEFmatrices,
        'GHIvectors': GHIvectors, 'X1matrices': X1matrices, 'X2matrices': X2matrices
    }