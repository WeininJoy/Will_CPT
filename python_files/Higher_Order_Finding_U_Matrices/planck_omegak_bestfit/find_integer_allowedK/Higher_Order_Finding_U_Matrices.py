# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021

@author: MRose
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from multiprocessing import Pool, cpu_count
from functools import partial

def process_single_k(k, s_init, endtime, swaptime, recConformalTime,
                     H0, OmegaLambda, OmegaM, OmegaR, OmegaK, Hinf,
                     s0, deltaeta, atol, rtol, num_variables):
    """
    Process a single k value to compute ABC matrix, DEF matrix, GHI vector, X1 matrix, and X2 matrix.

    This function is designed to be called in parallel for different k values.
    """

    # Define derivative functions (copied from main function)
    def dX2_dt(t,X):
        s,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
        sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

        rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
        rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)

        phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
        fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
        psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
        drdot = (4/3)*(3*phidot + (k**2)*vr)
        dmdot = 3*phidot + vm*(k**2)
        vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
        vmdot = (sdot/s)*vm - psi
        derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

        for j in range(8,num_variables):
            l = j - 5
            derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

        lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
        derivatives.append(lastderiv)
        return derivatives

    def dX3_dt(t,X):
        sigma,phi,psi,dr,dm,vr,vm,fr2 = X[0:8]
        sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaK+OmegaM*np.exp(sigma)
                                +OmegaR*np.exp(2*sigma)))

        rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
        rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))

        phidot = (sigmadot)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
        fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[8]
        psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*np.exp(sigma))*(sigmadot*np.exp(sigma)*fr2 + 0.5*np.exp(sigma)*fr2dot)
        drdot = (4/3)*(3*phidot + (k**2)*vr)
        dmdot = 3*phidot + vm*(k**2)
        vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
        vmdot = (sigmadot)*vm - psi
        derivatives = [sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

        for j in range(8,num_variables):
            l = j - 5
            derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

        lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
        derivatives.append(lastderiv)
        return derivatives

    print(f"Processing k = {k}")

    #---------------------------------------------------------------------------------------
    # For each K, find ABCmatrix
    #---------------------------------------------------------------------------------------

    ABC_matrix = np.zeros(shape=(num_variables, 6))

    for n in range(6):
        x0 = np.zeros(num_variables)
        x0[n] = 1
        inits = np.concatenate(([s_init], x0))

        solperf = solve_ivp(dX2_dt, [endtime,swaptime], inits, method='LSODA', atol=atol, rtol=rtol)

        inits2 = solperf.y[:,-1]
        inits2[0] = np.log(inits2[0])

        solperf2 = solve_ivp(dX3_dt, [swaptime,recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)

        nth_col = []
        for m in range(1,num_variables+1):
            nth_col.append(solperf2.y[m,-1])

        ABC_matrix[:,n] = nth_col

    #---------------------------------------------------------------------------
    # FOR each K, find DEF matrix
    #---------------------------------------------------------------------------

    DEF_matrix = np.zeros(shape=(num_variables, 2))

    for j in range(0,2):
        x0 = np.zeros(num_variables)
        inits = np.concatenate(([s_init], x0))
        inits[j+7] = 1

        sol3 = solve_ivp(dX2_dt, [endtime,swaptime], inits, method='LSODA', atol=atol, rtol=rtol)

        inits2 = sol3.y[:,-1]
        inits2[0] = np.log(inits2[0])

        sol4 = solve_ivp(dX3_dt, [swaptime,recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)

        nthcol = sol4.y[:,-1]
        nthcol = np.array(nthcol)
        nthcol = np.delete(nthcol, 0)

        DEF_matrix[:,j] = nthcol

    #----------------------------------------------------------------------------
    # Now find GHIx3 vectors by setting v_r^\infty to 1
    #----------------------------------------------------------------------------

    x3 = -(16/945)*(k**4)*(deltaeta**3)
    x0 = np.zeros(num_variables)
    inits = np.concatenate(([s_init], x0))
    inits[9] = x3

    sol5 = solve_ivp(dX2_dt, [endtime,swaptime], inits, method='LSODA', atol=atol, rtol=rtol)

    inits2 = sol5.y[:,-1]
    inits2[0] = np.log(inits2[0])

    sol6 = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits2, method='LSODA', atol=atol, rtol=rtol)

    vec = np.array(sol6.y[:,-1])
    vec = np.delete(vec, 0)
    GHI_vector = vec

    #-----------------------------------------------------------------------------------------------
    # Define X1 matrix for calculating x_endtimes from fcb values
    #-----------------------------------------------------------------------------------------------
    k2 = k**2
    k3 = k**3
    k4 = k**4
    k6 = k**6

    de = deltaeta
    de2 = deltaeta**2
    de3 = deltaeta**3
    de4 = deltaeta**4

    coeff1 = (Hinf**3) * (OmegaM / OmegaLambda) / s0**3
    coeff2 = (Hinf**4) * (OmegaR / OmegaLambda) / s0**4
    coeff3 = (Hinf**2) * (OmegaK / OmegaLambda) / s0**2

    common_denom = (k2 + 3 * coeff3)

    # Row 0: phi'
    x00 = 0.0
    phi1_dm = -3 * coeff1 / (2 * common_denom)
    phi3_dm = - (5 * coeff1 * coeff3) / (20 * common_denom)
    x01 = phi1_dm * de + phi3_dm * de3
    phi1_vr = -6 * coeff2 / common_denom
    phi3_vr = -(4 * k2 * coeff2 + 32 * coeff2 * coeff3) / (20 * common_denom)
    x02 = phi1_vr * de + phi3_vr * de3
    phi1_vm = -9 * coeff1 / (2 * common_denom)
    phi3_vm = -(15 * k2 * coeff1 + 60 * coeff1 * coeff3) / (20 * common_denom)
    x03 = phi1_vm * de + phi3_vm * de3

    # Row 1: psi'
    psi4_dr = -(coeff2 / 5) * de4
    x10 = psi4_dr
    psi1_dm = phi1_dm
    psi3_dm = (5 * coeff1 * coeff3) / (20 * common_denom)
    x11 = psi1_dm * de + psi3_dm * de3
    psi1_vr = phi1_vr
    psi3_vr = (28 * k2 * coeff2 + 64 * coeff3 * coeff2) / (20 * common_denom)
    x12 = psi1_vr * de + psi3_vr * de3
    psi1_vm = phi1_vm
    psi3_vm = (15 * k2 * coeff1) / (20 * common_denom)
    x13 = psi1_vm * de + psi3_vm * de3

    # Row 2: dr'
    dr2_dr = -1/6 * k2
    dr4_dr = (1/120 * k4) + (k2 * coeff3 / 90)
    x20 = 1.0 + dr2_dr * de2 + dr4_dr * de4
    dr1_dm = -6 * coeff1 / common_denom
    dr3_dm = -(-2 * k2 * coeff1 + 45 * coeff1 * coeff3) / (45 * common_denom)
    x21 = dr1_dm * de + dr3_dm * de3
    dr1_vr = (4 * k4 + 12 * k2 * coeff3 - 72 * coeff2) / (3 * common_denom)
    dr3_vr = -(6 * k6 + 26 * k4 * coeff3 + 288 * coeff2 * coeff3 + 12 * k2 * (2 * coeff3**2 - 7 * coeff2)) / (45 * common_denom)
    x22 = dr1_vr * de + dr3_vr * de3
    dr1_vm = -18 * coeff1 / common_denom
    dr3_vm = -(3 * k2 * coeff1 + 540 * coeff1 * coeff3) / (45 * common_denom)
    x23 = dr1_vm * de + dr3_vm * de3

    # Row 3: dm'
    x30 = 0.0
    dm1_dm = -9 * coeff1 / (2 * common_denom)
    dm3_dm = -(-10 * k2 * coeff1 + 15 * coeff1 * coeff3) / (20 * common_denom)
    x31 = 1.0 + dm1_dm * de + dm3_dm * de3
    dm1_vr = -18 * coeff2 / common_denom
    dm3_vr = -(-28 * k2 * coeff2 + 96 * coeff2 * coeff3) / (20 * common_denom)
    x32 = dm1_vr * de + dm3_vr * de3
    dm1_vm = -27 * coeff1 / (2 * common_denom)
    dm2_vm = 0.5 * k2
    dm3_vm = -(15 * k2 * coeff1 + 180 * coeff1 * coeff3) / (20 * common_denom)
    dm4_vm = (k2 * coeff3) / 24.0
    x33 = dm1_vm * de + dm2_vm * de2 + dm3_vm * de3 + dm4_vm * de4

    # Row 4: vr'
    vr1_dr = -1/4
    vr3_dr = (3 * k2 + 4 * coeff3) / 120.0
    x40 = vr1_dr * de + vr3_dr * de3
    vr2_dm = (15 * coeff1) / (10 * common_denom)
    vr4_dm = (-315 * k2 * coeff1 + 105 * coeff1 * coeff3) / (4200 * common_denom)
    x41 = vr2_dm * de2 + vr4_dm * de4
    vr2_vr = - (3 * k4 + 13 * k2 * coeff3 + 12 * (coeff3**2 - 5 * coeff2)) / (10 * common_denom)
    vr4_vr = (75 * k6 + 429 * k4 * coeff3 + 4 * k2 * (181 * coeff3**2 - 630 * coeff2) + 336 * coeff3 * (coeff3**2 - 10 * coeff2)) / (4200 * common_denom)
    x42 = 1.0 + vr2_vr * de2 + vr4_vr * de4
    vr2_vm = (45 * coeff1) / (10 * common_denom)
    vr4_vm = (630 * k2 * coeff1 + 5040 * coeff1 * coeff3) / (4200 * common_denom)
    x43 = vr2_vm * de2 + vr4_vm * de4

    # Row 5: vm'
    x50 = 0.0
    vm2_dm = 3 * coeff1 / (2 * common_denom)
    x51 = vm2_dm * de2
    vm2_vr = 6 * coeff2 / common_denom
    x52 = vm2_vr * de2
    vm3_vm = coeff3 / 6.0
    x53 = 1.0 * de + (9 * coeff1 / (2 * common_denom)) * de2 + vm3_vm * de3

    X1 = np.array([
        [x00, x01, x02, x03],
        [x10, x11, x12, x13],
        [x20, x21, x22, x23],
        [x30, x31, x32, x33],
        [x40, x41, x42, x43],
        [x50, x51, x52, x53]
    ])

    # Row 0: fr2'
    fr2_2_dr = 1/15 * k2
    fr2_4_dr = (k2 * (-15 * k2 - 14 * coeff3)) / 3150.0
    y00 = fr2_2_dr * de2 + fr2_4_dr * de4
    fr2_3_dm = -(4 * k2 * coeff1) / (15 * common_denom)
    y01 = fr2_3_dm * de3
    fr2_1_vr = -8/15 * k2
    fr2_3_vr = (4 * k2 * (30 * k4 + 118 * k2 * coeff3 + 84 * (coeff3**2 - 5 * coeff2))) / (1575 * common_denom)
    y02 = fr2_1_vr * de + fr2_3_vr * de3
    fr2_3_vm = -(12 * k2 * coeff1) / (15 * common_denom)
    y03 = fr2_3_vm * de3

    # Row 1: fr3'
    fr3_3_dr = 1/105 * k3
    y10 = fr3_3_dr * de3
    y11 = 0.0
    fr3_2_vr = -4/35 * k3
    fr3_4_vr = (k3 * (30 * k4 + 118 * k2 * coeff3 + 84 * (coeff3**2 - 5 * coeff2))) / (3675 * common_denom)
    y12 = fr3_2_vr * de2 + fr3_4_vr * de4
    fr3_4_vm = -(k3 * 3 * coeff1) / (35 * common_denom)
    y13 = fr3_4_vm * de4

    X2 = np.array([
        [y00, y01, y02, y03],
        [y10, y11, y12, y13]
    ])

    return (ABC_matrix, DEF_matrix, GHI_vector, X1, X2)

def compute_U_matrices(params, z_rec, folder_path, n_processes=None):
    """
    Compute ABC/DEF/GHI matrices and X1/X2 matrices for perturbation analysis.

    Parameters:
    -----------
    params : list or tuple
        [mt, kt, omega_b_ratio, h] cosmological parameters
    z_rec : float
        Recombination redshift
    folder_path : str
        Path to save output files
    nu_spacing : int, optional
        Spacing for allowed K values (default: 4)
    n_processes : int, optional
        Number of parallel processes to use. If None, uses all available CPU cores.

    Returns:
    --------
    dict : Dictionary containing kvalues, ABCmatrices, DEFmatrices, GHIvectors, X1matrices, X2matrices
    """

    # Unpack parameters
    mt, kt, omega_b_ratio, h = params

    # Constants
    lam = 1
    rt = 1
    Omega_gamma_h2 = 2.47e-5  # photon density
    Neff = 3.046

    def cosmological_parameters(mt, kt, h):
        Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

        def solve_a0(Omega_r, rt, mt, kt):
            def f(a0):
                return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
            sol = root_scalar(f, bracket=[1, 1.e3])
            return sol.root

        a0 = solve_a0(Omega_r, rt, mt, kt)
        s0 = 1/a0
        Omega_lambda = Omega_r * a0**4
        Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
        Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
        return s0, Omega_lambda, Omega_m, Omega_K

    s0, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
    OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    ###############################################################################

    #set tolerances
    atol = 1e-13
    rtol = 1e-13
    stol = 1e-10
    num_variables = 75  # number of pert variables
    swaptime = 2  #set time when we swap from s to sigma
    deltaeta = 6.6e-4
    H0 = 1/np.sqrt(3*OmegaLambda)  #we are working in units of Lambda=c=1
    Hinf = H0*np.sqrt(OmegaLambda)

    #```````````````````````````````````````````````````````````````````````````````
    #BACKGROUND EQUATIONS
    #```````````````````````````````````````````````````````````````````````````````

    #write derivative function for background
    def ds_dt(t, s):
        return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    t0 = 1e-8;

    #set coefficients for initial conditions
    smin1 = np.sqrt(3*OmegaLambda/OmegaR);
    szero = - OmegaM/(4*OmegaR);
    s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR));
    s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) ;
    s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
    s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

    s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

    print('Performing Initial Background Integration')
    def reach_FCB(t, s): return s[0]
    reach_FCB.terminal = True

    sol = solve_ivp(ds_dt, [t0,12], [s0], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
    print('Initial Background Integration Done')

    # Check if t_events[0] is not empty before trying to access its elements
    if sol.t_events and len(sol.t_events[0]) > 0:
        fcb_time = sol.t_events[0][0]
        print(f"fcb_time: {fcb_time}")
    else:
        print(f"Event 'reach_FCB' did not occur.")
        # You might want to assign a default value or 'None' to fcb_time here
        fcb_time = None # Or np.nan, or some other indicator

    # Rest of your code that uses fcb_time would go here
    # For example:
    if fcb_time is not None:
        print(f"Further processing with fcb_time = {fcb_time}")
    else:
        print(f"No fcb_time available for further processing.")

    endtime = fcb_time - deltaeta

    #``````````````````````````````````````````````````````````````````````````````
    #RECOMBINATION CONFORMAL TIME
    #```````````````````````````````````````````````````````````````````````````````

    #find conformal time at recombination
    s_rec = 1+z_rec  #reciprocal scale factor at recombination

    #take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
    recScaleFactorDifference = abs(sol.y[0] - s_rec) #take difference between s values and s_rec to find where s=s_rec 
    recConformalTime = sol.t[recScaleFactorDifference.argmin()]

    #print(dX2_dt(recConformalTime, inits))
    #terminate at fcb (ie when s is at its minimum positive value)
    def at_fcb(t,X):
        if X[0]<stol:
            X[0] = 0
            #print(t)
        return X[0]

    at_fcb.terminal = True

    #-------------------------------------------------------------------------------
    # For each K, find ACmatrices, BDvectors and Xmatrices
    #-------------------------------------------------------------------------------

    kvalues = np.linspace(1e-5,15,num=300);

    #first run background integration to determine s_init
    sol2 = solve_ivp(ds_dt, [t0,endtime], [s0], method='LSODA', events=at_fcb,
                    atol=atol, rtol=rtol);
    s_init = sol2.y[0,-1];

    # Determine number of processes to use
    if n_processes is None:
        n_processes = cpu_count()

    print(f"Using {n_processes} parallel processes to compute matrices for {len(kvalues)} k values")

    # Create partial function with fixed parameters
    worker_func = partial(
        process_single_k,
        s_init=s_init,
        endtime=endtime,
        swaptime=swaptime,
        recConformalTime=recConformalTime,
        H0=H0,
        OmegaLambda=OmegaLambda,
        OmegaM=OmegaM,
        OmegaR=OmegaR,
        OmegaK=OmegaK,
        Hinf=Hinf,
        s0=s0,
        deltaeta=deltaeta,
        atol=atol,
        rtol=rtol,
        num_variables=num_variables
    )

    # Use multiprocessing Pool to parallelize the computation
    with Pool(processes=n_processes) as pool:
        results = pool.map(worker_func, kvalues)

    # Unpack results
    ABCmatrices = []
    DEFmatrices = []
    GHIvectors = []
    X1matrices = []
    X2matrices = []

    for ABC_matrix, DEF_matrix, GHI_vector, X1, X2 in results:
        ABCmatrices.append(ABC_matrix)
        DEFmatrices.append(DEF_matrix)
        GHIvectors.append(GHI_vector)
        X1matrices.append(X1)
        X2matrices.append(X2)

    print("Parallel computation completed")
    np.save(folder_path+'L70_kvalues', kvalues);
    np.save(folder_path+'L70_ABCmatrices', ABCmatrices);
    np.save(folder_path+'L70_DEFmatrices', DEFmatrices);
    np.save(folder_path+'L70_GHIvectors', GHIvectors);
    np.save(folder_path+'L70_X1matrices', X1matrices);
    np.save(folder_path+'L70_X2matrices', X2matrices);
