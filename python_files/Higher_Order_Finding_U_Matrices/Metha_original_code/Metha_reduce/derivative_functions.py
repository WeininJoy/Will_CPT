# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:02:25 2022

@author: MRose
"""

"""
Writing differential equations in matrix form, and comparing with original form
We have written functions for the usual formulation (i.e. in almost identical form to Deaglans eqs. 3-5),
In full matrix form (from my paper, eq 4) and the reduced matrix form, Carola's eq 18, rewritten by me.
"""

from scipy.integrate import solve_ivp
import numpy as np
import config as c
from config import H0, OmegaM, OmegaR, OmegaLambda

#integrate background equation to find s_rec and fcb_time
sol = solve_ivp(c.ds_dt, [c.t0,12], [c.s0], method='LSODA', atol=c.atol, rtol=c.rtol);
#find FCB by finding smallest absolute value in list and printing that time
s_abs = abs(sol.y[0]);
fcb_time = sol.t[s_abs.argmin()];
#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol.y[0] - c.s_rec); #take difference between s values and s_rec to find where s=s_rec 
recConformalTime = sol.t[recScaleFactorDifference.argmin()];

k=1

"""
Intergrate perfect fluid equations for specific K
"""

#perfect fluids derivative function (with phi and psi) before rec
def deaglan_sigma(t,X):
    sigma,phi,psi,dr,dm,vr,vm = X;
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)));
    
    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma));
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma));
    
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma));
    psidot = phidot;
    drdot = (4/3)*(3*phidot + (k**2)*vr);
    dmdot = 3*phidot + vm*(k**2);
    vrdot = -(psi + dr/4);
    vmdot = sigmadot*vm - psi;
    return [sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot]

#perfect fluids derivative function (with phi and psi) after rec
def deaglan_s(t,X):
    s,phi,psi,dr,dm,vr,vm = X;
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    #calculate densities of matter and radiation
    rho_m = 3*(H0**2)*OmegaM*(abs(s**3));
    rho_r = 3*(H0**2)*OmegaR*(abs(s**4));
    
    phidot = (sdot/s)*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2);
    psidot = phidot;
    drdot = (4/3)*(3*phidot + (k**2)*vr);
    dmdot = 3*phidot + vm*(k**2);
    vrdot = -(psi + dr/4);
    vmdot = (sdot/s)*vm - psi;
    return [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot]

#Write function to take matrix input, vector input, and return derivs vector
def deriv_vec(A, X):
    Xdot = A.dot(X);
    return Xdot

#write function to output matrix A for full eqns
def get_full_matrix(X):
    s,phi,psi,dr,dm,vr,vm = X;
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))));
    sigmadot = sdot/s;
    A = np.array([[sigmadot, 0, 0, 0, -2*H0**2*OmegaR*s**2, -1.5*H0**2*OmegaM*abs(s)],
         [sigmadot, 0, 0, 0, -2*H0**2*OmegaR*s**2, -1.5*H0**2*OmegaM*abs(s)],
         [4*sigmadot, 0, 0, 0, (4/3)*k**2 - 8*H0**2*OmegaR*s**2, -6*H0**2*OmegaM*abs(s)],
         [3*sigmadot, 0, 0, 0, -6*H0**2*OmegaR*s**2, k**2 - 4.5*H0**2*OmegaM*abs(s)],
         [-1, 0, -0.25, 0, 0, 0],
         [-1, 0, 0, 0, 0, sigmadot]]);
    return A

#write function to output matrix A for full eqns
def get_full_matrix_a(X):
    a,phi,psi,dr,dm,vr,vm = X;
    adot = 1*H0*np.sqrt(OmegaLambda*abs(a**4) + OmegaM*abs(a) + OmegaR);
    curly_H = adot/a;
    A = np.array([[-curly_H, 0, 0, 0, -2*H0**2*OmegaR*a**(-2), -1.5*H0**2*OmegaM*abs(1/a)],
         [-curly_H, 0, 0, 0, -2*H0**2*OmegaR*a**(-2), -1.5*H0**2*OmegaM*abs(1/a)],
         [-4*curly_H, 0, 0, 0, (4/3)*k**2 - 8*H0**2*OmegaR*a**(-2), -6*H0**2*OmegaM*abs(1/a)],
         [-3*curly_H, 0, 0, 0, -6*H0**2*OmegaR*a**(-2), k**2 - 4.5*H0**2*OmegaM*abs(1/a)],
         [-1, 0, -0.25, 0, 0, 0],
         [-1, 0, 0, 0, 0, -curly_H]]);
    return A

def full_matrix(t, X):
    s,phi,psi,dr,dm,vr,vm = X;
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))));
    A = get_full_matrix(X);
    deriv_vec = A.dot(X[1:])
    derivs = np.concatenate(([sdot], deriv_vec))
    return derivs

def full_matrix_a(t, X):
    a,phi,psi,dr,dm,vr,vm = X;
    adot = 1*H0*np.sqrt(OmegaLambda*abs(a**4) + OmegaM*abs(a) + OmegaR);
    A = get_full_matrix_a(X);
    deriv_vec = A.dot(X[1:])
    derivs = np.concatenate(([adot], deriv_vec))
    return derivs
    
#write function to output matrix A for full eqns
def get_red_matrix(X):
    s,dr,dm,vr,vm = X;
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))));
    H = - sdot/s;
    A = np.array([[(6*H0**2*OmegaR*H*s**2)/(k**2), (6*H0**2*OmegaM*H*abs(s))/(k**2), (4/3)*k**2 - 8*H0**2*OmegaR*s**2 - (24*H0**2*OmegaR*H**2*s**2)/(k**2), -6*H0**2*OmegaM*abs(s) - (18*H0**2*OmegaM*H**2*abs(s))/(k**2)],
                 [(4.5*H0**2*OmegaR*H*s**2)/(k**2), (4.5*H0**2*OmegaM*H*abs(s))/(k**2), -6*H0**2*OmegaR*s**2 - (18*H0**2*OmegaR*H**2*s**2)/(k**2), k**2 - 4.5*H0**2*OmegaM*abs(s) - (27*H0**2*OmegaM*H**2*abs(s))/(2*k**2)],
                 [(3*H0**2*OmegaR*s**2)/(2*k**2) - 1/4, (3*H0**2*OmegaM*abs(s))/(2*k**2), -(6*H0**2*OmegaR*H*s**2)/(k**2), -(4.5*H0**2*OmegaM*H*abs(s))/(k**2)],
                 [(3*H0**2*OmegaR*s**2)/(2*k**2), (3*H0**2*OmegaM*abs(s))/(2*k**2), -(6*H0**2*OmegaR*H*s**2)/(k**2), -(4.5*H0**2*OmegaM*H*abs(s))/(k**2) - H]])
    return A

#write function to output matrix A for reduced eqns, but using sigma=log(s)
def get_red_matrix_sigma(X):
    sigma,dr,dm,vr,vm = X;
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)));
    H = - sigmadot;
    A = np.array([[(6*H0**2*OmegaR*H*np.exp(2*sigma))/(k**2), (6*H0**2*OmegaM*H*np.exp(sigma))/(k**2), (4/3)*k**2 - 8*H0**2*OmegaR*np.exp(2*sigma) - (24*H0**2*OmegaR*H**2*np.exp(2*sigma))/(k**2), -6*H0**2*OmegaM*np.exp(sigma) - (18*H0**2*OmegaM*H**2*np.exp(sigma))/(k**2)],
                 [(4.5*H0**2*OmegaR*H*np.exp(2*sigma))/(k**2), (4.5*H0**2*OmegaM*H*np.exp(sigma))/(k**2), -6*H0**2*OmegaR*np.exp(2*sigma) - (18*H0**2*OmegaR*H**2*np.exp(2*sigma))/(k**2), k**2 - 4.5*H0**2*OmegaM*np.exp(sigma) - (27*H0**2*OmegaM*H**2*np.exp(sigma))/(2*k**2)],
                 [(3*H0**2*OmegaR*np.exp(2*sigma))/(2*k**2) - 1/4, (3*H0**2*OmegaM*np.exp(sigma))/(2*k**2), -(6*H0**2*OmegaR*H*np.exp(2*sigma))/(k**2), -(4.5*H0**2*OmegaM*H*np.exp(sigma))/(k**2)],
                 [(3*H0**2*OmegaR*np.exp(2*sigma))/(2*k**2), (3*H0**2*OmegaM*np.exp(sigma))/(2*k**2), -(6*H0**2*OmegaR*H*np.exp(2*sigma))/(k**2), -(4.5*H0**2*OmegaM*H*np.exp(sigma))/(k**2) - H]])
    return A

#write function to output matrix A for full eqns
def get_red_matrix_a(X):
    a,dr,dm,vr,vm = X;
    adot = 1*H0*np.sqrt(OmegaLambda*abs(a**4) + OmegaM*abs(a) + OmegaR);
    H = adot/a;
    A = np.array([[(6*H0**2*OmegaR*H*a**(-2))/(k**2), (6*H0**2*OmegaM*H*abs(1/a))/(k**2), (4/3)*k**2 - 8*H0**2*OmegaR*a**(-2) - (24*H0**2*OmegaR*H**2*a**(-2))/(k**2), -6*H0**2*OmegaM*abs(1/a) - (18*H0**2*OmegaM*H**2*abs(1/a))/(k**2)],
                 [(4.5*H0**2*OmegaR*H*a**(-2))/(k**2), (4.5*H0**2*OmegaM*H*abs(1/a))/(k**2), -6*H0**2*OmegaR*a**(-2) - (18*H0**2*OmegaR*H**2*a**(-2))/(k**2), k**2 - 4.5*H0**2*OmegaM*abs(1/a) - (27*H0**2*OmegaM*H**2*abs(1/a))/(2*k**2)],
                 [(3*H0**2*OmegaR*a**(-2))/(2*k**2) - 1/4, (3*H0**2*OmegaM*abs(1/a))/(2*k**2), -(6*H0**2*OmegaR*H*a**(-2))/(k**2), -(4.5*H0**2*OmegaM*H*abs(1/a))/(k**2)],
                 [(3*H0**2*OmegaR*a**(-2))/(2*k**2), (3*H0**2*OmegaM*abs(1/a))/(2*k**2), -(6*H0**2*OmegaR*H*a**(-2))/(k**2), -(4.5*H0**2*OmegaM*H*abs(1/a))/(k**2) - H]])
    return A

def red_matrix(t, X):
    s,dr,dm,vr,vm = X;
    sdot = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))));
    A = get_red_matrix(X);
    deriv_vec = A.dot(X[1:])
    derivs = np.concatenate(([sdot], deriv_vec))
    return derivs

def red_matrix_a(t, X):
    a,dr,dm,vr,vm = X;
    adot = 1*H0*np.sqrt(OmegaLambda*abs(a**4) + OmegaM*abs(a) + OmegaR);
    A = get_red_matrix_a(X);
    deriv_vec = A.dot(X[1:])
    derivs = np.concatenate(([adot], deriv_vec))
    return derivs

def red_matrix_sigma(t, X):
    sigma, dr, dm, vr, vm = X;
    dsigma_dt = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma))); #ds/dt is sdot*exp(logt)
    A = get_red_matrix_sigma(X);
    deriv_vec = A.dot(X[1:]);
    derivs = np.concatenate(([dsigma_dt], deriv_vec));
    print(t)
    return derivs

def red_matrix_logtime(logt, X): #t here now is logtime, not actual time
    s, dr, dm, vr, vm = X;
    ds_dt = -1*H0*np.sqrt((OmegaLambda + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))*np.exp(logt); #ds/dt is sdot*exp(logt)
    A = get_red_matrix(X)*np.exp(logt);
    deriv_vec = A.dot(X[1:]);
    derivs = np.concatenate(([ds_dt], deriv_vec));
    return derivs

def red_matrix_logtime_sigma(logt, X): #t here now is logtime, not actual time, and using sigma instead of s
    sigma, dr, dm, vr, vm = X;
    dsigma_dt = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)))*np.exp(logt); #ds/dt is sdot*exp(logt)
    A = get_red_matrix_sigma(X)*np.exp(logt);
    deriv_vec = A.dot(X[1:]);
    derivs = np.concatenate(([dsigma_dt], deriv_vec));
    return derivs

def red_matrix_solve_for_y(t, Y): #we are solving for y = s^2 x now
    #need to take the normal matrix for the reduced equations (with sigmas) and add 2sigmadot
    sigma, dr_y, dm_y, vr_y, vm_y = Y;
    sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)
                            +OmegaR*np.exp(2*sigma)));
    matrix = get_red_matrix_sigma(Y) + 2*sigmadot;
    deriv_vec = matrix.dot(Y[1:]);
    derivs = np.concatenate(([sigmadot], deriv_vec));
    return derivs

function = red_matrix;