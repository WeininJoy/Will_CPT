#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMB Power Spectrum Generation with K-dependent Weighting

This script uses the modified CLASS code with k-dependent weighting to generate
CMB power spectra for closed universe models. It applies the weights generated
from eigenfunction analysis to modify the primordial power spectrum.

Author: Generated for CLASS weighting analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    from classy import Class
except ImportError:
    print("Error: Could not import CLASS Python wrapper.")
    print("Make sure CLASS is compiled and installed with Python wrapper enabled.")
    print("Run 'cd class_nu_spacing_weighting/python && python setup.py install --user'")
    sys.exit(1)

def extract_cosmological_parameters():
    """
    Extract cosmological parameters from solve_real_cosmology_vr.py
    Returns a dictionary with CLASS-compatible parameter names
    """
    # Parameters from solve_real_cosmology_vr.py (lines 27-35)
    h = 0.5409
    Omega_gamma_h2 = 2.47e-5
    Neff = 3.046
    OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    OmegaM = 0.483
    OmegaK = -0.0438
    OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
    z_rec = 1089.411

    # Other cosmological parameters
    tau_reio = 0.0495  # Reionization optical depth (typical value)
    A_s = 2.0706e-9  # Scalar amplitude (typical value)
    n_s = 0.97235  # Scalar spectral index (typical value)
    k_pivot = 0.05  # Pivot scale in 1/Mpc

    # Derived parameters
    a0 = 1
    s0 = 1/a0
    H0 = np.sqrt(1 / (3 * OmegaLambda))  # in units where c=1
    K = -OmegaK * a0**2 * H0**2
    
    # Convert to CLASS parameters
    # H0 in km/s/Mpc (standard units)
    H0_kmsMpc = h * 100  # Convert from h to actual H0
    
    # Density parameters (CLASS uses Omega_cdm, Omega_b, etc.)
    omega_b = 0.02237  # Baryon density (typical value)
    omega_cdm = OmegaM * h**2 - omega_b  # Cold dark matter density
    omega_g = Omega_gamma_h2  # Photon density
    omega_ur = omega_g * Neff * (7/8) * (4/11)**(4/3) - omega_g  # Ultra-relativistic species
    
    params = {
        'H0': H0_kmsMpc,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'Omega_k': OmegaK,
        'N_ur': Neff,
        'tau_reio': tau_reio,  # Reionization optical depth (typical value)
        'A_s': A_s,  # Scalar amplitude (typical value)
        'n_s': n_s,  # Scalar spectral index (typical value)
        'k_pivot': k_pivot,  # Pivot scale in 1/Mpc

        # Output settings
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': 2500,
        
        # Extended k range for weighting compatibility
        'k_max_tau0_over_l_max': 20.0,
        
        # Weighting parameters (our modification)
        'weighting_filename': 'mode_weights.dat',
        'max_weight_index': 31,  # 32 weights (0-31)
        'K': K,
        'nu_spacing': 1,  # From the original script analysis
        'sgnK': 1,  # Positive for closed universe
    }
    
    return params

def generate_cmb_spectrum_with_weights():
    """
    Generate CMB power spectrum using the modified CLASS with k-dependent weighting
    """
    print("Generating CMB power spectrum with k-dependent weighting...")
    
    # Get cosmological parameters
    params = extract_cosmological_parameters()
    
    print("Cosmological parameters:")
    for key, value in params.items():
        if key not in ['output', 'lensing', 'weighting_filename']:
            print(f"  {key}: {value}")
    
    # Check if weight file exists
    weight_file = 'mode_weights.dat'
    if not os.path.exists(weight_file):
        print(f"Error: Weight file '{weight_file}' not found!")
        print("Make sure you have generated the weights using solve_real_cosmology_weighting.py")
        return None, None
    
    print(f"\nUsing weight file: {weight_file}")
    
    # Initialize CLASS
    cosmo = Class()
    
    try:
        # Set parameters
        cosmo.set(params)
        
        # Compute cosmology
        print("Computing cosmology with CLASS...")
        cosmo.compute()
        
        # Get CMB power spectra
        l_max = params['l_max_scalars']
        ell = np.arange(2, l_max + 1)
        
        # Get temperature and polarization power spectra
        cl_tt = np.array([cosmo.lensed_cl(l)['tt'] for l in ell])
        cl_ee = np.array([cosmo.lensed_cl(l)['ee'] for l in ell])
        cl_te = np.array([cosmo.lensed_cl(l)['te'] for l in ell])
        
        # Convert to D_l = l(l+1)C_l/(2π) in μK²
        factor = ell * (ell + 1) / (2 * np.pi)
        Dl_tt = cl_tt * factor * (2.7255e6)**2  # Convert to μK²
        Dl_ee = cl_ee * factor * (2.7255e6)**2
        Dl_te = cl_te * factor * (2.7255e6)**2
        
        print(f"Successfully computed CMB power spectra up to l = {l_max}")
        
        # Clean up
        cosmo.struct_cleanup()
        cosmo.empty()
        
        return ell, {'tt': Dl_tt, 'ee': Dl_ee, 'te': Dl_te}
        
    except Exception as e:
        print(f"Error computing CMB spectrum: {e}")
        print("This might be due to:")
        print("1. CLASS not compiled with our modifications")
        print("2. Weight file format issues")
        print("3. Parameter compatibility issues")
        
        # Clean up even on error
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except:
            pass
        
        return None, None

def generate_reference_spectrum():
    """
    Generate a reference CMB spectrum without weighting for comparison
    """
    print("\nGenerating reference CMB spectrum (without weighting)...")
    
    params = extract_cosmological_parameters()
    
    # Remove weighting parameters for reference
    params_ref = params.copy()
    del params_ref['has_weighting']
    del params_ref['weighting_filename']
    del params_ref['max_weight_index']
    del params_ref['K']
    del params_ref['nu_spacing']
    del params_ref['sgnK']
    
    cosmo_ref = Class()
    
    try:
        cosmo_ref.set(params_ref)
        cosmo_ref.compute()
        
        l_max = params_ref['l_max_scalars']
        ell = np.arange(2, l_max + 1)
        
        cl_tt_ref = np.array([cosmo_ref.lensed_cl(l)['tt'] for l in ell])
        cl_ee_ref = np.array([cosmo_ref.lensed_cl(l)['ee'] for l in ell])
        cl_te_ref = np.array([cosmo_ref.lensed_cl(l)['te'] for l in ell])
        
        factor = ell * (ell + 1) / (2 * np.pi)
        Dl_tt_ref = cl_tt_ref * factor * (2.7255e6)**2
        Dl_ee_ref = cl_ee_ref * factor * (2.7255e6)**2
        Dl_te_ref = cl_te_ref * factor * (2.7255e6)**2
        
        cosmo_ref.struct_cleanup()
        cosmo_ref.empty()
        
        return ell, {'tt': Dl_tt_ref, 'ee': Dl_ee_ref, 'te': Dl_te_ref}
        
    except Exception as e:
        print(f"Error computing reference spectrum: {e}")
        try:
            cosmo_ref.struct_cleanup()
            cosmo_ref.empty()
        except:
            pass
        return None, None

def plot_cmb_spectra(ell, spectra_weighted, spectra_reference=None):
    """
    Plot CMB power spectra with and without weighting
    """
    print("\nPlotting CMB power spectra...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CMB Power Spectra with K-dependent Weighting', fontsize=16)
    
    # Temperature power spectrum
    ax1 = axes[0, 0]
    ax1.loglog(ell, spectra_weighted['tt'], 'r-', linewidth=2, label='With Weighting')
    if spectra_reference is not None:
        ax1.loglog(ell, spectra_reference['tt'], 'b--', linewidth=2, label='Reference (no weighting)')
    ax1.set_xlabel(r'Multipole $\ell$')
    ax1.set_ylabel(r'$D_\ell^{TT}$ [$\mu$K$^2$]')
    ax1.set_title('Temperature Power Spectrum')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # E-mode polarization power spectrum
    ax2 = axes[0, 1]
    ax2.loglog(ell, spectra_weighted['ee'], 'r-', linewidth=2, label='With Weighting')
    if spectra_reference is not None:
        ax2.loglog(ell, spectra_reference['ee'], 'b--', linewidth=2, label='Reference (no weighting)')
    ax2.set_xlabel(r'Multipole $\ell$')
    ax2.set_ylabel(r'$D_\ell^{EE}$ [$\mu$K$^2$]')
    ax2.set_title('E-mode Polarization Power Spectrum')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Temperature-E cross correlation
    ax3 = axes[1, 0]
    # Handle negative values for log plot
    positive_mask = spectra_weighted['te'] > 0
    negative_mask = spectra_weighted['te'] < 0
    
    if np.any(positive_mask):
        ax3.loglog(ell[positive_mask], spectra_weighted['te'][positive_mask], 'r-', linewidth=2)
    if np.any(negative_mask):
        ax3.loglog(ell[negative_mask], -spectra_weighted['te'][negative_mask], 'r--', linewidth=2)
    
    if spectra_reference is not None:
        pos_mask_ref = spectra_reference['te'] > 0
        neg_mask_ref = spectra_reference['te'] < 0
        if np.any(pos_mask_ref):
            ax3.loglog(ell[pos_mask_ref], spectra_reference['te'][pos_mask_ref], 'b-', linewidth=2, alpha=0.7)
        if np.any(neg_mask_ref):
            ax3.loglog(ell[neg_mask_ref], -spectra_reference['te'][neg_mask_ref], 'b--', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel(r'Multipole $\ell$')
    ax3.set_ylabel(r'$|D_\ell^{TE}|$ [$\mu$K$^2$]')
    ax3.set_title('Temperature-E Cross Correlation')
    ax3.grid(True, alpha=0.3)
    
    # Ratio plot (if reference available)
    if spectra_reference is not None:
        ax4 = axes[1, 1]
        ratio_tt = spectra_weighted['tt'] / spectra_reference['tt']
        ratio_ee = spectra_weighted['ee'] / spectra_reference['ee']
        
        ax4.semilogx(ell, ratio_tt, 'r-', linewidth=2, label='TT Ratio')
        ax4.semilogx(ell, ratio_ee, 'g-', linewidth=2, label='EE Ratio')
        ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel(r'Multipole $\ell$')
        ax4.set_ylabel('Ratio (Weighted/Reference)')
        ax4.set_title('Effect of K-dependent Weighting')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        # Remove unused subplot
        fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('cmb_power_spectra_with_weighting.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cmb_power_spectra_with_weighting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """
    Main function to generate and compare CMB power spectra
    """
    print("=" * 60)
    print("CMB Power Spectrum Generation with K-dependent Weighting")
    print("=" * 60)
    
    # Generate weighted spectrum
    ell, spectra_weighted = generate_cmb_spectrum_with_weights()
    
    if ell is None:
        print("Failed to generate CMB spectrum with weighting.")
        return
    
    # Generate reference spectrum
    ell_ref, spectra_reference = generate_reference_spectrum()
    
    # Plot results
    fig = plot_cmb_spectra(ell, spectra_weighted, spectra_reference)
    
    # Save data
    print("\nSaving results...")
    np.savez('cmb_spectra_results.npz',
             ell=ell,
             Dl_tt_weighted=spectra_weighted['tt'],
             Dl_ee_weighted=spectra_weighted['ee'],
             Dl_te_weighted=spectra_weighted['te'],
             Dl_tt_reference=spectra_reference['tt'] if spectra_reference else None,
             Dl_ee_reference=spectra_reference['ee'] if spectra_reference else None,
             Dl_te_reference=spectra_reference['te'] if spectra_reference else None)
    
    print("Results saved to 'cmb_spectra_results.npz'")
    print("Plots saved as 'cmb_power_spectra_with_weighting.pdf' and '.png'")
    
    if spectra_reference is not None:
        # Print some statistics
        ratio_tt = spectra_weighted['tt'] / spectra_reference['tt']
        ratio_ee = spectra_weighted['ee'] / spectra_reference['ee']
        
        print(f"\nStatistics for the effect of k-dependent weighting:")
        print(f"TT power spectrum ratio - Mean: {np.mean(ratio_tt):.4f}, Std: {np.std(ratio_tt):.4f}")
        print(f"EE power spectrum ratio - Mean: {np.mean(ratio_ee):.4f}, Std: {np.std(ratio_ee):.4f}")
        print(f"Maximum TT ratio: {np.max(ratio_tt):.4f} at ℓ = {ell[np.argmax(ratio_tt)]}")
        print(f"Minimum TT ratio: {np.min(ratio_tt):.4f} at ℓ = {ell[np.argmin(ratio_tt)]}")
    
    print("\nCMB power spectrum generation completed successfully!")

if __name__ == "__main__":
    main()