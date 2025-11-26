#!/usr/bin/env python3
"""
Compare CMB power spectra for different nu_spacing values
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_cmb_from_dat import read_class_cl_file
import glob

def compare_nu_spacing():
    """
    Plot multiple CMB power spectra from different CLASS runs
    """
    plt.figure(figsize=(12, 8))
    
    # Find all available CLASS output files
    cl_files = sorted(glob.glob("test_closed_*_cl.dat"))
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    for i, filename in enumerate(cl_files):
        try:
            l, cl_tt = read_class_cl_file(filename)
            
            # Convert to D_l in μK^2
            T_CMB = 2.7255  # CMB temperature in K
            conversion_factor = (1e6 * T_CMB)**2
            D_l = cl_tt * conversion_factor
            
            # Extract run number from filename for labeling
            run_num = filename.split('_')[2].split('.')[0]
            
            plt.loglog(l, D_l, 
                      label=f"Run {run_num}", 
                      color=colors[i % len(colors)], 
                      linewidth=1.5, 
                      alpha=0.8)
            
            print(f"Plotted {filename}: l ∈ [{l.min()}, {l.max()}], D_l ∈ [{D_l.min():.1e}, {D_l.max():.1e}] μK²")
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    # Set plot properties
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$\mathcal{D}_\ell^{TT} = \ell(\ell+1)C_\ell/(2\pi) \quad [\mu\mathrm{K}^2]$', fontsize=14)
    plt.title('CMB Temperature Power Spectrum Comparison\n(Closed Universe with nu_spacing)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, ncol=2)
    plt.xlim(2, 1000)
    
    # Add reference lines for typical CMB features
    plt.axvline(x=220, color='gray', linestyle='--', alpha=0.5, label='First acoustic peak (~220)')
    plt.axvline(x=540, color='gray', linestyle='--', alpha=0.3, label='Second acoustic peak (~540)')
    
    plt.tight_layout()
    plt.savefig('CMB_comparison_nu_spacing.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('CMB_comparison_nu_spacing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved as CMB_comparison_nu_spacing.pdf")

if __name__ == "__main__":
    compare_nu_spacing()