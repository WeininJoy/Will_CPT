# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.cosmology import LambdaCDM, Planck18

# # Baseline cosmology (e.g., Planck18)
# base_cosmo = Planck18

# print(base_cosmo.to_format("mapping"))

# # Parameter to vary (e.g., Obh2)
# param_name = 'Om0'
# param_values = np.linspace(0.25, 0.35, 10)  # Example range. Adjust as needed.

# z_rec_values = []

# for param_value in param_values:
#     # Create a new cosmology object with the varied parameter
#     cosmo = LambdaCDM(H0=67, Ode0=0.7, Tcmb0=2.725, Neff=3.04, m_nu=0.06, **{param_name: param_value})

#     # Calculate z_rec (using your chosen criterion - here, xe=0.5)
#     z = np.linspace(800, 1800, 500)  # Adjust z range if needed for extreme parameter values
#     xe = cosmo.recomb_ionization_frac(z)
#     z_rec = z[np.argmin(np.abs(xe - 0.5))]
#     z_rec_values.append(z_rec)

# # Plot the results
# plt.plot(param_values, z_rec_values)
# plt.xlabel(param_name)
# plt.ylabel('z_rec')
# plt.title(f'Dependence of z_rec on {param_name}')
# plt.grid(True)
# plt.show()

# from astropy.cosmology import LambdaCDM
# import numpy as np
# import astropy.units as u

# cosmo = LambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, Ode0=0.7, Tcmb0=2.725 * u.K)

# # Get the recombination table
# rec_table = cosmo.get_recombination_table(zstart=500, zend=1600, nz=500) # Adjust range/steps as needed

# # Extract redshift (z) and ionization fraction (xe)
# z = rec_table['z']
# xe = rec_table['xe']

# # Find z_rec (using your chosen criterion, e.g., xe=0.5)
# z_rec = z[np.argmin(np.abs(xe - 0.5))]
# print(f"Recombination redshift (xe=0.5): {z_rec}")


# # Example of varying multiple parameters (e.g., Obh2 and Omh2)
# param1_name = 'Obh2'
# param2_name = 'Omh2'
# param1_values = np.linspace(0.020, 0.024, 5)
# param2_values = np.linspace(0.12, 0.16, 5)

# Z_rec = np.zeros((len(param1_values), len(param2_values)))

# for i, param1_value in enumerate(param1_values):
#     for j, param2_value in enumerate(param2_values):
#         cosmo = Cosmology(H0=base_cosmo.H0, Ode0=base_cosmo.Ode0, Tcmb0=base_cosmo.Tcmb0, 
#                          Neff=base_cosmo.Neff, m_nu=base_cosmo.m_nu, **{param1_name: param1_value, param2_name: param2_value})
#         z = np.linspace(800, 1800, 500)
#         xe = cosmo.recomb_ionization_frac(z)
#         Z_rec[i, j] = z[np.argmin(np.abs(xe - 0.5))]



# # Plot the results (e.g. as a colormap)
# plt.imshow(Z_rec, extent=[param2_values.min(), param2_values.max(), param1_values.min(), param1_values.max()], origin='lower', aspect='auto')
# plt.colorbar(label='z_rec')
# plt.xlabel(param2_name)
# plt.ylabel(param1_name)
# plt.title('z_rec as a function of $Ω_b h^2$ and $Ω_m h^2$')
# plt.show()
