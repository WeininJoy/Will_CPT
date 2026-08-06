import numpy as np

params = [0.45354621043311405, -0.03973432289727319, 0.15567561903087945, 0.5645841034764626]
allowedK_integer = [335.45757105748567,339.456804620278,343.4566035981207,347.4559445231542,351.4559541750532,355.4555658545569,359.45580647031466,363.4555897987866,367.455795548429,371.45563757393836,375.45599586252877,379.4559157747516,383.45620230915824,387.45627348934454,391.45639260453754,395.45634130229183,399.4565407102397,403.45643242014233,407.456542016495,411.4562276152635,415.45618980599573,419.4558135584933,423.45551134194915,427.4548531905136,431.45455148805826]


# Unpack parameters
Omega_gamma_h2 = 2.47e-5  # photon density
Neff = 3.046
OmegaM, OmegaK, omega_b_ratio, h = params
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
OmegaLambda = 1 - OmegaM - OmegaK - OmegaR

H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
a0=1; K=-OmegaK * a0**2 * H0**2
allowedK = [k * np.sqrt(np.abs(K)) for k in allowedK_integer]

folder_path = './data_cosmo_param_direc/data_all_k50-65/'
np.save(folder_path+'allowedK.npy', allowedK)
np.save(folder_path+'allowedK_integer.npy', allowedK_integer)