import numpy as np

kvalues = np.load('./OmegaK_0.01_deltaEta-7/L70_kvalues.npy')
allowedK = np.load('./OmegaK_0.01_deltaEta-7/allowedK.npy')

delta_K = []
for i in range(len(allowedK)-1):
    delta_K.append(allowedK[i+1] - allowedK[i])

print(delta_K)
eta_fcb = 6.11647
print(np.sqrt(3)* np.pi/2./eta_fcb)