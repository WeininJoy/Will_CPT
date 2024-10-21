The setting of the data:
1. files with "_1": OmegaK = 0 # Flat universe
2. files with "_2": OmegaK = 0, and the same X1, X2 matrices as in Metha's code

# cosmological parameters
OmegaLambda = 0.679 # in Metha's code, OmegaLambda = 0.679 --> OmegaK = 0
OmegaM = 0.321 # in Metha's code, OmegaM = 0.321
OmegaR = 9.24e-5
OmegaK = 0
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1

lam = rt = 1
a0 = (OmegaLambda/OmegaR)**(1./4.)
s0 = 1/a0
mt = OmegaM / (OmegaLambda**(1./4.) * OmegaR**(3./4.))
kt = - OmegaK / np.sqrt(OmegaLambda* OmegaR) / 3

#set tolerances
atol = 1e-13
rtol = 1e-13
stol = 1e-10
num_variables = 75 # number of pert variables, 75 for original code
t0 = 1e-8 

# time 
deltaeta = fcb_time * 1.5e-6 # integrating from endtime-deltaeta to recombination time, instead of from FCB -> prevent numerical issues
endtime = fcb_time - deltaeta
swaptime = fcb_time / 3. #set time when we swap from s to sigma

# others
kvalukvalues = np.linspace(1e-4, 20/s0,num=300)
without GHI matrices