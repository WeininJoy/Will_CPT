The setting of the data:
1. files without "_0": OmegaLambda = 0.68
2. files with "_0": OmegaLambda = 0.72

# cosmological parameters
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1
lam = 1
rt = 1
mt_list = np.linspace(300, 450, 10)
kt_list = np.linspace(1.e-4, 1, 10)
mt = mt_list[input_number//10]
kt = kt_list[input_number%10]

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
1. files without "_0": kvalukvalues = np.linspace(1e-4, 15/s0,num=300)
2. files with "_0": kvalukvalues = np.linspace(1e-4, 20/s0,num=300)
without GHI matrices