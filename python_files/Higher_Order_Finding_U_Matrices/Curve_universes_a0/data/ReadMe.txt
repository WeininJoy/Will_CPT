# Setting for the code
# set s0=1 for running the code, and then transform the conformal k back to actual one by devided actual s0~0.1

OmegaLambda = 0.68
H0 = 1/np.sqrt(3*OmegaLambda) #we are working in units of Lambda=c=1
lam = 1
rt = 1
mt_list = np.linspace(300, 450, 10)
kt_list = np.linspace(1.e-4, 1, 10)
mt = mt_list[input_number//10]
kt = kt_list[input_number%10]

kvalues = np.linspace(1e-4/s0, 20/s0,num=300)