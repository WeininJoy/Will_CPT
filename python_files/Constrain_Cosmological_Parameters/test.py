import numpy as np 
import matplotlib.pyplot as plt

mt_list = np.linspace(1.2,5.0, 10)
# rt, mt, kt = 1, 1.6080402010050252, 0.0617093944077409
omega_lambda, rt, kt = 0.7, 1, 0.0617093944077409
a_list = np.linspace(1.e-3, 3, 100)
def f(a0, omega_lambda, kt, mt, rt):
    return (1./omega_lambda -1)*a0**4 + 3*kt*a0**2 - mt*a0 - rt

for mt in mt_list:
    plt.plot(a_list, f(a_list, omega_lambda, kt, mt, rt), label = rf'$\tilde m$ = {mt}')
plt.legend()
plt.show()
