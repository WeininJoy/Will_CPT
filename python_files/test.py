import numpy as np
import matplotlib.pyplot as plt

list_of_n = [1+1.j, 2+2.j, 3+3.j]

# np.savetxt('outfile.txt', list_of_n)
# list_of_n = np.loadtxt('outfile.txt')

array = np.array([[1+1.j, 2+2.j, 3+3.j], [4+4.j, 5+5.j, 6+6.j], [7, 8, 9]])
np.savetxt('outfile.txt', np.column_stack([array.real, array.imag]))
array_real, array_imag = np.loadtxt('outfile.txt', unpack=True, dtype=np.complex_)
array = array_real + 1j * array_imag

print(array)