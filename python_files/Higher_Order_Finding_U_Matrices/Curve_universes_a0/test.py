import numpy as np
import matplotlib.pyplot as plt

# def adaptive_grid(f, a, b, tol=1e-3, max_depth=10, x_values=None, y_values=None):
#     """
#     Generates data points on the (x, y) plane for a given function f(x, y)
#     using an adaptive grid.

#     Args:
#         f: The function to evaluate (takes two arguments x and y).
#         a: The starting x-value.
#         b: The ending x-value.
#         tol: The tolerance for the midpoint check.
#         max_depth: The maximum recursion depth.
#         x_values: A list to store the x-values (used in recursion).
#         y_values: A list to store the y-values (used in recursion).

#     Returns:
#         A tuple of two NumPy arrays: (x_values, y_values).
#     """

#     if x_values is None:
#         x_values = []
#     if y_values is None:
#         y_values = []

#     mid = (a + b) / 2

#     if max_depth == 0 or abs(f(mid) - (f(a) + f(b)) / 2) < tol:  # Base case
#         x_values.extend([a, b])  # extend is more efficient than append for multiple elements
#         y_values.extend([f(a), f(b)]) 
#         return

#     adaptive_grid(f, a, mid, tol, max_depth - 1, x_values, y_values)
#     adaptive_grid(f, mid, b, tol, max_depth - 1, x_values, y_values)



# # Example usage (1D function for demonstration):
# def my_function(x):
#     return x**2 + np.sin(x)


# a = -5
# b = 5
# x_vals, y_vals = [], []  # Initialize empty lists

# adaptive_grid(my_function, a, b, tol=0.01, max_depth=8, x_values=x_vals, y_values=y_vals)

# x_vals = np.array(x_vals)
# y_vals = np.array(y_vals)

# # Sort the values for plotting (because recursive calls might return out of order)
# sorted_indices = np.argsort(x_vals)
# x_vals = x_vals[sorted_indices]
# y_vals = y_vals[sorted_indices]


# plt.plot(x_vals, y_vals, marker='o', linestyle='-')
# plt.title("Adaptive Grid Example")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.grid(True)
# plt.show()


# Example for a 2D function: (You'll need to adapt plotting)

def adaptive_grid(f, a, b, y, tol=1e-3, max_depth=10, x_values=None, z_values=None):
    """
    Generates data points on the (x, y) plane for a given function f(x, y)
    using an adaptive grid, for a fixed y-value.

    Args:
        f: The function to evaluate (takes two arguments x and y).
        a: The starting x-value.
        b: The ending x-value.
        y: The fixed y-value for this slice.
        tol: The tolerance for the midpoint check.
        max_depth: The maximum recursion depth.
        x_values: A list to store the x-values (used in recursion).
        z_values: A list to store the z-values (f(x, y)) (used in recursion).

    Returns:
       None. Modifies x_values and z_values directly
    """
    if x_values is None:
        x_values = []
    if z_values is None:
        z_values = []

    mid = (a + b) / 2

    if max_depth == 0 or abs(f(mid, y) - (f(a, y) + f(b, y)) / 2) < tol:
        x_values.extend([a, b])
        z_values.extend([f(a, y), f(b, y)])
        return

    adaptive_grid(f, a, mid, y, tol, max_depth - 1, x_values, z_values)
    adaptive_grid(f, mid, b, y, tol, max_depth - 1, x_values, z_values)



def my_2d_function(x, y):
    return np.sin(x*y) + (x*y/10) # Example 2D function - more interesting surface


a = -5
b = 5
y_range = np.linspace(-5, 5, 20)  # More y-values for a smoother plot
x_vals_2d, y_vals_2d, z_vals_2d = [], [], []


for y in y_range:
    x_vals, z_vals = [], [] # Initialize for each y-slice
    adaptive_grid(my_2d_function, a, b, y, tol=0.02, max_depth=8, x_values=x_vals, z_values=z_vals)

    x_vals_2d.extend(x_vals)
    z_vals_2d.extend(z_vals)
    y_vals_2d.extend([y]*len(x_vals)) # Duplicate y-value for all x in this slice



# Convert to NumPy arrays
x = np.array(x_vals_2d)
y = np.array(y_vals_2d)
z = np.array(z_vals_2d)

# Create a regularly spaced grid for plotting (interpolation is necessary)
xi = np.linspace(min(x), max(x), 500)
yi = np.linspace(min(y), max(y), 500)
xi, yi = np.meshgrid(xi, yi)

# Interpolate z-values onto the regular grid
from scipy.interpolate import griddata  # Make sure you have SciPy installed
zi = griddata((x, y), z, (xi, yi), method='linear')  # You can also try 'cubic' or 'nearest'


# Plot the results using contourf
plt.contourf(xi, yi, zi, levels=20, cmap='viridis')  # Adjust levels and cmap as needed
plt.colorbar(label='f(x, y)')
plt.title("Adaptive Grid for 2D Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()