import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp2d, Rbf # Import the relevant functions


# Sample Data (replace with your actual data)
# Make sure x and y define a grid if using interp2d
x = np.linspace(0, 4, 5)
y = np.linspace(0, 4, 5)
X, Y = np.meshgrid(x, y)  # Create the grid
Z = np.sin(X) + np.cos(Y) # Example function

# Example scattered data for griddata and Rbf
x_scatter = np.random.rand(100)*4
y_scatter = np.random.rand(100)*4
z_scatter = np.sin(x_scatter) + np.cos(y_scatter)

# Interpolation methods (choose one or compare multiple)
methods = ['linear', 'cubic', 'rbf'] #'nearest' for griddata

# Create finer grid for plotting and evaluation
xi = np.linspace(0, 4, 50)
yi = np.linspace(0, 4, 50)
xi, yi = np.meshgrid(xi, yi)

fig, axes = plt.subplots(1, len(methods) + 1, figsize=(15, 5))

# Plot the original data
axes[0].contourf(X, Y, Z, 20, cmap='viridis') # or contour for lines
axes[0].scatter(X, Y, c=Z, marker='o', edgecolor='k', label="Original Data") # Show original data points
axes[0].set_title("Original Data")
axes[0].legend()




for i, method in enumerate(methods):
    if method == 'rbf':
        interp_func = Rbf(x_scatter, y_scatter, z_scatter)  # Example RBF interpolation
        zi = interp_func(xi, yi)
    elif method in ['nearest', 'linear', 'cubic']:
        # griddata example:
        zi = griddata((x_scatter, y_scatter), z_scatter, (xi, yi), method=method)
    else:  # interp2d (ensure data is on a grid for interp2d)
        interp_func = interp2d(x, y, Z, kind=method)
        zi = interp_func(xi, yi)

    axes[i+1].contourf(xi, yi, zi, 20, cmap='viridis')

    if method == 'rbf' or method in ['nearest', 'linear', 'cubic']:
        axes[i+1].scatter(x_scatter, y_scatter, c=z_scatter, marker='o', edgecolor='k', label="Original Data") # Show original data points
    else:
        axes[i+1].scatter(X, Y, c=Z, marker='o', edgecolor='k', label="Original Data")

    axes[i+1].set_title(f"Interpolation: {method.capitalize()}")
    axes[i+1].legend()

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((zi - np.sin(xi) - np.cos(yi))**2))  # Replace with your true function if known
    print(f"RMSE for {method}: {rmse}")



plt.tight_layout()
plt.show()


