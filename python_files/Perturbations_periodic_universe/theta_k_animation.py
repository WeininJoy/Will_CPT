import numpy as np
from scipy.integrate import quad
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Define theta-k function

def theta(kc, k_int):
    trans_constant = np.sqrt(2*kc/3.)
    k_com = np.real(np.sqrt( k_int*(k_int+2) ))
    k = k_com * trans_constant
    def psi_curved(a, kc, k):
        return (k**2)**0.5 * ((k**2-2*kc)**2-4)**0.5 * a**2 / (1+a**4-2*kc*a**2)**0.5 / (a**4 + (k**2-2*kc)*a**2 +1)  # new Phi equation 
    try:
        theta_int  = 2* quad(psi_curved, 0, 1, args=(kc, k))[0]
    except:
        theta_int = 0
    return 2/ np.pi * theta_int


# Create a figure and axis
fig, ax = plt.subplots(figsize=(13.5,10.8)) 
k_int_list = np.linspace(0, 20, 200) 

# Set up the dynamic plot (initially blue)
kc_ini = 1.e-2
omega_lambda, omega_r = 0.7, 4.15e-5 / 0.7**2
theta_list = [theta(kc_ini, k_int) for k_int in k_int_list]
line, = ax.plot(k_int_list, theta_list, color='orange',lw=4)  # Current solution in blue
ax.set_xticks([i for i in range(int(k_int_list[-1]))])  # Set x-axis ticks
ax.set_xlim(k_int_list[0], k_int_list[-1])
ax.set_ylim(0, 45)
ax.set_xlabel(r"$k$")  # Set x-axis label
ax.set_ylabel(r"$\frac{2}{\pi}\theta(k,\eta=\eta_\infty)$")  # Set y-axis label


# Define multiple y_marker positions where vertical lines, horizontal lines, and markers will be
y_markers = [n for n in range(1,50)]  # Multiple y-marker points

# Create lists to store vertical lines, horizontal lines, and markers
vlines = []
hlines = []
points = []

# Loop through each x_marker to initialize vertical lines, horizontal lines, and markers
func_interpolate = interpolate.interp1d(theta_list, k_int_list, fill_value="extrapolate")

for y_marker in y_markers:

    # Vertical lines will go from (0,0) to (x_marker, y_marker)
    x_marker = func_interpolate(y_marker)  # Compute the y-value at the current x_marker
    vline, = ax.plot([x_marker, x_marker], [0, y_marker], color='darkorange', linestyle='--', lw=1.6)
    # Horizontal lines will go from (0, y_marker) to (x_marker, y_marker)
    hline, = ax.plot([0, x_marker], [y_marker, y_marker], color='darkorange', linestyle='--', lw=1.6)
    # Point marker at (x_marker, y_marker)
    point, = ax.plot([x_marker], [y_marker], 'o', color='darkorange', markersize=5)
    
    # Append the lines and points to their respective lists
    hlines.append(hline)
    vlines.append(vline)
    points.append(point)


# Function to update the plot for each frame of the animation
def update(kc):
    # Update the dynamic plot
    theta_list = [theta(kc, k_int) for k_int in k_int_list]  # Calculate the new solution
    func_interpolate = interpolate.interp1d(theta_list, k_int_list, fill_value="extrapolate")  # Use interpolator to construct interpolated function

    line.set_ydata(theta_list)  # Update the y data of the plot

    # Loop through each x_marker and update the corresponding lines and markers
    for i, y_marker in enumerate(y_markers):
        x_marker = func_interpolate(y_marker)  # Compute the x-value at the current y_marker
        # Update the vertical line to extend from (x_marker, 0) to (x_marker, y_marker)
        vlines[i].set_data([x_marker, x_marker], [0, y_marker])
        # Update the horizontal line to extend from (0, y_marker) to (x_marker, y_marker)
        hlines[i].set_data([0, x_marker], [y_marker, y_marker])
        # Update the point marker
        points[i].set_data([x_marker], [y_marker])
    
    # # Check if the current frequency is one of the specific ones
    # for allowed_kc in allowed_kc_list:
    #     if np.isclose(kc, allowed_kc, atol=0.008):
    #         # Add the current solution to the background with the designated color
    #         slope = kc_slope_map[allowed_kc]
    #         color = kc_color_map[allowed_kc]
    #         background_lines.append(ax.plot(k_int_list, theta_list, color=color,label=r"$\frac{2}{\pi}d\theta/dk=$ "+slope, lw=4)[0])
    #         # ax.legend(loc='upper right')  # Add a legend to the plot     
    # return [line]+ background_lines  # Return dynamic plot and background plots

    omega_kappa = - 2* kc * np.sqrt(omega_lambda* omega_r)
    ax.set_title(rf"Current curvature $\Omega_{'\kappa'}=$ {omega_kappa:.3f}", fontsize=35)  # Update title to show current curvature

    return [line] + vlines + hlines + points # Return dynamic plot


# List to hold the background lines for allowed kc
background_lines = []
# allowed_kc_list = [0.04226149,0.06525782,0.11329486, 0.23823409, 0.67165184, 0.98589979, 0.99846078]
# slope_list = [ r"$\frac{1}{5}$", r"$\frac{1}{4}$", r"$\frac{1}{3}$", r"$\frac{1}{2}$", r"$1$", r"$2$", r"$3$"]
# color_list = ['aliceblue', 'lightcyan', 'lightblue', 'lightskyblue', 'deepskyblue', 'blue', 'darkblue'] # slope = [1/5, 1/4, 1/3, 1/2, 1, 2, 3] respectively 
allowed_kc_list = [0.11329486, 0.23823409, 0.67165184, 0.98589979, 0.99846078]
slope_list = [  r"$\frac{1}{3}$", r"$\frac{1}{2}$", r"$1$", r"$2$", r"$3$"]
color_list = [ 'lightblue', 'lightskyblue', 'deepskyblue', 'blue', 'darkblue'] # slope = [1/5, 1/4, 1/3, 1/2, 1, 2, 3] respectively 
# Dictionary to map specific frequencies to their colors
kc_slope_map = dict(zip(allowed_kc_list, slope_list))
kc_color_map = dict(zip(allowed_kc_list, color_list))

for allowed_kc in allowed_kc_list:
    theta_list = [theta(allowed_kc, k_int) for k_int in k_int_list] 
    slope = kc_slope_map[allowed_kc]
    color = kc_color_map[allowed_kc]
    ax.plot(k_int_list, theta_list, color=color,label=r"$\frac{2}{\pi}d\theta/dk=$ "+slope, lw=4)[0]

ax.legend(loc='upper left')  # Add a legend to the plot

# Create the animation
kc_list = np.logspace(np.log(1.e-2),np.log(0.99999999), 200)
kc_frame = 1.-kc_list[::-1]
ani = FuncAnimation(fig, update, frames=kc_frame, interval=120, blit=True)

# # Show the animation
# plt.show()

# ani.save('theta_kint_animation.gif', writer='imagemagick')  # Save as a GIF
ani.save('theta_kint_animation.mp4', writer='ffmpeg', fps=30, dpi=100) # Save as an MP4