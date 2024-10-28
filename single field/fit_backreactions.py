#Required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# First dataset: values of kIR and the corresponding initial field values H_ini
x_values_kIR = np.array([1, 10, 100, 400, 600, 800, 1000])  # k_IR values 
#values of H_ini taken from the first entry in average_scale_factor.txt in the repositories k_IR=1, k_IR=10 etc.
y_values_kIR = np.array([0.635604155900893, 0.642612941876462, 9.46480080455345, 151.94589459015, 342.262687894599, 610.405464569445, 952.28690648199])  

# Simulation errors for the first dataset (assumed relative uncertainty for each data point)
simulation_errors_kIR = np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001])  # Tiny relative uncertainties

# Second dataset: values of N and the corresponding initial field values H_ini
x_values_N = np.array([32, 64, 128, 160, 224, 256])  # N values 
#values of H_ini taken from the first entry in average_scale_factor.txt in the repositories N=32, N=64 etc.
y_values_N = np.array([0.635604155900893, 0.635615019687291, 0.635790487352316, 0.636061373413851, 0.637363129782582, 0.638605524689164])

# Simulation errors for the second dataset (assumed uncertainties)
simulation_errors_N = np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001])  # Tiny relative uncertainties

# Function to determine the number of decimal places in a number.
# This is used to estimate the rounding error for each data point.
def decimal_places(y):
    y_str = f'{y:.16f}'.rstrip('0')  # Convert the number to a string with up to 16 decimal places, then remove trailing zeros
    if '.' in y_str:
        return len(y_str.split('.')[1])  # Return the number of digits after the decimal point
    return 0  # If there is no decimal point, return 0

# Calculate rounding errors for each y-value in the first and second datasets.
# The rounding error is estimated as 10^(-decimal_places) / sqrt(12).
# sqrt(12) comes from the assumption of a uniform distribution in rounding error.
rounding_errors_kIR = np.array([10**(-decimal_places(y)) / np.sqrt(12) for y in y_values_kIR])
rounding_errors_N = np.array([10**(-decimal_places(y)) / np.sqrt(12) for y in y_values_N])

# Combine the rounding errors and simulation errors to get the total error for each y-value.
# Total error is calculated using the standard formula for combining independent errors in quadrature:
# Total error = sqrt((rounding error)^2 + (relative error * y_value)^2)
y_errors_kIR = np.sqrt(rounding_errors_kIR**2 + (simulation_errors_kIR * y_values_kIR)**2)
y_errors_N = np.sqrt(rounding_errors_N**2 + (simulation_errors_N * y_values_N)**2)




# Define the quadratic function to be fitted to the first dataset (kIR vs. H_ini).
def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c  

# Define the cubic function to be fitted to the second dataset (N vs. H_ini).
def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d  




# Perform a quadratic curve fit for the first dataset (kIR, H_ini).
# The curve_fit function adjusts the parameters of the quadratic function (a, b, c) to best fit the data.
# The errors (y_errors_kIR) are used to weight the fit and provide uncertainty estimates for the parameters.
popt_kIR, pcov_kIR = curve_fit(quadratic_func, x_values_kIR, y_values_kIR, sigma=y_errors_kIR, absolute_sigma=True)
a_fit_kIR, b_fit_kIR, c_fit_kIR = popt_kIR  # Extract the best-fit parameters for the quadratic function
y_fit_kIR = quadratic_func(x_values_kIR, *popt_kIR)  # Calculate the fitted y-values using the best-fit parameters

# Calculate the chi-squared statistic for the first fit.
chi2_kIR = np.sum(((y_values_kIR - y_fit_kIR) / y_errors_kIR) ** 2)
dof_kIR = len(x_values_kIR) - len(popt_kIR)  # Degrees of freedom = number of data points - number of parameters
chi2_per_dof_kIR = chi2_kIR / dof_kIR  # Reduced chi-squared (chi2 divided by degrees of freedom)

# Perform a cubic curve fit for the second dataset (N, H_ini).
# The curve_fit function adjusts the parameters of the cubic function (a, b, c, d) to best fit the data.
popt_N, pcov_N = curve_fit(cubic_func, x_values_N, y_values_N, sigma=y_errors_N, absolute_sigma=True)
a_fit_N, b_fit_N, c_fit_N, d_fit_N = popt_N  # Extract the best-fit parameters for the cubic function
y_fit_N = cubic_func(x_values_N, *popt_N)  # Calculate the fitted y-values using the best-fit parameters

# Calculate the chi-squared statistic for the second fit.
chi2_N = np.sum(((y_values_N - y_fit_N) / y_errors_N) ** 2)
dof_N = len(x_values_N) - len(popt_N)  # Degrees of freedom
chi2_per_dof_N = chi2_N / dof_N  # Reduced chi-squared





# Generate smooth x-values for plotting the fitted curves.
x_fit_curve_kIR = np.linspace(min(x_values_kIR), max(x_values_kIR), 100)
y_fit_curve_kIR = quadratic_func(x_fit_curve_kIR, *popt_kIR)  # Calculate the fitted curve for kIR

x_fit_curve_N = np.linspace(min(x_values_N), max(x_values_N), 100)
y_fit_curve_N = cubic_func(x_fit_curve_N, *popt_N)  # Calculate the fitted curve for N





# Set up the plot style and parameters for aesthetics.
plt.style.use('seaborn-v0_8')  # Use the seaborn style for cleaner plots
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12, 'lines.linewidth': 2.5})

plt.figure(figsize=(12, 8))  # Create a figure with a size of 12x8 inches

# Plot the first dataset (kIR vs H_ini) with a quadratic fit.
plt.subplot(2, 1, 1)  # First subplot
plt.errorbar(x_values_kIR, y_values_kIR, yerr=y_errors_kIR, fmt='o', color='blue', ecolor='red', elinewidth=0, capsize=4, label='Data points for different $k_{IR}$')  # Data points with error bars
plt.plot(x_fit_curve_kIR, y_fit_curve_kIR, '-', color='green', label=r'Quadratic Fit: $y = %.4f k_{IR}^2 + %.4f k_{IR} + %.4f$' % (a_fit_kIR, b_fit_kIR, c_fit_kIR))  # Plot the fitted curve
plt.xlabel(r'$k_{IR}$', fontsize=14)  # Label for x-axis
plt.ylabel(r'$H_{ini}$', fontsize=14)  # Label for y-axis
plt.title(r'Comparison between data for different $k_{IR}$ with quadratic curve', fontsize=16)  # Title of the plot
plt.grid(True, which="both", ls="--", linewidth=0.6)  # Add a grid to the plot
plt.xlim(-10,1010)  # Set x-axis limits
plt.ylim(-30,1200)  # Set y-axis limits
plt.legend(fontsize=12)  # Add a legend

# Plot the second dataset (N vs H_ini) with a cubic fit.
plt.subplot(2, 1, 2)  # Second subplot
plt.errorbar(x_values_N, y_values_N * 10, yerr=y_errors_N * 10, fmt='o', color='blue', ecolor='red', elinewidth=0, capsize=4, label='Data points for different $N$')  # Data points with error bars
plt.plot(x_fit_curve_N, y_fit_curve_N * 10, '-', color='green', label=r'Cubic Fit: $y =%.6e N^3 + %.6e N^2 + %.6e N + %.6e$' % (a_fit_N * 10, b_fit_N * 10, c_fit_N * 10, d_fit_N*10))  # Plot the fitted curve (scaled by 10)
plt.xlabel(r'$N$', fontsize=14)  # Label for x-axis
plt.ylabel(r'$H_{ini} \times 10$', fontsize=14)  # Label for y-axis, scaled by a factor of 10
plt.title(r'Comparison between data for different $N$ with cubic curve', fontsize=16)  # Title of the plot
plt.xlim(30,260)  # Set x-axis limits
plt.ylim(6.355,6.395)  # Set y-axis limits (scaled)
plt.grid(True, which="both", ls="--", linewidth=0.6)  # Add a grid to the plot
plt.legend(fontsize=12)  # Add a legend

plt.tight_layout()

# Save the figure in high resolution (300 DPI)
plt.savefig('fit_plots_high_res.png', dpi=300)

# Display the plot
plt.show()





# Print the best-fit parameters and chi-squared results for the first dataset (kIR vs H_ini)
print(f'Fit Parameters (kIR): a = {a_fit_kIR:.6f}, b = {b_fit_kIR:.6f}, c = {c_fit_kIR:.6f}')
print(f'Chi^2 (kIR) = {chi2_kIR:.3f}')
print(f'Chi^2/DOF (kIR) = {chi2_per_dof_kIR:.3f}')

# Print the best-fit parameters and chi-squared results for the second dataset (N vs H_ini)
print(f'Fit Parameters (N): a = {a_fit_N:.6e}, b = {b_fit_N:.6e}, c = {c_fit_N:.6e}, d = {d_fit_N:.6e}')
print(f'Chi^2 (N) = {chi2_N:.3f}')
print(f'Chi^2/DOF (N) = {chi2_per_dof_N:.3f}')


















