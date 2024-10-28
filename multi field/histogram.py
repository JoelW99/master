#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:10:26 2024

@author: joel
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Choose plotting time
time =115

# Original size of the cube
n = 256

# Size of the subsection
sub_n = 256

# Path to the HDF5 file
file_path = '/home/joel/cosmolattice/model_new/scalar_singlet.h5'

# Open the HDF5 file and load the data
with h5py.File(file_path, 'r') as f:
    # Load the datasets 'scalar_0(x)' and 'scalar_1(x)'
    f0 = f["scalarfield_0"]
    f1 = f["scalarfield_1"]
    data_0 = np.array(f0['115.']) # read out inflaton distribution at t=115
    data_1 = np.array(f1['115.']) # read out waterfall field distribution at t=115

# Check that both arrays have the correct shape
assert data_0.shape == (n, n, n), f"Dataset scalar_0 must have shape {n}x{n}x{n}."
assert data_1.shape == (n, n, n), f"Dataset scalar_1 must have shape {n}x{n}x{n}."

# Extract a subsection of size (sub_n x sub_n x sub_n) from both datasets
sub_data_0 = data_0[:sub_n, :sub_n, :sub_n] # inflaton field distribution
sub_data_1 = data_1[:sub_n, :sub_n, :sub_n] # waterfall field distribution

# Subtract the mean from each subsection to center around zero
sub_data_0 -= np.mean(sub_data_0) # inflaton fluctuation distribution
sub_data_1 -= np.mean(sub_data_1) # waterfall field fluctuation distribution



# Background values at specific time
t = 10*time # factor 10, because output of simulation has frequency t_frec=0.1 (10 times more data)

# Read additional data for scaling
data2 = pd.read_csv("/home/joel/cosmolattice/model_new/average_scale_factor.txt", sep="\s+", header=None)
data2 = pd.DataFrame(data2)
H = data2[3][t]

data3 = pd.read_csv("/home/joel/cosmolattice/model_new/average_scalar_0.txt", sep="\s+", header=None)
data3 = pd.DataFrame(data3)
phidot_1 = data3[2][t] # Squared value of time derivative of inflaton

data4 = pd.read_csv("/home/joel/cosmolattice/model_new/average_scalar_1.txt", sep="\s+", header=None)
data4 = pd.DataFrame(data4)
phidot_2 = data4[2][t] # Squared value of time derivative of waterfall field

# Add the two datasets together with the given scaling factors to get the distribution of the curvature perturbation R
combined_data = (H * phidot_1 / (phidot_1**2 + phidot_2**2)) * sub_data_0 + (H * phidot_2 / (phidot_1**2 + phidot_2**2)) * sub_data_1

# Flatten the combined data for the histogram
combined_data_flattened = combined_data.flatten()

# Create the histogram of the combined data for R
hist_values, bin_edges = np.histogram(combined_data_flattened, bins=5000)

# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot the non-nomralized histogram 
bin_width = np.diff(bin_edges)
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist_values, width=bin_width[0], alpha=0.6, label='Non-normalized Histogram')
plt.xlabel('$\mathcal{R}$')
plt.ylabel('Counts')
plt.title('Non-normalized Histogram for distribution of $\mathcal{R}$ for t=115')
plt.grid(True)
plt.legend()
plt.savefig("histogram_lin_t=115.png", dpi=300)
plt.show()




# Logarithmize the histogram values (only for non-zero values)
hist_values_nonzero = hist_values[hist_values > 0]
log_hist_values = np.log(hist_values_nonzero)
log_bin_centers = bin_centers[hist_values > 0]

# Define a quadratic function for fitting
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Define a quartic (4th degree) function for fitting
def quartic(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Error as 1/sqrt(non-logarithmized value)
errors = 1 / np.sqrt(hist_values_nonzero)

# Fit the logarithmized histogram data with a quadratic function
params_quad, cov_quad = curve_fit(quadratic, log_bin_centers, log_hist_values, sigma=errors)
errors_quad = np.sqrt(np.diag(cov_quad))

# Fit the logarithmized histogram data with a quartic function
params_quart, cov_quart = curve_fit(quartic, log_bin_centers, log_hist_values, sigma=errors)
errors_quart = np.sqrt(np.diag(cov_quart))

# Calculate Chi-squared per degree of freedom for quadratic fit
residuals_quad = log_hist_values - quadratic(log_bin_centers, *params_quad)
chi2_quad = np.sum((residuals_quad / errors)**2)
dof_quad = len(log_hist_values) - len(params_quad)
chi2_per_dof_quad = chi2_quad / dof_quad

# Calculate Chi-squared per degree of freedom for quartic fit
residuals_quart = log_hist_values - quartic(log_bin_centers, *params_quart)
chi2_quart = np.sum((residuals_quart / errors)**2)
dof_quart = len(log_hist_values) - len(params_quart)
chi2_per_dof_quart = chi2_quart / dof_quart

# Choose plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12, 'lines.linewidth': 2.5})

# Plot the logarithmized data and the fits
plt.figure(figsize=(10, 6))
plt.errorbar(log_bin_centers, log_hist_values, yerr=errors, fmt='o', label='Logarithmized Histogram Data', alpha=0.3, zorder=1)
plt.plot(log_bin_centers, quadratic(log_bin_centers, *params_quad), color='red',linewidth=3, label=f'Quadratic Fit ($\chi^2/\\text{{dof}}$={chi2_per_dof_quad:.2f})',zorder=2)
plt.plot(log_bin_centers, quartic(log_bin_centers, *params_quart), color='green',linewidth=3, label=f'Quartic Fit ($\chi^2/\\text{{dof}}$={chi2_per_dof_quart:.2f})',zorder=2)


# Customize the plot
plt.title('Logarithmized Histogram with Quadratic and Quartic Fits for t=115')
plt.xlabel('$\mathcal{R}$')
plt.ylabel('Log(Counts)')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('log_histogram_fit_lin_test.png',dpi=300)

# Show the plot
plt.show()
