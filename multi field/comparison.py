#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:46:53 2024

@author: joel
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12, 'lines.linewidth': 2.5})


# Calculated variance of single field model at the end of the simulation
sigma_squared =  1.00793388e-8

# Definition of the PDF for the inflaton phi
def f_c_phi(x):
    return 1 / np.sqrt(2 * np.pi * sigma_squared) * np.exp(-x**2 / (2 * sigma_squared))

# Time step of the evaluation
time = 115

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

# Flatten combined_data for histogram calculation
combined_data_flat = combined_data.flatten()

# Range for R
r_values = np.linspace(-0.0006, 0.0005, 1000)

# Plot the data
plt.figure(figsize=(10, 6))
plt.hist(combined_data_flat, bins=500, density=True, alpha=0.7, color='blue',label="pdf of multifield data")
plt.plot(r_values,f_c_phi(r_values),color="red", label="pdf of single field")
plt.title('Comparison of model and simulated pdfs $p_{\mathcal{R}}(\mathcal{R})$')
plt.xlabel('$\mathcal{R}$')
plt.ylabel('$p_{\mathcal{R}}(\mathcal{R})$')
plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.axvline(0, color='gray', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.savefig("pdf.png", dpi=300)
plt.show()