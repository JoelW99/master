#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:48:35 2024

@author: joel
"""

# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.70711e-39  # Gravitational constant
pi = np.pi  # Value of pi

# Inflaton parameters
m = 1.42e13  # Mass of the field
phi_ini = 3.7885e19  # Initial value of the inflaton phi
phidot_const = -m / np.sqrt(12 * pi * G)  # Initial value of the derivative of the inflaton field with respect to cosmic time. In this model phidot is approximately constant

# Normalization parameters of Cosmolattice for these runs
omega = 10 * m  # Normalization of space and time
f_ini = phi_ini  # Normalization of field value

# Slow-roll solutions for phi and H with respect to cosmic time t:

# Function to calculate phi(t)
def phi(t):
    return phi_ini - (m / np.sqrt(12 * pi * G)) * (t / omega)

# \phidot is constant, calculate it once
phidot_const = -m / np.sqrt(12 * pi * G)

# Function to calculate normalized H^2(t) 
def H_squared_theory(phi, phidot):
    return (4 * pi * G / 3) * (phidot**2 + m**2 * phi**2) / omega**2

# Paths for the different parameters: kIR=1 and N=32, N=64, N=128, N=256
params = ["N=32", "N=64", "N=128", "N=256"]

# Initialize two empty lists to store data for both plots
H_values = {}
H_squared_values = {}
N_e_values_dict = {}
phidotsq_values = {}
t_max = None  # Variable for t_max at ln(a) = 12

# Loop through all parameters and load the corresponding files
#IMPORTANT: Datafiles need to loaded with the right path!
for param in params:
    # Load average_scale_factor.txt for H and t
    scale_factor_path = f"/home/joel/cosmolattice/{param}/average_scale_factor.txt"
    data_scale = pd.read_csv(scale_factor_path, sep="\s+", header=None)
    
    H = data_scale[3][::]  # H values
    a = data_scale[1][::]  # a values
    t = data_scale[0][::]  # Time values t
    ln_a = np.log(a)  # Calculate ln(a) to obtain N_e

    # Load average_scalar_0.txt for phidot_sq
    scalar_path = f"/home/joel/cosmolattice/{param}/average_scalar_0.txt"
    data_scalar = pd.read_csv(scalar_path, sep="\s+", header=None)
    
    phidotsq = (data_scalar[4][::])  # phidot_sq 
    
    # Filter the data to keep only values up to ln(a) = 12
    mask = ln_a <= 12
    N_e_values_dict[param] = ln_a[mask]
    H_squared_values[param] = H[mask]**2  # Simulated H^2 values
    phidotsq_values[param] = phidotsq[mask]  # Simulated phidot^2 values

    # Determine t_max if it hasn't been set yet
    if t_max is None:
        t_max = t[mask].max()

# Calculate the theoretical predictions for H^2 (divided by the factor \omega^2) up to t_max
t_values = np.linspace(0, t_max, 100)
phi_theory = phi(t_values)
H_theory_squared = H_squared_theory(phi_theory, phidot_const)

# Calculate the constant value for \phidot^2 (divided by the factor (\omega f_ini)^2)
phidot_theory_squared = (phidot_const / (omega * f_ini))**2

plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12, 'lines.linewidth': 2.5})

# Plot 1: H^2 vs N_e (logarithmic y-axis) for all parameters
plt.figure(figsize=(10, 6))
for param in params:
    plt.plot(N_e_values_dict[param], H_squared_values[param], label=param) #Simulated H^2 values
plt.plot(np.linspace(0, 12, len(H_theory_squared)), H_theory_squared, '--', color='black', label=r'Theoretical $H^2/\omega_{*}^2$') #Theoretical H^2 values
plt.xlabel(r'$N_e$', fontsize=14)  # Label x-axis as N_e
plt.ylabel(r'$H^2$', fontsize=14)  # Label y-axis as H^2 
plt.title(r'Plot of program $H^2$ with $k_{IR}$=1', fontsize=16)
plt.xlim(0, 12)  # Limit x-axis to ln(a) = 12
plt.ylim(3e-1, 5e-1) # Limit y-axis to the range from 3e-1 to 5e-1
plt.legend() #Include legend
plt.grid(True, which="both", ls="--", linewidth=0.6) #Include grid
plt.tight_layout()
plt.savefig('HvsN.png', dpi=300) #Save figure
plt.show() #Show figure

# Plot 2: phidot_sq vs N_e (logarithmic y-axis) for all parameters
plt.figure(figsize=(10, 6))
for param in params:
    plt.plot(N_e_values_dict[param], phidotsq_values[param], label=param) #Simulated phidot^2 values
# Draw the theoretical line as a horizontal line since \phidot^2 is constant
plt.axhline(y=phidot_theory_squared, color='black', linestyle='--', label=r'Theoretical $(\dot{\phi}/(\omega_{*} f_{*}))^2$')
plt.xlabel(r'$N_e$', fontsize=14)  # Label x-axis as N_e
plt.ylabel(r'$\dot{\phi}^2$', fontsize=14)   #Label y-axis as phidot^2 
plt.title(r'Plot of $\dot{\phi}^2$ with $k_{IR}$=1', fontsize=16)
plt.yscale("log")  # Logarithmic y-axis
plt.xlim(0, 12)  # Limit x-axis to ln(a) = 12
plt.ylim(1e-6, 1e-4) # Limit y-axis to the range from 1e-6 to 1e-4
plt.legend(loc='lower right') #Include legend
plt.grid(True, which="both", ls="--", linewidth=0.6) #Include grid
plt.tight_layout()
plt.savefig('phidotvsN.png', dpi=300) #Save figure
plt.show() #Show figure




