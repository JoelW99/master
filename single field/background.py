#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:02:29 2023

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
phidot_const = -m / np.sqrt(12 * pi * G) # Initial value of the derivative of the inflaton field with respect to cosmic time. In this model phidot is approximately constant

# Normalisation parameters of Cosmolattice for these runs
omega = 10 * m  # Normalisation of space and time
f_ini = phi_ini  # Normalisation of field value


#Slow roll solutions of phi and H with respect to cosmic time t:

# Function to calculate phi(t)
def phi(t):
    return phi_ini + phidot_const*(t/omega) 

# Function to calculate normalized H^2(t)
def H_squared_theory(phi, phidot):
    return (4 * pi * G / 3) * (phidot**2 + m**2 * phi**2) / omega**2


# Paths for the different parameters: N=32 and kIR=1, 10, 100, 400, 600, 800, 1000
params = ["kIR=1", "kIR=10", "kIR=100", "kIR=400", "kIR=600", "kIR=800", "kIR=1000"]

# Initialize two empty lists to store data for both plots
H_values = {}
H_squared_values = {}
ln_a_values = {}
phidotsq_values = {}

# Loop through all parameters and load the corresponding files
#IMPORTANT: Datafiles need to loaded with the right path!
for param in params:
    # Load average_scale_factor.txt for H and a
    scale_factor_path = f"/home/joel/cosmolattice/{param}/average_scale_factor.txt"
    data_scale = pd.read_csv(scale_factor_path, sep="\s+", header=None)
    
    H = data_scale[3][::]  # H values
    a = data_scale[1][::]  # a values
    ln_a = np.log(a)  # Calculate ln(a)
    t = data_scale[0][::]  # Time values t

    # Load average_scalar_0.txt for phidot_sq
    scalar_path = f"/home/joel/cosmolattice/{param}/average_scalar_0.txt"
    data_scalar = pd.read_csv(scalar_path, sep="\s+", header=None)
    
    phidotsq = (data_scalar[4][::])  # phidot_sq 
    
    # Filter data to keep only values up to ln(a) = 12
    mask = ln_a <= 12
    ln_a_values[param] = ln_a[mask]
    H_squared_values[param] = H[mask]**2  # Simulated H^2 values
    phidotsq_values[param] = phidotsq[mask]  # Simulated phidot^2 values

# Create a list of N values (corresponding to ln(a)) and t values
N_values = np.linspace(0, 12, 100)
t_values = np.linspace(min(t), max(t), len(N_values))

# Calculate the theoretical predictions for H^2 (divided by the factor \omega^2)
phi_theory = phi(t_values)
H_theory_squared = H_squared_theory(phi_theory, phidot_const)

# Calculate the constant value for \phidot^2 (divided by the factor (\omega f_ini)^2)
phidot_theory_squared = (phidot_const / (omega * f_ini))**2

plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12, 'lines.linewidth': 2.5})

# Plot 1: H^2 vs N_e (logarithmic y-axis) for all parameters
plt.figure(figsize=(10, 6))
for param in params:
    plt.plot(ln_a_values[param], H_squared_values[param], label=param) #Simulated H^2 values
plt.plot(N_values, H_theory_squared, '--', color='black', label=r'Theoretical $H^2/\omega_{*}^2$') #Theoretical H^2 values
plt.xlabel(r'$N_e$', fontsize=14)  # Label x-axis as N_e
plt.ylabel(r'$H^2$', fontsize=14)  # Label y-axis as H^2 
plt.title(r'Plot of program $H^2$ with N=32', fontsize=16)
plt.yscale("log")  # Logarithmic y-axis
plt.xlim(0, 12)  # Limit x-axis to ln(a) = 12
plt.legend() #Include legend
plt.grid(True, which="both", ls="--", linewidth=0.6) #Inclued grid
plt.tight_layout()
plt.savefig('HvskIR.png', dpi=300) #Save figure
plt.show() # Show figure

# Plot 2: phidot_sq vs N_e (logarithmic y-axis) for all parameters
plt.figure(figsize=(10, 6))
for param in params:
    plt.plot(ln_a_values[param], phidotsq_values[param], label=param) #Simulated phidot^2 values
# Draw the theoretical line as a horizontal line since \phidot^2 is constant
plt.axhline(y=phidot_theory_squared, color='black', linestyle='--', label=r'Theoretical $(\dot{\phi}/(\omega_{*} f_{*}))^2$')
plt.xlabel(r'$N_e$', fontsize=14)  # Label x-axis as N_e
plt.ylabel(r'$\dot{\phi}^2$', fontsize=14)  #Label y-axis as phidot^2  
plt.title(r'Plot of program $\dot{\phi}^2$ with N=32', fontsize=16)
plt.yscale("log")  # Logarithmic y-axis
plt.xlim(0, 12)  # Limit x-axis to ln(a) = 12
plt.legend() #Include legend
plt.grid(True, which="both", ls="--", linewidth=0.6) #Include grid
plt.tight_layout()
plt.savefig('phidotvskIR.png', dpi=300) #Save figure
plt.show() #Show figure






