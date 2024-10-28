#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:02:29 2023

@author: joel
"""

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve


# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12, 'lines.linewidth': 2.5})

# Model parameters
mi = 1.461381503e13 # Mass of Inflaton
lamb = 0.07 # Waterfall self interaction
g = 1.732050808e-4 # Interaction term between inflaton and waterfall field
M = 6.327967532e15 # Mass of waterfall field
mp = 2.435635838e18 #Planck mass
phi_c = M / g # Critical field value
C = g**2 * M**2 / (lamb * mi**2) # C<1 corresponds to m^2 \phi^2 dominance of the potential

# Initial conditions
phi_ini = 3.685e19 #Inflaton initial value
H_i = np.sqrt((1/(3*mp**2))*(0.5*mi**2*phi_ini**2 + (M**4/(4*lamb)))) # Initial H value


#Background of waterfall field
alpha = mi**2 / H_i**2
beta = M**2 / H_i**2
epsilon = np.sqrt((2/3) * alpha * beta) #sharpness of waterfall pahse transition (second stage)
r = 3/2 - np.sqrt(9/4 - alpha) # exponential decay factor (first stage)


#Loading data of scale factor
#IMPORTANT: Datafiles need to loaded with the right path!
data2 = pd.read_csv("/home/joel/cosmolattice/lpsi422/average_scale_factor.txt", sep="\s+", header=None)
data2 = pd.DataFrame(data2)

# Get e-fold number
N = np.log(data2[1][::])

#Loading the evolution of inflaton
#IMPORTANT: Datafiles need to loaded with the right path!
data3 = pd.read_csv("/home/joel/cosmolattice/lpsi422/average_scalar_0.txt", sep="\s+", header=None)
data3 = pd.DataFrame(data3)

phi1 = data3[1][::] 

#Loading the evolution of waterfall field
#IMPORTANT: Datafiles need to loaded with the right path!
data4 = pd.read_csv("/home/joel/cosmolattice/lpsi422/average_scalar_1.txt", sep="\s+", header=None)
data4 = pd.DataFrame(data4)

phi2 = np.sqrt(data4[3][::])






#SINGLE FIELD COMPARISON

# Function to solve equation for inflaton 
def equation(phi, mp, phi_c, C, N):
    return phi**2 - phi_c**2 * (1 - C * np.log(phi / phi_c)) + 4 * mp**2 * (N-1)

phi_initial_guess = np.full_like(N, 3e19) #Initial guess of solution


# Solve the equation
phi_solv = fsolve(equation, phi_initial_guess, args=(mp, phi_c, C, N)) / phi_ini # Normalized solution


# Plot the comparison of inflaton field evolutions
plt.figure(figsize=(10, 6))
plt.plot(N, phi1, linestyle='-', color='b', label="CosmoLattice simulation")
plt.plot(N, phi_solv, color="r", linestyle="--", label="theoretical prediction")
plt.xlabel(r'$N $', fontsize=14)
plt.ylabel(r'$\varphi$/$\varphi_{ini}$')
plt.title('Comparison between normalized inflaton field evolutions')
plt.grid(True, which="both", ls="--")
plt.legend(loc="lower left")
plt.savefig("phi_multi.png", dpi=300)
plt.show()






#MULTI FIELD COMPARISON

# Filter for N values between 0 and 2
mask = (N >= 0) & (N <= 2)

# Apply the mask to the waterfall field to extract the corresponding values
phi2_filtered = phi2[mask]

# Find the minimum and its position
min_index = np.argmin(phi2)
N_min = N[min_index]


#Position of the minimum of the waterfall field (start of the phase transition)
min_pos = N_min 

#End of exponential growth (end of phase transition)
N_end = 2.11

#Intervall of first stage
n1 = np.linspace(0, min_pos, num=50)

#Intervall of second stage
n2 = np.linspace(min_pos, N_end, num=50)

#Index of waterfall minimum for the first stage
index = np.abs(N - n2[0]).argmin()

#Evolution of waterfall field during phase transition (second stage)
phi2_filtered = phi2[index] * np.exp((2/3) * epsilon * (n2-n2[0])**(3/2)) 

# Intervall of third stage
mask = N > N_end
n3 = N[mask]

# inflaton values during third stage
phi1_res = phi1[mask]


# Plot the comparison of waterfall field evolutions
plt.figure(figsize=(10, 6))
plt.plot(N, phi2, linestyle='-', color='g', label="CosmoLattice simulation")
plt.plot(n1, phi2[index] * np.exp(-(3-r)*(n1-min_pos)/2), color="b", linestyle="--", label="first stage in theory") # first stage evolution. There is a exponential decay until min_pos
plt.plot(n2, phi2_filtered, color="r", linestyle="--", label="second stage in theory") # Exponential growth during second stage with starting point of min_pos
plt.plot(n3, np.sqrt(M**2/lamb - g**2*(phi_ini * phi1_res)**2/lamb) / phi_ini, color="y", linestyle='--', label="third stage in theory") # Third stage: waterfall field is in local minimum determined by inflaton value at this time
plt.xlabel(r'$N $', fontsize=14)
plt.ylabel(r'$\chi$/$\varphi_{ini}$')
plt.title('Comparison between normalized waterfall field evolutions')
plt.grid(True, which="both", ls="--")
plt.legend(loc="upper left")
plt.savefig("chi_multi.png", dpi=300)
plt.show()
