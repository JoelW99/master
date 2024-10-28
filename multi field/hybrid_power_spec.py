#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:27:54 2024

@author: joel
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.special import lambertw
import scipy.integrate as integrate


# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5
})

# Choosing between approximated theoretical calculation for the curvature power spectrum or the numerical calculated result
mode = int(input("Mode of the power spectrum. Put in 0 for approximated result and 1 for numerical result: "))

# Model parameters
lamb = 0.07 
M = 6.327967532e15  # in GeV
m = 1.2*1.461381503e13 # in GeV
g = 1.732050808e-4 
phi_ini = 3.684919219567746e+19 # in GeV
phi_dot_ini = -3.5349813913224693e+31 # in GeV^2
C = g**2*M**2/(lamb*m**2)


# Other parameters and initial conditions
mp = 2.435635838e18 # Planck mass in GeV
pi = np.pi # Pi
eta = 2.7182818 # numerical factor needed for calculation of peak of the power spectrum
H_ini = np.sqrt((1/(3*mp**2))*(0.5*m**2*phi_ini**2+(M**4/(4*lamb)))) # Initial H in GeV
phi_dot_ini = - (2*mp*H_ini)/phi_ini # Initial phi_dot in GeV^2 
alpha = m**2/H_ini**2 # Needed for calculation of epsilon
beta = M**2/H_ini**2 # Needed for calculation of epsilon
phi_c = M/g # Critical value of phi in GeV
kc = (H_ini/(63*m))*np.exp(1) # Critical scale k of phase transition
epsilon = np.sqrt((2/3)*alpha*beta) # Sharpness of phase transition
nt_max = 1/(2*epsilon)**(2/3) # Peak position for waterfall contribution to power spectrum
k_max = epsilon*kc*np.sqrt(nt_max*np.exp(2*nt_max)) # scale of peak in power spectrum
n_f = (np.log(2*32*pi**2*epsilon**(2/3)/(6*lamb))/epsilon)**(2/3)  # Duration of phase transition
eps = 2*mp**2/phi_c**2 # Slow roll parameter


# Definition of function nt(k), needed for describing the functional behaviour of the peak
def nt(k):
    return 0.5*lambertw(2*(k/(epsilon*kc))**2).real

# Program/Lattice parameters
N = 256 # number of lattice points along edge
ntime = 120 # end time of simulation
nslice = 10 # frequency of outputs
n = int(math.sqrt(3) * N / 2)  # number of entries for power_spectrum at each time step
nstep = n * nslice # step for going through data


# create empty lists for the x and y data
k = np.zeros(n) # for scales k
y = np.zeros(n) # for inflaton power spectrum
y2 = np.zeros(n) # for waterfall field power spectrum


# Read data of repository mbig 
data = pd.read_csv(
    "/home/joel/cosmolattice/mbig/average_scale_factor.txt", sep="\s+", header=None)
Dat = pd.DataFrame(data)

data2 = pd.read_csv(
    "/home/joel/cosmolattice/mbig/spectra_scalar_0.txt", sep="\s+", header=None)
Dat2 = pd.DataFrame(data2)

data3 = pd.read_csv(
    "/home/joel/cosmolattice/mbig/average_scalar_0.txt", sep="\s+", header=None)
Dat3 = pd.DataFrame(data3)

data4 = pd.read_csv(
    "/home/joel/cosmolattice/mbig/spectra_scalar_1.txt", sep="\s+", header=None)
Dat4 = pd.DataFrame(data4)

data5 = pd.read_csv(
    "/home/joel/cosmolattice/mbig/average_scalar_1.txt", sep="\s+", header=None)
Dat5 = pd.DataFrame(data5)



# Curvature power spectrum should be plotted at the end of the simulation, when all respective modes are deep in sub-Hubble regime
time_index = 119 

# Prepare the figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

# Extract data for the given time index
H = Dat[3][time_index * nslice]
a = Dat[1][time_index * nslice]

phi1_dotsq = Dat3[4][time_index * nslice] # Squared value of time derivative of inflaton 
phi2_dotsq = Dat5[4][time_index * nslice] # Squared value of time derivative of waterfall field

factor1 = phi1_dotsq * H**2 / (phi1_dotsq + phi2_dotsq)**2 # Factor for contribution of simulated inflaton to curvature power spectrum
factor2 = phi2_dotsq * H**2 / (phi1_dotsq + phi2_dotsq)**2 # Factor for contribution of simulated waterfall field to curvature power spectrum

# Fill in the data for k and the spectrum values
for j in range(n):
    k[j] = Dat2[0][j]
    y[j] = Dat2[1][j + time_index * nstep] # read in inflaton power spectrum at given time index
    y2[j] = Dat4[1][j + time_index * nstep] # read int waterfall field power spectrum at given time index

# Compute log(k/kc) and curvature power spectrum R
logk = np.log(k / kc)
R = factor1 * y + factor2 * y2



# Approximated theoretical calculation for the curvature power spectrum
if mode==0:
    # Calculated contribution of inflaton to the curvature power spectrum
    p_phi = (1/(96*pi**2))*(1+C)**2*(1-C/2)*m**2*phi_c**4/mp**6

    # Calculated contribution of waterfall field to the curvature power spectrum
    p_wf_max = (64/9)*C**2*eps**2*n_f*epsilon**(-8/3)*eta**3*nt_max*np.exp(-(4/3)*epsilon*nt_max**(3/2))

    # curvature power spectrum
    p_r = p_phi*np.ones(len(k)) + p_wf_max*(nt(k)/nt_max)*np.exp(-(4*epsilon/3)*(nt(k)**(3/2)-nt_max**(3/2)))
    
    ax.plot(logk, p_r, "g", label="calculated power spectrum of $\mathcal{R}$")
    
    
# Functions for numerical calculation of curvature power spectrum    

# Lambert W function for n_t(q)
def n_t(q, epsilon=epsilon, kc=kc):
    z = 2 * (q / (epsilon * kc))**2
    return 0.5 * lambertw(z).real  # Use only the real part

# Dimensionless waterfall field power spectrum 
def P_delta_chi(q, kc=kc, H_ini=H_ini, epsilon=epsilon):
    if q < kc:
        return (H_ini**2 / (4 * np.pi**2 * epsilon)) * (q / kc)**2
    else:
        nt_q = n_t(q)
        return (H_ini**2 * epsilon**2 / (4 * np.pi**2)) * nt_q * np.exp(-(4/3) * epsilon * nt_q**(3/2))

# Waterfall field power spectrum
def delta_chi_sq(k, q, cos_theta, epsilon=epsilon, kc=kc, H_ini=H_ini):
    kq_mag = np.sqrt(k**2 + q**2 - 2*k*q*cos_theta)  # |\vec(k) - \vec(q)|
    
    if kq_mag < kc:
        return H_ini**2 / (2 * epsilon * kc**3)
    else:
        nt_kq = n_t(kq_mag)
        return (H_ini**2 / (2 * kq_mag * kc**2)) * np.exp(-2 * nt_kq - (4/3) * epsilon * nt_kq**(3/2))

# Angular integral over cos(theta)
def integrand_cos_theta(cos_theta, k, q):
    return delta_chi_sq(k, q, cos_theta)

def angle_integral(k, q):
    return integrate.quad(integrand_cos_theta, -1, 1, args=(k, q))[0]

# Radial integral over momentum q
def integrand_q(q, k):
    P_q = P_delta_chi(q)
    angle_int = angle_integral(k, q)
    return P_q / q * angle_int

def radial_integral(k):
    return integrate.quad(integrand_q, 0, np.inf, args=(k))[0]



# Numerical calculation of curvature power spectrum 
if mode != 0:
    # Main program for the integral for k-vector from 0.1 to 22.1
    k_values = np.linspace(0.1, 22.1, 221)
    results = []

    for k in k_values:
        result = radial_integral(k)
        results.append(result)
        print(f'Integral for k={k}: {result}')

    # Output results as a vector
    results = np.array(results)
    
    # Calculated contribution of inflaton to the curvature power spectrum
    p_phi = (1/(96*pi**2))*(1+C)**2*(1-C/2)*m**2*phi_c**4/mp**6

    # Numerical calculated contribution of the waterfall field to the curvature power spectrum
    prefactor = (C**2 * eps**2 * n_f * (16)**2 * np.pi**2) / (18 * H_ini**4 * epsilon**(14/3)) * (3/1.49)**2
    y_values = prefactor * k_values**3 * results
    
    #Plotting of the complete calculated curvature power spectrum
    ax.plot(logk, p_phi*np.ones(len(k_values)) + y_values, "r", label="corr. calculated power spectrum of $\mathcal{R}$")



# Plot the data
ax.set_xlim([-1, 3.5]) # Choose different parameters for different simulations
ax.set_ylim([1*10**(-10), 10**(-7)])
ax.plot(logk, R, "y", label="simulated power spectrum of $\mathcal{R}$") # Simulated power spectrum


# Customize the plot
ax.set_yscale("log")
ax.set_xlabel("ln [k/kc]", fontsize=14)
ax.set_ylabel("$\u0394_{\mathcal{R}}(k)$", fontsize=14)
ax.grid(which="major", color="#d3d3d3")
ax.grid(which="minor", color="white")
ax.set_title("Theory and simulation for $\Delta_{\mathcal{R}}(k)$ with $m_{new}$=1.2*m", fontsize=16)
plt.legend(loc="lower right")
plt.savefig("power_spec_mbig.png", dpi=300)
plt.show()
