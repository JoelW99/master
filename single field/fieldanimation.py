#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:24:55 2024

@author: joel
"""

#required libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import matplotlib.ticker as ticker

# Set plot style and font sizes
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12, 'lines.linewidth': 2.5})


#Parameters
N=128 #lattcie points along an edge
n=int(np.sqrt(3)*N/2) # number of different k-modes in a lattice
ntime = 2999 # number of timesteps for a simulation with t=30 and delta_t=0.01



# Create empty lists for the x and y data
k = np.zeros(n)
y = np.zeros(n)



# Load data from text files
#IMPORTANT: Datafiles need to loaded with the right path!
data2 = pd.read_csv("/home/joel/cosmolattice/N=128/spectra_scalar_0.txt", sep="\s+", header=None)
data2 = pd.DataFrame(data2)
Dat2 = data2  # loading different modes k and power spectrum of phi at different times

data = pd.read_csv("/home/joel/cosmolattice/N=128/average_scale_factor.txt", sep="\s+", header=None)
data = pd.DataFrame(data)
Dat = data # loading scalefactor a and Hubble parameter H at different times

data3 = pd.read_csv("/home/joel/cosmolattice/N=128/average_scalar_0.txt", sep="\s+", header=None)
data3 = pd.DataFrame(data3)
Dat3 = data3 # loading phi and phi_dot at different times




# Select time points for the power spectrum
selected_frames = np.linspace(0, ntime / 2, 7, dtype=int)

# Create figure for plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Colors for different time points
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_frames)))

# Plot the power spectra at selected time points
for i, frame in enumerate(selected_frames):
    t = frame  # timestep
    H = Dat[3][frame]  # H at timestep
    a = Dat[1][frame]  # scalefactor a at timestep
    H2 = H**2
    dscal2 = Dat3[4][frame] # phi_dot^2 at timestep
    
    for j in range(n):
        k[j] = Dat2[0][j] # different modes k inside the lattice
        y[j] = (H2 / dscal2) * Dat2[1][j + frame * n] # power spectrum of R at timestep for all k values

    ax.plot(k, y, label=f"$N_{{e}}$ = {round(math.log(a), 1)}", color=colors[i]) # plotting of curvature power spectrum at different times with different colors

# Axis and plot settings
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("wavenumber [k]", fontsize=14)
ax.set_ylabel("$\u0394_R (k)$", fontsize=14)
ax.grid(True)
ax.set_title("Evolution of curvature power spectrum", fontsize=16)
ax.set_xlim([0.8 * np.min(k), 1.1 * np.max(k)]) #set x-range such that all k-modes are captured
ax.set_ylim([10**(-14), 10**(-4)]) # set y-range such that all power spectra can be seen
plt.legend(loc="lower left")
plt.savefig("power-spec_evol.png", dpi=300) # save plot
plt.show()





# Create another figure for comparison between simulation and theory after all modes crossed the horizon
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(k, y, "b", label="CosmoLattice") #Use the last plot of the evolution, when all modes are super-Hubble 
ax.plot(k, 2.1e-9 * (k)**(0.9665 - 1), "g", label="analytic solution") # Theory prediction for curvature power spectrum
plt.vlines(x=4, ymin=10**(-13), ymax=10**(-4), color='m', linestyle='--', alpha=0.5)  # Lines showing the region where theory and simulation agree well
plt.vlines(x=30, ymin=10**(-13), ymax=10**(-4), color='m', linestyle='--', alpha=0.5) # Lines showing the region where theory and simulation agree well
plt.text(3.8, 10**(-7), 'k=4', color='m', ha='right', va='bottom', rotation=0, alpha=0.7) #labeling of the left line
plt.text(28.5, 10**(-7), 'k=30', color='m', ha='right', va='bottom', rotation=0, alpha=0.7) #labeling of the right line
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("wavenumber [k]", fontsize=14)
ax.set_ylabel("$\u0394_R (k)$", fontsize=14)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.6)
ax.set_title("Curvature power spectrum for kIR = 1, N=128", fontsize=16)
ax.set_xlim([0.8 * np.min(k), 1.1 * np.max(k)]) #set x-range such that all k-modes are captured
ax.set_ylim([10**(-13), 10**(-4)]) #set y-range such that the power spectra lay central
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10)) # create sub-divisions of y-axis
plt.legend(loc="lower right") #create legend
plt.savefig("last_frame_high_res.png", dpi=300) # save plot
plt.show()





# Create histogram of occupation numbers n_k
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(k[:n], Dat2[3][:n]) #plot occupation number for each mode k
ax.set_xscale("log")
plt.vlines(x=4, ymin=0, ymax=6 * 10**(4), color='m', linestyle='--', alpha=0.5) # create lines like in the plot before
plt.vlines(x=30, ymin=0, ymax=6 * 10**(4), color='m', linestyle='--', alpha=0.5) # create lines like in the plot before
plt.text(3.8, 4 * 10**(4), 'k=4', color='m', ha='right', va='bottom', rotation=0, alpha=0.7) # label lines like in the plot before
plt.text(28.5, 4 * 10**(4), 'k=30', color='m', ha='right', va='bottom', rotation=0, alpha=0.7) # label lines like in the plot before
ax.set_xlabel("Wavenumber [k]", fontsize=14)
ax.set_ylabel("$n_{k}$", fontsize=14)
ax.set_title("Histogram of occupation numbers $n_{k}$", fontsize=16)
plt.grid(True)
plt.savefig("histogram.png", dpi=300) #save plot
plt.show()



