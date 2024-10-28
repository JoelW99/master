#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:03:23 2024

@author: joel
"""
#Import libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
import matplotlib.ticker as ticker

# Function to format colorbar in scientific notation
def scientific_notation(x, pos):
    if x == 0:
        return "0"
    else:
        return f'{x:.1e}'  # Format in scientific notation

# Load scale factor data from a CSV file
data = pd.read_csv("/home/joel/cosmolattice/model_new2/average_scale_factor.txt", sep="\s+", header=None)
data = pd.DataFrame(data)
Dat = data

# Path to the HDF5 file containing scalar field data
file_path = "/home/joel/cosmolattice/model_new2/scalar_singlet.h5"

# Define the cube size for the 128^3 grid
cube_size = 128

# Define time steps to loop over
time_steps = range(24, 26, 2)

# Function to extract a cube of data from the 3D grid
def extract_cube(data, cube_size):
    # Calculate start and end index for extracting the center of the cube
    start = (data.shape[0] - cube_size) // 2
    end = start + cube_size
    return data[start:end, start:end, start:end]

# Create output directory if it doesn't exist
output_dir = 'frames_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up the figure for 3D plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a 3D grid for the plot
x, y, z = np.meshgrid(np.arange(cube_size), np.arange(cube_size), np.arange(cube_size), indexing='ij')

# Initialize the color mapper with a colormap (seismic)
mappable = plt.cm.ScalarMappable(cmap='seismic')

# Create a colorbar for the plot
cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('$\chi$', fontsize=14)  # Label for colorbar
cbar.ax.tick_params(labelsize=12)

# Use FuncFormatter for custom scientific notation on colorbar
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))

# Function to plot each frame at different time steps
def plot_frame(i):
    ax.clear()  # Clear the previous plot
    
    # Extract number of e-folds from data
    N = np.log(Dat[1][10*i])

    # Open the HDF5 file and read the scalar field data for a specific time step
    with h5py.File(file_path, "r") as f:
        subgroup_name = f'scalarfield_1/{i}.'  # Dataset path in HDF5
        data = f[subgroup_name][:]
    
    # Extract the central cube of data
    cube = extract_cube(data, cube_size)
    cube_flat = cube.flatten()  # Flatten the cube for plotting
    
    # Get the minimum and maximum values in the cube
    cube_min, cube_max = np.min(cube_flat), np.max(cube_flat)

    # Normalize the color scale based on cube values
    norm = mcolors.Normalize(vmin=cube_min, vmax=cube_max)

    # Apply the seismic colormap to the cube data
    colors = plt.cm.seismic(norm(cube_flat))
    
    # Create a scatter plot in 3D with transparency (alpha)
    ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=colors, s=1)  # Alpha for transparency

    # Set title and labels for the axes
    ax.set_title(f'number of e-folds $N_e$: {N:.2f}', fontsize=20)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    
    # Set limits for the 3D plot
    ax.set_xlim([0, cube_size])
    ax.set_ylim([0, cube_size])
    ax.set_zlim([0, cube_size])

    # Update color map settings
    mappable.set_array(cube_flat)
    mappable.set_norm(norm)
    cbar.update_normal(mappable)

    # Save the plot as a PNG file in the output directory
    frame_filename = os.path.join(output_dir, f'frame_{i:03d}.png')
    plt.savefig(frame_filename, dpi=300)
    print(f'Saved {frame_filename}')

# Loop through each time step and generate frames
for i in time_steps:
    plot_frame(i)

# Close the figure after plotting
plt.close(fig)



