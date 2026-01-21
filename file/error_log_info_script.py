#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:16:41 2025

@author: mmallon
"""

import os
import re
import csv
import matplotlib.pyplot as plt

# === USER: Set your log file path here ===
log_path = 'F:/sim_segmented_300W_800mm_s/log.ruun'  # <-- Change this to your log file path

# Extract directory and filename
directory = os.path.dirname(log_path)
filename = os.path.basename(log_path)

# Initialize lists to store extracted data
time = []
move_particles = []
final_particles = []
energy_absorbed = []
power_laser = []
area_laser = []

# Open and read the log file
with open(log_path, 'r') as file:
    for line in file:
        if line.startswith('Move particles...'):
            match = re.search(r'Move particles\.\.\.(\d+)', line)
            if match:
                move_particles.append(int(match.group(1)))
        if line.startswith('Time = '):
            match = re.search(r'Time = ([\d\.]+)', line)
            if match:
                time.append(float(match.group(1)))
        elif line.startswith('Final number of particles...'):
            match = re.search(r'Final number of particles\.\.\.(\d+)', line)
            if match:
                final_particles.append(int(match.group(1)))
        elif line.startswith('Total energy absorbed [W]:'):
            match = re.search(r'Total energy absorbed \[W\]: ([\d\.]+)', line)
            if match:
                energy_absorbed.append(float(match.group(1)))
        elif line.startswith('Total Power in the laser :'):
            match = re.search(r'Total Power in the laser : ([\d\.]+)', line)
            if match:
                power_laser.append(float(match.group(1)))
        elif line.startswith('Total Area in the laser :'):
            match = re.search(r'Total Area in the laser : ([\deE\.\-]+)', line)
            if match:
                area_laser.append(float(match.group(1)))

# Ensure all lists have the same length by truncating to the shortest list
min_length = min(len(move_particles), len(final_particles), len(energy_absorbed), len(power_laser), len(area_laser))
move_particles = move_particles[:min_length]
final_particles = final_particles[:min_length]
energy_absorbed = energy_absorbed[:min_length]
power_laser = power_laser[:min_length]
area_laser = area_laser[:min_length]

# === Save extracted data to CSV ===
csv_filename = os.path.join(directory, 'simulation_data.csv')
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'Step', 'Time', 'MoveParticles', 'FinalParticles', 'EnergyAbsorbed_W',
        'PowerLaser', 'AreaLaser', 'LogDirectory', 'LogFileName'
    ])
    for i in range(min_length):
        writer.writerow([
            i+1,
            time[i],
            move_particles[i],
            final_particles[i],
            energy_absorbed[i],
            power_laser[i],
            area_laser[i],
            directory,
            filename
        ])

print(f"Data written to {csv_filename}")

# === Plotting the data ===
steps = list(range(1, min_length+1))
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(steps, move_particles, label='Move Particles')
plt.xlabel('Step')
plt.ylabel('Move Particles')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(steps, final_particles, label='Final Particles', color='orange')
plt.xlabel('Step')
plt.ylabel('Final Particles')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(steps, power_laser, label='Power in Laser [W]', color='red')
plt.xlabel('Step')
plt.ylabel('Power in Laser [W]')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(steps, energy_absorbed, label='Energy Absorbed [W]', color='green')
plt.xlabel('Step')
plt.ylabel('Energy Absorbed [W]')
plt.legend()

plt.suptitle(f"Simulation Data from {filename}\nDirectory: {directory}", fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plot_filename = os.path.join(directory, 'simulation_plots.png')
plt.savefig(plot_filename)
plt.show()

print(f"Plot saved to {plot_filename}")
