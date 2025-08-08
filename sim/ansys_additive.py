import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from ansys.additive.core import (
    Additive,
    AdditiveMachine,
    MeltPoolColumnNames,
    SimulationError,
    SingleBeadInput,
)

# Start connection to Additive service
additive = Additive()
print(additive.about())

###################
# Define material #
###################

# Show available materials
print("Available material names: {}".format(additive.materials_list()))

# Select chosen material
material_name = "AlSi10Mg"
print('Material selected: ', material_name)
material = additive.material(material_name)
# material.powder_packing_density = 0

solidus_temp_C = material.solidus_temperature - 273.15
vaporization_temp_C = material.vaporization_temperature - 273.15

#############################
# Define machine parameters #
#############################

# Initialise machine object
machine = AdditiveMachine()

# Define chosen parameters
bead_length = 0.004 # m
machine.laser_power = 300 # W
machine.scan_speed = 0.8 # m/s
machine.heater_temperature = 20 # deg C
# machine.layer_thickness = 10.1 * 1e-6 # m     <--- Weld approximation
machine.layer_thickness = 65 * 1e-6 # m
machine.beam_diameter = 80 * 1e-6 # m

# Print machine parameters
print(machine)

# Create SingleBeadInput pbject with specified parameters
input = SingleBeadInput(
    machine=machine,
    material=material,
    bead_length=bead_length,
    output_thermal_history=True,
    thermal_history_interval=1,
)

####################
# Generate results #
####################

# Run simulation
summary = additive.simulate(input)
if isinstance(summary, SimulationError):
    raise Exception(summary.message)

# Plot output dimensions
df = summary.melt_pool.data_frame().multiply(1e6)  # convert from meters to microns
df.index *= 1e3  # convert bead length from meters to millimeters

df.to_csv(f"ansys_additive_{machine.laser_power}W_{machine.scan_speed}m_s_{machine.layer_thickness*1e6}_um_layer.csv")

df.plot(
    y=[
        MeltPoolColumnNames.LENGTH,
        MeltPoolColumnNames.WIDTH,
        MeltPoolColumnNames.DEPTH,
        MeltPoolColumnNames.REFERENCE_WIDTH,
        MeltPoolColumnNames.REFERENCE_DEPTH,
    ],
    ylabel="Melt Pool Dimensions (µm)",
    xlabel="Bead Length (mm)",
    title="Melt Pool Dimensions vs Bead Length",
)
plt.show()

# Plot thermal history
plotter_xy = pv.Plotter(notebook=False, off_screen=True)
plotter_xy.open_gif("thermal_history_xy.gif")

path = summary.melt_pool.thermal_history_output
files = [f for f in os.listdir(path) if f.endswith(".vtk")]

for i in range(len(files)):
    i = f"{i:07}"
    mesh = pv.read(os.path.join(path, f"GridFullThermal_L0000000_T{i}.vtk"))
    plotter_xy.add_mesh(mesh, scalars="Temperature_(C)", cmap="inferno")
    plotter_xy.view_xy()
    plotter_xy.write_frame()

plotter_xy.close()


# New plotter
# Create a plotter for the animation
plotter_xy = pv.Plotter(notebook=False, off_screen=True)
plotter_xy.open_gif("thermal_history_xy.gif", fps=30, loop=0, quality=100)

# Get the path and list of VTK files
path = summary.melt_pool.thermal_history_output
files = [f for f in os.listdir(path) if f.endswith(".vtk")]
files.sort()  # Ensure files are in order

# Calculate time per frame (in ms)
total_frames = len(files)
total_time_ms = (bead_length / machine.scan_speed) * 1000  # Convert to ms
time_per_frame_ms = total_time_ms / total_frames

# Loop through each file
for i, filename in enumerate(files):
    # Clear previous meshes but keep camera position
    plotter_xy.clear_actors()
    
    # Read the VTK file
    mesh = pv.read(os.path.join(path, filename))
    
    # Calculate current timestamp in ms
    current_time_ms = i * time_per_frame_ms
    
    # Add the mesh with temperature coloring
    plotter_xy.add_mesh(mesh, scalars="Temperature_(C)", cmap="inferno", show_edges=False)
    
    # Add contour lines at specified temperatures
    contour_solid = mesh.contour(isosurfaces=[solidus_temp_C], scalars="Temperature_(C)")
    plotter_xy.add_mesh(contour_solid, color="white", line_width=2, opacity=1.0)
    
    contour_vapour = mesh.contour(isosurfaces=[vaporization_temp_C], scalars="Temperature_(C)")
    plotter_xy.add_mesh(contour_vapour, color="black", line_width=2, opacity=1.0)
    
    # Add a colorbar
    plotter_xy.add_scalar_bar(title="Temperature (°C)", n_labels=5, italic=False, 
                             font_family="arial", shadow=True, fmt="%.0f", 
                             position_x=0.8, position_y=0.05)
    
    # Add timestamp text
    plotter_xy.add_text(f"Time: {current_time_ms:.2f} ms", position="upper_left", 
                       font_size=12, color="white", shadow=True)
    
    # Add contour temperature text
    plotter_xy.add_text(f"Contours: Solidus ({round(solidus_temp_C)} °C , white), Vaporization ({round(vaporization_temp_C)} °C, black)", 
                   position=(0.01, 0.94), font_size=12, color="white", shadow=True)
    
    # Set the view to XY plane
    plotter_xy.view_xy()
    
    # Add this frame to the animation
    plotter_xy.write_frame()

# Close the plotter when done
plotter_xy.close()