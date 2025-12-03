'''
Harry Chapman 25/11/21
Raytracing code
I have gone through and made some comments, but its not the best commented or structured code
If there is enough interest, I will at some point refactor it so its more broadly usable, but when I have more time
'''

# imports

# numerical processing
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

# used for constructing the shapes to raytrace around
from shapely.geometry import LineString, Polygon, Point
from scipy.interpolate import CubicSpline

# used to create gaussian beam source
from scipy.stats import truncnorm

# path operators, accessing dictionaries, etc
import os

# string matching
import re

# paralleling
from tqdm import tqdm

# used for storing data combined for plotting elsewhere
import pickle

# main raytracer class
class RayTracer:
    def __init__(self, output_directory, path, file_name="", polygon=None, eta=0.25,
                 scale_factor=4.0, flip_y=True):

        # polygon is expecting a shapely polygon, there are tools for constructing this in the class, so None is fine to start
        self.polygon = polygon
        # starting absorptivity
        self.eta = eta
        #where you want files to come out to
        self.output_directory = output_directory
        #path to input data file - looking for a csv file, containing sequential points defining a surface,
        # containing X, Y data points, starting at the top right of the surface, and ending at the top left
        #it also expects a sequential number for "slice", which is just counting for frame i.e. it will process slice 1, then slice 2 etc
        self.path = path
        self.file_name = file_name
        # Coordinate transformation parameters
        # scale_factor: Multiplier to convert to microns (default: 4.0 for legacy data, 1.0 for pre-scaled HDF5 data)
        # flip_y: Whether to flip Y-axis (default: True for legacy data, False for HDF5 data)
        self.scale_factor = scale_factor
        self.flip_y = flip_y
        self.df, self.data = self.load_data(path)
        self.current_slice = self.slices[0]
        self.NoSlices = len(self.data["Slice"].unique())

    # returning unit vectors for a given vector v
    @staticmethod
    def unit_vector(v):
        return v / np.linalg.norm(v)

    # calculating the reflected direction, based on the incident direction,and the normal direction
    @staticmethod
    def reflect(incidence, normal):
        return incidence - 2 * np.dot(incidence, normal) * normal

    # load in data based off the path given, and a slice
    # Applies coordinate transformations based on scale_factor and flip_y parameters
    def load_data(self, path, slice=None):
        data = pd.read_csv(path)

        # Apply Y-flip if enabled (for legacy data)
        if self.flip_y:
            data["Y"] = max(data["Y"]) - data["Y"]

        # Apply scaling (for legacy data: 4.0, for pre-scaled HDF5 data: 1.0)
        data["Y"] *= self.scale_factor
        data["X"] *= self.scale_factor
        # numerous fixes for dodgy slices indexing
        if self.file_name == "40US":
            data = data.groupby('Slice', group_keys=False).apply(lambda g: g.iloc[:-1]).reset_index(drop=True)

        # store data
        self.data = data.copy()
        # slices are needed to process multiple surfaces, so check slice exists as a column in data
        if 'Slice' not in data.columns:
            raise ValueError(f"'Slice' column not found in: {path}")

        # check that there are slice numbers
        unique_slices = sorted(data['Slice'].unique())
        if not unique_slices:
            raise ValueError(f"No slices found in: {path}")

        # make sure slices are sequential
        slice_map = {old: new for new, old in enumerate(unique_slices, start=1)}
        data['Slice'] = data['Slice'].map(slice_map)

        # store unique slices that exist in this data
        self.slices = sorted(data['Slice'].unique())

        # if no slice is selected, default to slice 1
        if slice is None:
            slice = 1

        # filter data for current slice
        df = data[data['Slice'] == slice]

        # make sure data exists for that slice
        if df.empty:
            raise ValueError(f"No data for slice {slice} at: {path}")

        # store everything and return data
        self.df = df
        self.data = data
        self.current_slice = slice
        self.NoSlices = len(self.slices)
        return df, data

    # build surface to raytrace on, if curved is true, extrapolate the surface to be covered, as supposed to straight
    # point to point. padding is how much space to add to corners to make sure that the shape has some space on it, not
    # needed, but visually better
    def construct_polygon(self, curved=False, num_curve_points=1000,padding=200):
        # check that you have loaded data, and default to loading anything
        if self.df.empty:
            print(f"No Data currently loaded, loading slice:{slice}at: {path}")
            try:
                self.load_data(self.path)
            except:
                print(f"Failed to load {self.path}")

        # set extent of surface
        y_min = self.df["Y"].min()
        y_max = self.df["Y"].max()
        y_range = y_max - y_min

        # Construct corners of surface, with some padding to make sure it looks nicer
        bottom_right = (self.df.iloc[0]["X"] + padding, min(y_min - y_range * 0.1,0))
        top_right = (self.df.iloc[0]["X"] + padding, self.df.iloc[0]["Y"])
        bottom_left = (self.df.iloc[-1]["X"] - padding, min(y_min - y_range * 0.1,0))
        top_left = (self.df.iloc[-1]["X"] - padding, self.df.iloc[-1]["Y"])

        if curved:
            # Calculate parameter t as cumulative distance along the points
            points = self.df[['X', 'Y']].values
            deltas = np.diff(points, axis=0)
            dist = np.sqrt((deltas ** 2).sum(axis=1))
            t = np.concatenate(([0], np.cumsum(dist)))

            # Create splines for x(t) and y(t)
            cs_x = CubicSpline(t, points[:, 0])
            cs_y = CubicSpline(t, points[:, 1])

            # Sample points evenly spaced in parameter t
            t_new = np.linspace(0, t[-1], num_curve_points)
            x_smooth = cs_x(t_new)
            y_smooth = cs_y(t_new)

            top_surface = list(zip(x_smooth, y_smooth))

        else:
            top_surface = list(zip(self.df['X'], self.df['Y']))

        # Combine into closed polygon, adding corners
        polygon_points = (
                top_surface +
                [
                    top_left,
                    bottom_left,
                    bottom_right,
                    top_right
                ]
        )
        self.polygon = Polygon(polygon_points)
        return self.polygon

    # function to find intersection!
    def find_intersection(self, ray_start, ray_dir):
        # create a ray at the start point, going effectively infinitely in ray direction
        ray = LineString([ray_start, ray_start + ray_dir * 10000])

        # setup interaction with surface
        min_dist = np.inf
        closest_pt = None
        closest_normal = None

        # pickup surface coordinates
        coords = list(self.polygon.exterior.coords)

        # check surface for intersection, by checking segments along the surface
        for i in range(len(coords) - 1):
            edge = (coords[i], coords[i + 1])
            edge_line = LineString(edge)
            intersection = ray.intersection(edge_line)
            # if there is intersection, ensure that there is no funky interaction by being too close
            if not intersection.is_empty:
                pt = np.array(intersection.coords[0])
                dist = np.linalg.norm(pt - ray_start)
                if 1e-6 < dist < min_dist:  # avoid self-hit and find closest
                    min_dist = dist
                    closest_pt = pt
                    edge_vec = np.array(edge[1]) - np.array(edge[0])
                    edge_vec = self.unit_vector(edge_vec)
                    normal = np.array([-edge_vec[1], edge_vec[0]])
                    if np.dot(normal, ray_dir) > 0:
                        normal = -normal
                    closest_normal = normal
        return closest_pt, closest_normal

    # define where the laser should start, by using the dip in the centre of the melt pool as the source
    def define_laser_source(self,minimum_dip_depth=10):
        lowest = []
        # check every slice
        for slice in range(1, self.NoSlices + 1):
            slice_df = self.data[self.data["Slice"] == slice]
            if slice_df.empty:
                continue

            # check every slice for where the dip is, and either store the dip, or the median x position
            y_range = slice_df["Y"].max() - slice_df["Y"].min()
            if y_range < minimum_dip_depth:  # No clear dip in microns
                x_pos = slice_df["X"].median()
            else:  # Clear dip
                x_pos = slice_df.loc[slice_df["Y"].idxmin(), "X"]

            lowest.append(x_pos)
            self.ray_source = np.median(lowest)
        return self.ray_source

    # function to generate the starting points for the rays, either as a gaussian or as a uniform distribution
    # includes option to visualise the distribution, for bug checking
    def generate_ray_origins(self, x_start, x_end, n_rays=100, distribution='uniform', sigma=None, Visualise=False):
        ray_source = (x_start + x_end) / 2
        radius = (x_end - x_start) / 2

        if distribution == 'uniform':
            x_vals = np.linspace(x_start, x_end, n_rays)
        elif distribution == 'gaussian':
            if sigma is None:
                sigma = radius / 1.48  # approx for 1-1/e^2 containment
            a, b = (x_start - ray_source) / sigma, (x_end - ray_source) / sigma
            x_vals = truncnorm.rvs(a, b, loc=ray_source, scale=sigma, size=n_rays)
            x_vals = np.sort(x_vals)
        else:
            raise ValueError("Distribution must be 'uniform' or 'gaussian'")

        # Visualize distribution
        if Visualise:
            plt.hist(x_vals, bins=20, edgecolor='black', weights=np.ones_like(x_vals) * 100.0 / len(x_vals))
            plt.title(f"Distribution of Ray Origins ({distribution})")
            plt.xlabel("X value")
            plt.ylabel("Percentage (%)")
            plt.tight_layout()
            output_path = os.path.join(self.output_directory, f"{self.file_name}_Ray_Distribution.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()

        return x_vals

    # generate rays, and then trace them
    def trace_rays(self, n_rays=100, max_bounces=10,
                   distribution='uniform', sigma=None,
                   ray_source=None, ray_radius=None,
                   Visualise=False):
        min_x, _, max_x, max_y = self.polygon.bounds

        if ray_radius is None:
            ray_radius = max_x - min_x
        if ray_source is None:
            try:
                ray_source = self.define_laser_source()
            except:
                ray_source = (max_x + min_x) / 2

        x_start = ray_source - ray_radius
        x_end = ray_source + ray_radius
        ray_origins = self.generate_ray_origins(x_start, x_end, n_rays, distribution, sigma,Visualise=Visualise)


        # collect ray paths, energy absorbed, and the stats on bounces
        self.paths = []
        total_absorbed = 0
        self.bounce_data = []

        for ray_x in ray_origins:
            ray_origin = np.array([ray_x, max(120,max_y * 1.2)])  # start just above polygon
            ray_dir = np.array([0, -1])
            path = [ray_origin.copy()]
            point = ray_origin.copy()

            for bounce in range(max_bounces):
                hit, normal = self.find_intersection(point, ray_dir)
                if hit is None:
                    # Extend ray visibly outward if no hit
                    path.append(point + ray_dir * 1000)
                    break

                path.append(hit)
                if np.dot(normal, ray_dir) > 0:
                    normal = -normal
                ray_dir = self.reflect(ray_dir, normal)
                point = hit
            else:
                # Extend slightly if max bounces reached
                path.append(point + ray_dir * 20)

            # store details for this ray
            num_bounces = max(len(path) - 2, 0)
            absorbed_energy = 1 - (1 - self.eta) ** num_bounces
            total_absorbed += absorbed_energy
            self.paths.append(path)
            self.bounce_data.append(num_bounces)

            # check if ray has gotten trapped
            if num_bounces == max_bounces - 1 and self.polygon.contains(Point(point)):
                print("Warning: ray trapped inside polygon:", path)

        self.absorption = total_absorbed / (n_rays) * 100
        return self.paths, self.absorption, self.bounce_data

###
# Below here are just plotting functions, which plot the raytracing data in different ways
# inexplicitly, some of them are in the class, and some of them are not
# I am also baffled by why I chose this
###


    def plot_all_paths(self, save=False, show=False):
        if self.absorption is None:
            print("You must call do the raytracing before plotting")
            return
        increase_absorption = self.absorption / (self.eta * 100)
        # Plot polygon and rays
        fig, ax = plt.subplots()
        x, y = self.polygon.exterior.xy
        ax.plot(x, y, 'k-', linewidth=2)

        for path in self.paths:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], color='r', alpha=0.1)

        min_x, min_y, max_x, max_y = self.polygon.bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(-100, max(max_y * 1.2, 100))
        ax.set_aspect('equal')
        plt.title(f"Ray Tracing: Geometry leads to {increase_absorption:.2f} x absorption compared to a flat plane")
        if save:
            output_path = os.path.join(self.output_directory, f"{self.file_name}_AllPaths.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_bounce_counts(self, save=False, show=False):
        bins, percentages = self.bounce_counts()
        plt.figure(figsize=(10, 10))
        plt.bar(bins[:-1], percentages, width=0.8, edgecolor='black', align='center')
        plt.xlabel("Number of Reflections")
        plt.ylabel("Percentage of Rays (%)")
        plt.title("Bounce Distribution as Percentage")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(bins[:-1])
        if save:
            output_path = os.path.join(self.output_directory, f"{self.file_name}_bounce_counts.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def bounce_counts(self):
        max_bounce = max(self.bounce_data) if self.bounce_data else 1
        bins = np.arange(1, max_bounce + 2)
        counts, _ = np.histogram(self.bounce_data, bins=bins)
        percentages = 100 * counts / counts.sum()
        return bins, percentages

    # plot out reflections 1,2,3 etc, to show the overall paths in steps
    def plot_reflections(self,save=False, show=False):
        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        axes = axes.flatten()

        # Plot polygon shape (same on each subplot)
        x, y = self.polygon.exterior.xy

        # Max frame index (segment index)
        max_frame = 4

        for frame in range(max_frame):
            ax = axes[frame]
            ax.plot(x, y, 'k-', linewidth=2)

            for path in self.paths:
                if frame < len(path) - 1:
                    seg = np.array([path[frame], path[frame + 1]])
                    ax.plot(seg[:, 0], seg[:, 1], 'r-', alpha=0.1)

            min_x, min_y, max_x, max_y = self.polygon.bounds
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(-100, max(max_y * 1.2,100))
            ax.set_title(f"Path {frame + 1}")
            ax.set_aspect('equal')

        plt.tight_layout()
        if save:
            output_path = os.path.join(self.output_directory, f"{self.file_name}_reflections.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    # Plot all ray paths using LineCollection, colored by bounce count.
    def plot_paths_colored_by_bounce(self, cmap='viridis', linewidth=0.4, save=False, show=False, name=""):

        if not hasattr(self, 'paths') or not self.paths:
            print("No rays traced. Call trace_rays() first.")
            return

        segments = []
        bounce_values = []
        max_bounces = 5

        for path, bounces in zip(self.paths, self.bounce_data):
            for i in range(len(path) - 1):
                segments.append([path[i], path[i + 1]])
                bounce_values.append(min(bounces, max_bounces))

        bounce_values = np.array(bounce_values)

        # Create LineCollection with bounce_values as continuous color array
        lc = LineCollection(segments, cmap=cmap, linewidths=linewidth, alpha=0.8)
        lc.set_array(bounce_values)
        lc.set_clim(0, max_bounces)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.add_collection(lc)

        # Plot polygon boundary
        x, y = self.polygon.exterior.xy
        ax.plot(x, y, 'k-', linewidth=1.5)

        min_x, min_y, max_x, max_y = self.polygon.bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(-100, max(max_y * 1.2, 400))
        ax.set_aspect('equal')

        # Continuous horizontal colorbar below x-axis
        cbar = plt.colorbar(lc, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label("Number of Bounces", rotation=0, labelpad=15)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.xaxis.set_ticks_position('bottom')


        # Add overflow text (e.g., "5+")
        # Place at the right end of colorbar
        cbar_ax = cbar.ax
        cbar_ax.text(1.02, 0.5, f"{max_bounces}+", va='center', ha='left', transform=cbar_ax.transAxes)

        plt.title("Ray Paths Colored by Bounce Count")
        plt.xlabel("Length (µm)")
        plt.ylabel("Depth (µm)")
        plt.tight_layout()

        if save:
            output_path = os.path.join(self.output_directory, f"{self.file_name}_coloured_paths_{name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


def plot_slice_histogram(bounce_distribution, output_directory, file_name, show=False):
    max_bin_display = 4
    bin_labels = [str(i) for i in range(1, max_bin_display + 1)] + [f"{max_bin_display + 1}+"]

    hist_matrix = []
    for dist in bounce_distribution:
        if len(dist) <= max_bin_display:
            padded = np.pad(dist, (0, max_bin_display + 1 - len(dist)), mode='constant')
        else:
            padded = np.zeros(max_bin_display + 1)
            padded[:max_bin_display] = dist[:max_bin_display]
            padded[-1] = np.sum(dist[max_bin_display:])
        hist_matrix.append(padded)

    heatmap_data = np.array(hist_matrix).T
    df = pd.DataFrame(heatmap_data, index=bin_labels)
    df.columns = [f"Slice {i + 1}" for i in range(len(bounce_distribution))]

    plt.figure(figsize=(10, 5))
    sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Relative Frequency (%)'})
    plt.xlabel("Slice")
    plt.ylabel("Bounce Count")
    plt.title("Bounce Distributions (Limited to 4 Bounces + Overflow)")
    plt.tight_layout()
    output_path = os.path.join(output_directory, f"{file_name}_Bounce_Distribution_Over_Time_Capped.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_absorption_time(absorptions, output_directory, file_name, show=False):
    time = np.arange(len(absorptions)) * 10
    plt.plot(time, absorptions)
    plt.xlabel("Time (µs)")
    plt.ylabel("Absorption %")
    plt.title("Absorption over time")
    plt.grid(True)
    output_path = os.path.join(output_directory, f"{file_name}_Absorption.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_subplots(polygons, paths_list, labels, bounce_data_list, output_directory, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable

    max_bounces = 5
    cmap = plt.get_cmap("inferno_r")
    boundaries = np.arange(0, max_bounces + 2) - 0.5
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in enumerate(axs):
        poly = polygons[i]
        x, y = poly.exterior.xy
        ax.plot(x, y, 'k-', linewidth=1.5)

        segments = []
        bounce_values = []

        for path, bounces in zip(paths_list[i], bounce_data_list[i]):
            for j in range(len(path) - 1):
                segments.append([path[j], path[j + 1]])
                bounce_values.append(min(bounces, max_bounces))

        bounce_values = np.array(bounce_values)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.5, alpha=0.8)
        lc.set_array(bounce_values)
        ax.add_collection(lc)

        ax.set_aspect('equal')
        ax.set_title(labels[i])
        # Optionally set limits here to zoom/fix axis
        ax.autoscale()
        ax.margins(0.05)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1,
                        boundaries=boundaries, ticks=np.arange(0, max_bounces + 1))
    cbar.set_label("Number of Bounces")
    labels_cbar = [str(t) for t in range(max_bounces)] + [f"{max_bounces}+"]
    cbar.set_ticks(np.arange(0, max_bounces + 1))
    cbar.ax.set_xticklabels(labels_cbar)
    output_path = os.path.join(output_directory, "MultiBouncePlot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


#################################
# Run code #
#################################

eta = 0.175
WD = r'D:\ProcessedData\Paper Materials\01_Laser_coupling\fig_03-Laser_Coupling_RayTracing'
output_directory = os.path.join(WD, 'plots')
os.makedirs(output_directory, exist_ok=True)

files = [
    f for f in os.listdir(WD)
    if f.lower().endswith('.csv') and 'track_statistics.csv' not in f.lower()
]
#files = [files[-1]]
track_statistics = []
polygons = []
plotting_paths = []
bounce_data = []
labels = []

for file in tqdm(files, desc="Processing files"):
    try:
        file_name = file.split(".")[0]
        path = os.path.join(WD, file)
        match = re.search(r'(\d+)US', file)
        US_amplitude = int(match.group(1)) if match else None
        if file_name == "":
            show=True
        else:
            show=False

        tracer = RayTracer(output_directory=output_directory, eta=eta, file_name=file_name, path=path)

        absorptions = []
        bounce_distribution = []
        for s in tracer.slices:
            tracer.load_data(tracer.path, slice=s)
            tracer.construct_polygon(curved=True)
            paths, absorption, bounce_counts = tracer.trace_rays(
                n_rays=200, distribution='gaussian', ray_radius=200)
            if show:
                tracer.plot_paths_colored_by_bounce(save=False,show=show)
            absorptions.append(absorption)
            bins, percentages = tracer.bounce_counts()
            bounce_distribution.append(percentages)

        plot_slice_histogram(bounce_distribution, tracer.output_directory, tracer.file_name,show=show)
        plot_absorption_time(absorptions, tracer.output_directory, tracer.file_name,show=show)

        median_absorption = np.median(absorptions)
        median_slice = np.argmin(np.abs(absorptions - median_absorption)) + 1
        tracer.load_data(tracer.path, slice=median_slice)
        polygon = tracer.construct_polygon(curved=True)
        paths, absorption, bounce_counts = tracer.trace_rays(
            n_rays=1000, distribution='gaussian', ray_radius=200)
        polygons.append(polygon)
        plotting_paths.append(paths)
        bounce_data.append(bounce_counts)
        labels.append(US_amplitude)
        track_statistics.append({
            "File": file_name,
            "US Amplitude": US_amplitude,
            "Absorption": median_absorption,
            "Starting Eta": eta,
        })

        tracer.plot_paths_colored_by_bounce(cmap="inferno_r", save=True, name="Median_Slice")
        tracer.plot_all_paths(save=True,show=show)
        tracer.plot_reflections(save=True,show=show)
        tracer.plot_bounce_counts(save=True,show=show)
        print(f"Absorption for straight segments in {file_name}: {absorption:.2f} %")

    except Exception as e:
        print(f"Error processing {file}: {e}")

track_statistics_df = pd.DataFrame(track_statistics)
stats_path = os.path.join(output_directory, "track_statistics.csv")
track_statistics_df.to_csv(stats_path, index=False)
print(f"Saved statistics to: {stats_path}")

# Create ray tracing data
ray_tracing_data = [
    {
        "polygon": poly,
        "path": path,
        "bounce_data": bounces,
        "US_amplitude": amp
    }
    for poly, path, bounces, amp in zip(polygons, plotting_paths, bounce_data, labels)
]

# Save to pickle (for potentially plotting at a later point, without regenerating details)
with open(os.path.join(output_directory, "ray_tracing_data.pkl"), "wb") as f:
    pickle.dump(ray_tracing_data, f)

# Filter desired plots
desired_plots = [0, 30, 40]
filtered = [d for d in ray_tracing_data if d["US_amplitude"] in desired_plots]

# Extract filtered components
polygons = [d["polygon"] for d in filtered]
paths = [d["path"] for d in filtered]
bounce_data = [d["bounce_data"] for d in filtered]
labels = [d["US_amplitude"] for d in filtered]

# Plot
plot_subplots(polygons, paths, labels, bounce_data, output_directory, show=True)