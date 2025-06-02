import matplotlib.pyplot as plt
import tilemapbase
from pyproj import Proj
import numpy as np
import os, sys
from utils import utm_to_latlon
from segmentation_utils import street_segmentation
from glob import glob
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
mapinmeters_path = os.path.abspath(os.path.join(current_dir, '..', 'mapinmeters'))
sys.path.append(mapinmeters_path)
from mapinmeters.extentutm import ExtentUTM 

def plot_trajectories_from_folder(seq="00", filter_keywords=None):
    """
    Function that plot the saved trajectories on the IRI background.
    """

    folder_path = f"../datasets/predicted/trajectories/{seq}/"
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    # Check if the folder exists
    if filter_keywords:
        # List all .txt files in the folder
        all_txt_files = glob(os.path.join(folder_path, "*.txt"))
        
        # Keep only those that contain all keywords
        files = []
        for file_path in all_txt_files:
            filename = os.path.basename(file_path)
            if any(keyword in filename for keyword in filter_keywords):
                files.append(file_path)
        
        files = sorted(files)
    else:
        files = sorted(glob(os.path.join(folder_path, "*.txt")))
    if not files:
        print(f"No .txt trajectory files found in: {folder_path} (filter: {filter_keywords})")
        return

    # Harcoded for the IRI dataset max_lat, min_lat, max_lon, min_lon
    if seq == "00":
        max_lat = 41.384280
        min_lat = 41.381470
        max_lon = 2.117390
        min_lon = 2.114900
        zone_number = 31
        initial_point = (426069.90, 4581718.85)


    # Create only one plot
    _, ax1 = plt.subplots(figsize=(10, 10))

     # Use ExtentUTM
    proj_utm = Proj(proj="utm",zone=zone_number, ellps="WGS84",preserve_units=False)
    extent_utm = ExtentUTM(min_lon, max_lon, min_lat, max_lat, zone_number, proj_utm)
    extent_utm_sq = extent_utm.to_aspect(1.0, shrink=False) # square aspect ratio
    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    tiles = tilemapbase.tiles.build_OSM()
    plotter1 = tilemapbase.Plotter(extent_utm_sq, tiles, width=600)
    plotter1.plot(ax1, tiles)

    # Load OSM street data for the area around the initial point
    initial_point_latlon =  utm_to_latlon(initial_point[0], initial_point[1], zone_number)
    zone = f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Extract useful data
    edges, road_area, walkable_area, *_ = street_segmentation(initial_point_latlon, zone)

    # Plot the edges, roads and walkable areas
    edges.plot(ax=ax1, linewidth=1, edgecolor="dimgray", label='Graph from OSM')
    road_area.plot(ax=ax1, color="paleturquoise", alpha=0.7)
    walkable_area.plot(ax=ax1, color="lightgreen", alpha=0.7)

    for i, file_path in tqdm(enumerate(files), desc="Loading trajectories", total=len(files)):
        try:
            trajectory = np.loadtxt(file_path)
            xs, ys = trajectory[:, 0], trajectory[:, 1]
            label = os.path.splitext(os.path.basename(file_path))[0]

            # Slightly vary color if needed
            ax1.plot(xs, ys, label=label)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
    
    ax1.set_title("Estimated Trajectories")
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='small', ncol=1)
    plt.tight_layout()
    plt.savefig(f'../datasets/predicted/figures/00_all.png', dpi=300, bbox_inches='tight')
    plt.show()

    return

def plot_trajectories_from_values(array_of_trajectories, seq="00", labels=None):
    """
    Function that plot the computed trajectories on the IRI background.
    """

    if not array_of_trajectories:
        print(f"No trajectories provided.")
        return

    # Harcoded for the IRI dataset max_lat, min_lat, max_lon, min_lon
    if seq == "00":
        max_lat = 41.384280
        min_lat = 41.381470
        max_lon = 2.117390
        min_lon = 2.114900
        zone_number = 31
        initial_point = (426069.90, 4581718.85)

    # Create only one plot
    _, ax1 = plt.subplots(figsize=(10, 10))

     # Use ExtentUTM
    proj_utm = Proj(proj="utm",zone=zone_number, ellps="WGS84",preserve_units=False)
    extent_utm = ExtentUTM(min_lon, max_lon, min_lat, max_lat, zone_number, proj_utm)
    extent_utm_sq = extent_utm.to_aspect(1.0, shrink=False) # square aspect ratio
    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    tiles = tilemapbase.tiles.build_OSM()
    plotter1 = tilemapbase.Plotter(extent_utm_sq, tiles, width=600)
    plotter1.plot(ax1, tiles)

    # Load OSM street data for the area around the initial point
    initial_point_latlon =  utm_to_latlon(initial_point[0], initial_point[1], zone_number)
    zone = f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Extract useful data
    edges, road_area, walkable_area, *_ = street_segmentation(initial_point_latlon, zone)

    # Plot the edges, roads and walkable areas
    edges.plot(ax=ax1, linewidth=1, edgecolor="dimgray", label='Graph from OSM')
    road_area.plot(ax=ax1, color="paleturquoise", alpha=0.7)
    walkable_area.plot(ax=ax1, color="lightgreen", alpha=0.7)

    for i in range(len(array_of_trajectories)):
        try:
            trajectory = array_of_trajectories[i]
            xs, ys = trajectory[:, 0], trajectory[:, 1]
            label = labels[i] if labels else None

            # Slightly vary color if needed
            ax1.plot(xs, ys, label=label)
        except Exception as e:
            print(f"Could not load trajectory {i}: {e}")

    ax1.set_title("Estimated Trajectories")
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='small', ncol=1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    # plot_trajectories_from_folder("00", filter_keywords = ["ZED", "ORB", "RANSAC"])
    # plot_trajectories_from_values([traj1, traj2, traj3], seq="00")
    
    # Plot trajectories from a specific folder
    # plot_trajectories_from_folder(seq="00", filter_keywords = ["ZED-Adria", "VIO"])
    plot_trajectories_from_folder(seq="00", filter_keywords = ["lightglue_HitNet_magsac++", "VIO.txt"])