import json
import numpy as np
import matplotlib.pyplot as plt
import tilemapbase
from pyproj import Proj
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os
import sys

from utils import utm_to_latlon
from segmentation_utils import street_segmentation

# Import ExtentUTM from mapinmeters
current_dir = os.path.dirname(__file__)
mapinmeters_path = os.path.abspath(os.path.join(current_dir, '..', 'mapinmeters'))
sys.path.append(mapinmeters_path)
from mapinmeters.extentutm import ExtentUTM

"""
This script performs IMU dead reckoning using linear acceleration and angular velocity data.
It loads IMU data from a file, processes it to estimate the trajectory of the device,
and plots the estimated trajectory on a map using OpenStreetMap tiles.

NOTE: Not working!
"""

def load_imu_data(file_path):
    """ Load all IMU data"""
    imu_data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("is_available", False):
                    imu_data.append(data)
            except json.JSONDecodeError:
                continue
    return imu_data

def load_pose_trajectory(file_path):
    ''' Load only the pose trajectory from a file.'''
    trajectory = []
    current_position = np.array([0.0, 0.0, 0.0])

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                delta = data.get("pose", {}).get("translation", None)
                if delta:
                    delta = np.array(delta, dtype=np.float32)
                    current_position += delta
                    trajectory.append(current_position.copy())
            except json.JSONDecodeError:
                continue

    return np.array(trajectory)

def imu_dead_reckoning(imu_data, dt=0.01):
    positions = []
    velocities = []
    orientations = []

    # Initialize state
    position = np.array([0.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.0, 0.0])
    orientation = R.from_quat([0.0, 0.0, 0.0, 1.0])  # Identity quaternion

    for data in tqdm(imu_data):
        # Extract linear acceleration and angular velocity
        linear_acc = np.array(data["linear_acceleration"])
        angular_vel = np.array(data["angular_velocity"])

        # Update orientation
        delta_angle = angular_vel * dt
        delta_rotation = R.from_rotvec(delta_angle)
        orientation = orientation * delta_rotation

        # Rotate acceleration to world frame
        acc_world = orientation.apply(linear_acc)

        # Subtract gravity (assuming gravity is along Z-axis)
        acc_world[2] -= 9.81

        # Update velocity and position
        velocity += acc_world * dt
        position += velocity * dt

        # Store the state
        positions.append(position.copy())
        velocities.append(velocity.copy())
        orientations.append(orientation.as_quat())

    return np.array(positions), np.array(velocities), np.array(orientations)

def plot_trajectory_on_map(positions, initial_utm, zone_number, map_extent):
    # Initialize tilemapbase
    tilemapbase.init(create=True)
    tiles = tilemapbase.tiles.build_OSM()

    # Define projection
    proj_utm = Proj(proj="utm", zone=zone_number, ellps="WGS84", preserve_units=False)
    extent_utm = ExtentUTM(map_extent['min_lon'], map_extent['max_lon'],
                            map_extent['min_lat'], map_extent['max_lat'],
                            zone_number, proj_utm)
    extent_utm_sq = extent_utm.to_aspect(1.0, shrink=False)
    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    tiles = tilemapbase.tiles.build_OSM()

    # Create plot
    fig, ax = plt.subplots(figsize=(22, 10))
    plotter = tilemapbase.Plotter(extent_utm_sq, tiles, width=600)
    plotter.plot(ax, tiles)

    # Load OSM street data for the area around the initial point
    initial_point_latlon =  utm_to_latlon(initial_utm[0], initial_utm[1], zone_number)
    zone = f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Extract useful data
    edges, road_area, walkable_area, *_ = street_segmentation(initial_point_latlon, zone)

    # Plot the edges, roads and walkable areas
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray", label='Graph from OSM')
    road_area.plot(ax=ax, color="paleturquoise", alpha=0.7)
    walkable_area.plot(ax=ax, color="lightgreen", alpha=0.7)

    # Convert positions to UTM coordinates
    xs = initial_utm[0] + positions[:, 0]
    ys = initial_utm[1] + positions[:, 1]

    # Plot trajectory
    ax.plot(xs, ys, c='red', label='IMU Dead Reckoning')
    ax.set_title("IMU Dead Reckoning Trajectory")
    ax.legend()
    plt.show()

def main(dead_reckoning=False):
    
    imu_file = '../datasets/BIEL/00/imu_data.txt'

    if dead_reckoning:
        imu_data = load_imu_data(imu_file)

        if not imu_data:
            print("No valid IMU data found.")
            return

        # Perform dead reckoning
        positions, velocities, orientations = imu_dead_reckoning(imu_data, dt=0.01)

    else:
        positions = load_pose_trajectory(imu_file)
        if positions.size == 0:
            print("No valid pose trajectory found.")
            return
    
    # Define initial UTM position (example: Barcelona)
    initial_utm = (426069.90, 4581718.85)  # Replace with actual initial UTM coordinates
    zone_number = 31  # UTM zone for Barcelona

    # Define map extent (example values; adjust as needed)
    map_extent = {
        'min_lat': 41.381470,
        'max_lat': 41.384280,
        'min_lon': 2.114900,
        'max_lon': 2.117390
    }

    # Plot trajectory on map
    plot_trajectory_on_map(positions, initial_utm, zone_number, map_extent)

if __name__ == "__main__":
    main()
