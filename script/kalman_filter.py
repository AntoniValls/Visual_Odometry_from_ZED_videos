import numpy as np
import os
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import tilemapbase
from pyproj import Proj
import numpy as np
import os, sys
from utils import utm_to_latlon
from segmentation_utils import street_segmentation

current_dir = os.path.dirname(__file__)
mapinmeters_path = os.path.abspath(os.path.join(current_dir, '..', 'mapinmeters'))
sys.path.append(mapinmeters_path)
from mapinmeters.extentutm import ExtentUTM 

if __name__ == "__main__":

    """
    Use the VO position as the prediction and ZED position as the correction.

    In this case, the Kalman filter is mainly doing measurement fusion — it doesn’t need to model motion at all.
    """

    folder_dir = "../datasets/predicted/trajectories/00"
    
    # Load VO and ZED data
    vo_trajectory = np.loadtxt(os.path.join(folder_dir, "lightglue_ZED_ransac_maxdepth50.txt"))
    zed_trajectory = np.loadtxt(os.path.join(folder_dir, "ZED_estimation.txt"))
    
     # Use only the X and Y positions
    vo_xy = vo_trajectory[:, :2]
    zed_xy = zed_trajectory[:, :2]

    # Ensure both trajectories have the same number of frames
    n_steps = min(len(vo_xy), len(zed_xy))
    vo_xy = vo_xy[:n_steps]
    zed_xy = zed_xy[:n_steps]

    dt = 1.0  # Assuming constant frame rate

    # Initialize 2D Kalman Filter (X and Y only)
    kf = KalmanFilter(dim_x=2, dim_z=2)

    # Transition and measurement matrices (identity: no internal dynamics)
    kf.F = np.eye(2)  
    kf.H = np.eye(2)

    # Initial state
    kf.x = vo_xy[0].reshape(2, 1)

    # Covariances
    kf.P *= 1.0         # Initial state uncertainty.
    kf.Q *= 0.9        # Process noise (uncertainty in VO).
    kf.R *= 0.2        # Measurement noise (uncertainty in ZED).

    estimates = []

    for i in range(n_steps):
        # Inject the VO position as a prediction directly
        kf.x = vo_xy[i].reshape(2, 1)
        kf.P += kf.Q  # simulate prediction uncertainty
        
         # Correct using ZED measurement
        z = zed_xy[i].reshape(2, 1)
        kf.update(z)

        estimates.append(kf.x.flatten())

    estimates = np.array(estimates)

    # Plotting the results

    # Harcoded for the IRI dataset max_lat, min_lat, max_lon, min_lon
    max_lat = 41.384280
    min_lat = 41.381470
    max_lon = 2.117390
    min_lon = 2.114900
    zone_number = 31

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
    initial_point = (426069.90, 4581718.85)
    initial_point_latlon =  utm_to_latlon(initial_point[0], initial_point[1], zone_number)
    zone = f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Extract useful data
    edges, road_area, walkable_area, *_ = street_segmentation(initial_point_latlon, zone)

    # Plot the edges, roads and walkable areas
    edges.plot(ax=ax1, linewidth=1, edgecolor="dimgray", label='Graph from OSM')
    road_area.plot(ax=ax1, color="paleturquoise", alpha=0.7)
    walkable_area.plot(ax=ax1, color="lightgreen", alpha=0.7)
    
    # Plot the trajectories
    ax1.plot(vo_xy[:, 0], vo_xy[:, 1], label='VO', linestyle='--', alpha=0.6)
    ax1.plot(zed_xy[:, 0], zed_xy[:, 1], label='ZED Tracking', alpha=0.6)
    ax1.plot(estimates[:, 0], estimates[:, 1], label='Kalman Filter Output', linewidth=2)
    ax1.legend()
    ax1.set_title('2D Trajectory Fusion (VO + ZED via Kalman Filter)')
 
    plt.tight_layout()
    plt.show()