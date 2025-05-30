import numpy as np
import os
from filterpy.kalman import KalmanFilter
from plot_trajectories import plot_trajectories_from_values 

if __name__ == "__main__":

    """
    Use the VO position as the prediction and ZED position as the correction.

    In this case, the Kalman filter is mainly doing measurement fusion — it doesn’t need to model motion at all.
    """

    folder_dir = "../datasets/predicted/trajectories/00"
    
    # Load VO and ZED data
    vo_trajectory = np.loadtxt(os.path.join(folder_dir, "lightglue_HitNet_magsac++_threshold1000_maxdepth50.txt"))
    zed_trajectory = np.loadtxt(os.path.join(folder_dir, "ZED_best.txt"))
    
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
    kf.Q *= 0.5        # Process noise (uncertainty in VO).
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

    # plot the results
    plot_trajectories_from_values([vo_xy, zed_xy, estimates], 
                                   seq="00",
                                   labels=["Toni's VO", "Adria's ZED", "Kalman Filter Estimation"])