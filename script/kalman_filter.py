import numpy as np
import os
from filterpy.kalman import KalmanFilter
from plot_trajectories import plot_trajectories_from_values 

if __name__ == "__main__":
    """
    Proper Kalman filter implementation for trajectory tracking.
    State: [x, y, vx, vy] - position and velocity in 2D
    Uses motion model for prediction and sensor measurements for correction.
    """
    seq_id = "07"
    folder_dir = f"../datasets/predicted/trajectories/{seq_id}"
    
    # Load VO and ZED data
    vo_trajectory = np.loadtxt(os.path.join(folder_dir, "lightglue_HitNet_ransac_maxdepth50.txt"))
    zed_trajectory = np.loadtxt(os.path.join(folder_dir, "ZED_VIO_estimation.txt"))
    
    # Use only the X and Y positions
    vo_xy = vo_trajectory[:, :2]
    zed_xy = zed_trajectory[:, :2]
    
    # Ensure both trajectories have the same number of frames
    n_steps = min(len(vo_xy), len(zed_xy))
    print(len(vo_xy), len(zed_xy))
    vo_xy = vo_xy[:n_steps]
    zed_xy = zed_xy[:n_steps]
    
    dt = 1/15  # Time step (15 Hz)
    
    # Initialize 4D Kalman Filter: [x, y, vx, vy]
    kf = KalmanFilter(dim_x=4, dim_z=4)
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Measurement matrix - we observe positions from both sensors
    kf.H = np.array([[1, 0, 0, 0],  # VO x position
                     [0, 1, 0, 0],  # VO y position
                     [1, 0, 0, 0],  # ZED x position
                     [0, 1, 0, 0]]) # ZED y position
    
    # Initial state: position from first VO measurement, zero velocity
    kf.x = np.array([vo_xy[0, 0], vo_xy[0, 1], 0, 0]).reshape(4, 1)
    
    # Initial state covariance
    kf.P = np.eye(4) * 1000  # High uncertainty initially
    kf.P[2:, 2:] *= 10       # Even higher uncertainty for velocities
    
    # Process noise covariance (motion model uncertainty)
    # This models uncertainty in the constant velocity assumption
    q_std = 0.5  # Adjust based on expected acceleration/jerk
    kf.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                     [0, dt**4/4, 0, dt**3/2],
                     [dt**3/2, 0, dt**2, 0],
                     [0, dt**3/2, 0, dt**2]]) * q_std**2
    
    # Measurement noise covariance
    # Adjust these based on the actual accuracy of your sensors
    vo_noise_std = 1.0    # VO measurement noise (tune this)
    zed_noise_std = 1.0   # ZED measurement noise (typically more accurate)
    
    kf.R = np.diag([vo_noise_std**2, vo_noise_std**2, 
                    zed_noise_std**2, zed_noise_std**2])
    
    estimates = []
    velocities = []
    
    for i in range(n_steps):
        # Prediction step - use motion model to predict next state
        kf.predict()
        
        # Create measurement vector [vo_x, vo_y, zed_x, zed_y]
        z = np.array([vo_xy[i, 0], vo_xy[i, 1], 
                      zed_xy[i, 0], zed_xy[i, 1]]).reshape(4, 1)
        
        # Update step - correct prediction with measurements
        kf.update(z)
        
        # Store results
        estimates.append(kf.x[:2].flatten())  # Position only
        velocities.append(kf.x[2:].flatten()) # Velocity for analysis
    
    estimates = np.array(estimates)
    velocities = np.array(velocities)
    
    # Plot the results
    plot_trajectories_from_values([vo_xy, zed_xy, estimates],
                                  seq=seq_id,
                                  labels=["VO", "ZED", "Kalman Filter Estimation"])
    
    # Optional: Print some statistics
    print(f"Final estimated velocity: {velocities[-1]} units/frame")
    print(f"Average speed: {np.mean(np.linalg.norm(velocities, axis=1)):.3f} units/frame")
    
    # Calculate tracking performance metrics
    vo_error = np.mean(np.linalg.norm(estimates - vo_xy, axis=1))
    zed_error = np.mean(np.linalg.norm(estimates - zed_xy, axis=1))
    
    print(f"Mean distance to VO trajectory: {vo_error:.3f}")
    print(f"Mean distance to ZED trajectory: {zed_error:.3f}")