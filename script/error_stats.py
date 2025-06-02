
import numpy as np
import matplotlib.pyplot as plt
import os, json
from plot_trajectories import plot_trajectories_from_folder

"""
Script to obtain the statistics of ZED VIO's estimation based on the hardcoded groundtruths

Steps:

    1. Load both files (GT and predicted trajectory).

    2. Interpolate GT linearly to match timestamps or positions in the predicted trajectory.

    3. Compute Euclidean distances between predicted and interpolated GT positions.

    4. Return statistics like mean, max, min, RMSE.

"""

def interpolate_gt(gt_points, pred_points):
    """
    For each predicted point, find the closest segment in GT and interpolate.
    """
    interpolated_gt = []
    for px, py, pz in pred_points:
        min_dist = float('inf')
        closest_point = None
        for i in range(len(gt_points) - 1):
            # Segment endpoints
            a = gt_points[i, :2]
            b = gt_points[i + 1, :2]
            ab = b - a
            ap = np.array([px, py]) - a
            # Project point onto segment
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
            proj = a + t * ab
            dist = np.linalg.norm(np.array([px, py]) - proj)
            if dist < min_dist:
                min_dist = dist
                closest_point = proj
        interpolated_gt.append([closest_point[0], closest_point[1]])
    return np.array(interpolated_gt)

def compute_error_statistics(predicted, gt_interpolated):
    errors = np.linalg.norm(predicted[:, :2] - gt_interpolated, axis=1)
    stats = {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'mse': np.mean(errors**2),
        'rmse': np.sqrt(np.mean(errors**2)),
    }
    return stats, errors

def cumulative_distance(points):
    diffs = np.diff(points[:, :2], axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0], np.cumsum(dists)])

def relative_distance(dists):
    return dists/np.max(dists) * 100

def plot_errors(errors, frame_rate=15.0, trajectory=None):
    time_axis = np.arange(len(errors)) / frame_rate
    distance_axis = cumulative_distance(trajectory)
    relative_axis = relative_distance(distance_axis)
    print(relative_axis)

    plt.figure(figsize=(14, 5))

    # Plot error over time
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, errors, label="Error over Distance (meters)")
    plt.xlabel("Travelled distance (meters)")
    plt.ylabel("Error (meters)")
    plt.title("2D Error over Time")
    plt.grid(True)

    # Plot error over distance
    plt.subplot(1, 2, 2)
    rel = errors/relative_axis
    plt.plot(distance_axis, rel, color='orange', label="Error over Distance (m)")
    plt.xlabel("Travelled distance (meters)")
    plt.ylabel("Error (meters) / Travelled distance (%)")
    plt.ylim([0, np.quantile(rel, 0.99)])
    plt.title("2D Error over Distance")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_trajectory_error(seqs, plot=False, save=False):
    """
    Computes error statistics for a given trajectory sequence.
    
    Args:
        seqs: array of identifiers for the sequence
    
    Returns:
        Dictionary of error statistics
    """

    all_stats = {}

    for seq_num in seqs:
        dir_path = f"../datasets/predicted/trajectories/{seq_num}"
        pred_file = os.path.join(dir_path, "ZED_VIO_estimation.txt")
        gt_file = os.path.join(dir_path, "GT.txt")
        
        predicted = np.loadtxt(pred_file)
        gt = np.loadtxt(gt_file)
        
        interpolated_gt = interpolate_gt(gt, predicted)
        stats, errors = compute_error_statistics(predicted, interpolated_gt)
        
        all_stats[seq_num] = stats  

        if plot:
            plot_errors(errors, frame_rate=15.0, trajectory=predicted)
            plot_trajectories_from_folder(seq=seq_num, filter_keywords=["ZED_VIO", "GT"])

        print("\nTrajectory Error Statistics")
        print("=" * 35)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').capitalize():<20}: {value:.3f} meters")
        print("=" * 35 + "\n")

    if save:
         output_file = f"../datasets/predicted/error_stats"
         with open(output_file, "w") as f:
            json.dump(all_stats, f, indent=4)

    return 

if __name__ == "__main__":
    
    seqs = [str(i).zfill(2) for i in range(0,23)]
    stats = evaluate_trajectory_error(seqs, plot=False, save=True)

    


    
