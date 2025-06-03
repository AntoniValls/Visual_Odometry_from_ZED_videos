
import numpy as np
import matplotlib.pyplot as plt
import os, json
import cmocean
import pandas as pd
from collections import defaultdict
from glob import glob
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

def compute_error_statistics(predicted, gt_interpolated, subset=None):
    
    # Subset of frames
    if subset != None:
        predicted = predicted[:subset]
        gt_interpolated = gt_interpolated[:subset]

    errors = np.linalg.norm(predicted[:, :2] - gt_interpolated, axis=1)
    stats = {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'mse': np.mean(errors**2),
        'rmse': np.sqrt(np.mean(errors**2)),
        'fda' : errors[-1]
    }
    return stats, errors

def cumulative_distance(points):
    diffs = np.diff(points[:, :2], axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0], np.cumsum(dists)])

def relative_distance(dists):
    return dists/np.max(dists) * 100

def plot_errors(seq_list= None, from_csv=False, layout='grid', figsize = (12, 8)):
    """
    Computes error statistics for given trajectory sequences or precomputed errors from csv files.
    """

    dataframes = []
    output_path = f"../datasets/predicted/stats"

    # If not read from precomputed
    if not from_csv:
        all_stats = {}

        for seq_num in seq_list:
            dir_path = f"../datasets/predicted/trajectories/{seq_num}"
            pred_file = os.path.join(dir_path, "ZED_VIO_estimation.txt")
            gt_file = os.path.join(dir_path, "GT.txt")
            
            predicted = np.loadtxt(pred_file)
            gt = np.loadtxt(gt_file)
            
            interpolated_gt = interpolate_gt(gt, predicted)
            stats, errors = compute_error_statistics(predicted, interpolated_gt)
            
            all_stats[seq_num] = stats 

            # Save errors in a CSV file
            distance = cumulative_distance(predicted)
            relative_d = relative_distance(distance)
            df = pd.DataFrame({
            'distance': distance,
            'relative_distance': relative_d,
            'errors': errors})

             # Save to CSV file
            dataframes.append(df)
            df.to_csv(os.path.join(output_path,f'data_results_{seq_num}.csv'), index=False)

        # Save stats in a JSON file
        output_file = os.path.join(output_path, "error_stats.json")
        with open(output_file, "w") as f:
            json.dump(all_stats, f, indent=4)
    
    else:
        # Load data from CSV files
        all_csv_files = glob(os.path.join(output_path, "*.csv"))
        
        # Keep only those that contain are in 'seq_list'
        csv_files = []
        for file_path in all_csv_files:
            filename = os.path.basename(file_path)
            if any(keyword in filename for keyword in seq_list):
                csv_files.append(file_path)
    
        csv_files = sorted(csv_files)

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        
        # Sort the dataframes list based on the last 'distance' value in each dataframe --> Cool for plotting
        dataframes = sorted(dataframes, key=lambda df: df['distance'].iloc[-1], reverse=True)
        
        if not dataframes:
            print("No valid CSV files found!")
            return
        
        # Load stats
        with open(os.path.join(output_path, "error_stats.json"), 'r') as f:
            all_stats = json.load(f)
            
    # Initialize dictionary to accumulate values for each stat key
    mean_stats = defaultdict(list)

    for seq, stats in all_stats.items():
        print(f"\nTrajectory Error Statistics: Sequence {seq}")
        print("=" * 35)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').capitalize():<20}: {value:.3f} meters")
            mean_stats[key].append(value)
        print("=" * 35 + "\n")

    # Compute and display the mean for each stat key
    print("=" * 45)
    print("\nOverall Mean Trajectory Error Statistics")
    print("=" * 45)
    for key, values in mean_stats.items():
        mean_value = np.mean(values)
        print(f"{key.replace('_', ' ').capitalize():<20}: {mean_value:.3f} meters")
   
    n_sequences = len(seq_list)

    # Define the color palette
    colors = [cmocean.cm.thermal(i) for i in np.linspace(0, 1, n_sequences)]
    
    if layout == 'grid':
        
        # Create grid layout with separate subplots for each sequence
        n_cols = 2
        n_rows = n_sequences
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_sequences == 1:
            axes = axes.reshape(1, -1)
        
        for i, (df, name, color) in enumerate(zip(dataframes, seq_list, colors)):
            
            # Error over Distance
            axes[i, 0].plot(df['distance'], df['errors'], color=color, label=name)
            axes[i, 0].set_xlabel("Distance (meters)")
            axes[i, 0].set_ylabel("Error (meters)")
            axes[i, 0].set_title(f"{name} - Error over Distance")
            axes[i, 0].grid(True)
            
            # Error over distance (relative)
            rel_errors = df['errors'] / (df['relative_distance'] + 1e-9)
            axes[i, 1].plot(df['distance'], rel_errors, color=color, label=name)
            axes[i, 1].set_xlabel("Travelled distance (meters)")
            axes[i, 1].set_ylabel("Error / Travelled distance (%)")
            axes[i, 1].set_ylim([0, np.quantile(rel_errors, 0.99)])
            axes[i, 1].set_title(f"{name} - Relative Error over Distance")
            axes[i, 1].grid(True)
            axes[i, 1].legend()
    
    elif layout == 'overlay':
        # Create overlaid plots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        all_rel_errors = []
        
        for i, (df, name, color) in enumerate(zip(dataframes, seq_list, colors)):
           
            # Error over distance
            axes[0].plot(df['distance'], df['errors'], color=color, label=name, alpha=0.8, linewidth=2)
            
            # Error over distance (relative)
            rel_errors = df['errors'] / (df['relative_distance'] + 1e-9)
            axes[1].plot(df['distance'], rel_errors, color=color, label=name, alpha=0.8, linewidth=2)
            all_rel_errors.extend(rel_errors)
        
        axes[0].set_xlabel("Distance (meters)")
        axes[0].set_ylabel("Error (meters)")
        axes[0].set_title("Error over Distance")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel("Travelled distance (meters)")
        axes[1].set_ylabel("Error / Travelled distance (%)")
        axes[1].set_title("Relative Error over Distance")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=2)
        axes[1].set_ylim([0, np.quantile(all_rel_errors, 0.99)])
    
    elif layout == 'both':
        # Create both views
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 1.2))
        
        all_rel_errors = []
        
        # Top row: overlay plots
        for i, (df, name, color) in enumerate(zip(dataframes, seq_list, colors)):
            rel_errors = df['errors'] / (df['relative_distance'] + 1e-9)
            all_rel_errors.extend(rel_errors)
            
            # Overlay plots
            axes[0, 0].plot(df['distance'], df['errors'], color=color, label=name, alpha=0.8, linewidth=2)
            axes[0, 1].plot(df['distance'], rel_errors, color=color, label=name, alpha=0.8, linewidth=2)
        
        axes[0, 0].set_xlabel("Distance (meters)")
        axes[0, 0].set_ylabel("Error (meters)")
        axes[0, 0].set_title("Error over Distance - Overlay")
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel("Travelled distance (meters)")
        axes[0, 1].set_ylabel("Error / Travelled distance (%)")
        axes[0, 1].set_title("Relative Error over Distance - Overlay")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(ncols=2)
        axes[0, 1].set_ylim([0, np.quantile(all_rel_errors, 0.99)])
        
        # Bottom row: individual comparison
        plot_comparison_bars(dataframes, seq_list, axes[1, :])
    
    plt.tight_layout()
    plt.show()

    return all_stats, dataframes

def plot_comparison_bars(dataframes, 
                        sequence_names, 
                        axes):
    """Helper function to create comparison bar charts."""
    
    # Calculate statistics
    stats = {}
    for df, name in zip(dataframes, sequence_names):
        rel_errors = df['errors'] / (df['relative_distance'] + 1e-9)
        stats[name] = {
            'mean_error': df['errors'].mean(),
            'max_error': df['errors'].max(),
            'mean_rel_error': rel_errors.mean(),
            'final_distance': df['distance'].iloc[-1] if len(df) > 0 else 0
        }
    
    names = list(stats.keys())
    x_pos = np.arange(len(names))
    
    # Mean and Max error comparison
    mean_errors = [stats[name]['mean_error'] for name in names]
    max_errors = [stats[name]['max_error'] for name in names]
    
    width = 0.35
    axes[0].bar(x_pos - width/2, mean_errors, width, label='Mean Error', alpha=0.8, color=cmocean.cm.haline(0.1))
    axes[0].bar(x_pos + width/2, max_errors, width, label='Max Error', alpha=0.8, color=cmocean.cm.haline(0.8))
    axes[0].set_title('Error Comparison')
    axes[0].set_ylabel('Error (meters)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Relative error and distance comparison
    mean_rel_errors = [stats[name]['mean_rel_error'] for name in names]
    final_distances = [stats[name]['final_distance'] for name in names]
    
    ax2 = axes[1]
    ax3 = ax2.twinx()
    
    bars1 = ax2.bar(x_pos - width/2, mean_rel_errors, width, label='Mean Rel. Error', alpha=0.8, color=cmocean.cm.haline(0.1))
    bars2 = ax3.bar(x_pos + width/2, final_distances, width, label='Total Distance', alpha=0.8, color=cmocean.cm.haline(0.8))
    
    ax2.set_title('Relative Error & Distance Comparison')
    ax2.set_ylabel('Mean Relative Error (%)', color=cmocean.cm.haline(0.1))
    ax3.set_ylabel('Total Distance (meters)', color=cmocean.cm.haline(0.8))
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

def plot_combined_error_vs_distance(dataframes, figsize=(10, 6), alpha=0.3, point_size=10, bins=100):
    """
    Plots a combined scatterplot of error vs distance using multiple dataframes,
    and overlays a line showing the average error as a function of distance.

    Parameters:
    - dataframes: list of pandas DataFrames, each with 'distance' and 'errors' columns.
    - figsize: tuple, size of the figure.
    - alpha: float, transparency of scatter points.
    - point_size: int, size of scatter points.
    - bins: int, number of bins to group distances for averaging.
    """
    # Combine all data
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Scatter plot of all points
    plt.figure(figsize=figsize)
    scatter_color = cmocean.cm.thermal(0.9)
    avg_color = cmocean.cm.thermal(0.2)

    plt.scatter(combined_df['distance'], combined_df['errors'],
                alpha=alpha, s=point_size, color=scatter_color, label='All points')

    # Bin distances to compute average error per bin
    binned = pd.cut(combined_df['distance'], bins)
    avg_df = combined_df.groupby(binned)['errors'].mean().reset_index()

    # Use bin midpoints for plotting
    bin_centers = [interval.mid for interval in avg_df['distance']]
    plt.plot(bin_centers, avg_df['errors'], color=avg_color, linewidth=4, label='Average error')

    plt.xlabel("Distance (m)")
    plt.ylabel("Error (m)")
    plt.title("Combined Scatterplot: Error vs Distance with Average Line")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    seqs = [str(i).zfill(2) for i in range(0,23)]

    _, dataframes = plot_errors(seqs, from_csv=True, layout="both")
    plot_combined_error_vs_distance(dataframes)

    


    
