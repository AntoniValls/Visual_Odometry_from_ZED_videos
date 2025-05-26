import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Append LightGlue path
current_dir = os.path.dirname(__file__)
lightglue_path = os.path.abspath(os.path.join(current_dir, '..', 'LightGlue'))
sys.path.append(lightglue_path)

from lightglue.utils import load_image

# FLAGS -----------------------------------------------------
detector1 = "lightglue"
detector2 = "LoFTR"
detector3 = "Harris-SIFT"
idx = 200        # Frame index to visualize
show = True
# -----------------------------------------------------------

# Image paths
sequence_dir = "../datasets/BIEL/00/"
img_0_path = os.path.join(sequence_dir, "image_0", sorted(os.listdir(os.path.join(sequence_dir, "image_0")))[idx])
img_1_path = os.path.join(sequence_dir, "image_0", sorted(os.listdir(os.path.join(sequence_dir, "image_0")))[idx + 1])

def load_matches(detector):
    """Load and filter matches for a given detector."""
    cache_dir = f"../datasets/predicted/prefiltered_matches/00/{detector}"
    cache_path = os.path.join(cache_dir, f"matches_{idx}.npz")
    
    if not os.path.exists(cache_path):
        print(f"[ERROR] Cache file not found: {cache_path}")
        return None, None, None
    
    data = np.load(cache_path, allow_pickle=True)
    kpts0 = data["keypoint_left_first"]
    kpts1 = data["keypoint_left_next"]
    matches = data["matches"] if "matches" in data.files else None
    scores = data["scores"] if "scores" in data.files else None
    
    mask=np.argsort(-scores)[:1000]
    if detector == "lightglue":
        #mask = scores > 0.995
        top_scores = scores[mask]
        filtered_matches = matches[mask]
        kpts0 = kpts0[filtered_matches[:, 0]]
        kpts1 = kpts1[filtered_matches[:, 1]]
    elif detector == "LoFTR":
        #mask = scores > 0.995
        top_scores = scores[mask]
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
    elif detector == "Harris-SIFT":
        threshold = 1000
        kpts0 = kpts0[:threshold]
        kpts1 = kpts1[:threshold]
        top_scores = scores[:threshold]
    
    return kpts0, kpts1, top_scores

def draw_matches_matplotlib(img1, img2, kpts1, kpts2, ax, color='lime', title="Matches"):
    """Draw matches between two images using matplotlib."""
    # Handle different image formats (tensors and numpy arrays)
    def prepare_image(img):
        # Convert tensor to numpy if needed
        if hasattr(img, 'cpu'):  # PyTorch tensor
            img = img.cpu().numpy()
        elif hasattr(img, 'numpy'):  # Other tensor types
            img = img.numpy()
        
        if len(img.shape) == 3:
            # If channels are first (C, H, W), transpose to (H, W, C)
            if img.shape[0] == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                img = np.transpose(img, (1, 2, 0))
            # Convert BGR to RGB if needed
            if img.shape[2] == 3:
                # Check if it's likely BGR (opencv format) by looking at max values
                if img.max() > 1:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                else:
                    # Assume it's already RGB if values are normalized
                    img = img
        return img
    
    img1_rgb = prepare_image(img1)
    img2_rgb = prepare_image(img2)
    
    # Create side-by-side image
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    
    # Make sure both images have the same height
    if h1 != h2:
        if h1 > h2:
            img2_rgb = cv2.resize(img2_rgb, (w2, h1))
            h2 = h1
        else:
            img1_rgb = cv2.resize(img1_rgb, (w1, h2))
            h1 = h2
    
    # Ensure images are in the right format for matplotlib
    if img1_rgb.max() <= 1:
        img1_rgb = (img1_rgb * 255).astype(np.uint8)
    if img2_rgb.max() <= 1:
        img2_rgb = (img2_rgb * 255).astype(np.uint8)
    
    # Create combined image
    combined_img = np.hstack([img1_rgb, img2_rgb])
    
    # Display the combined image
    ax.imshow(combined_img)
    ax.set_title(title, fontsize=12, loc='left', fontweight='bold')
    ax.axis('off')
    
    # Draw matches
    if kpts1 is not None and kpts2 is not None and len(kpts1) > 0:
        # Adjust keypoints for the combined image (shift right image keypoints)
        kpts2_shifted = kpts2.copy()
        kpts2_shifted[:, 0] += w1  # Add width of first image
        
        # Draw lines connecting matches
        for i in range(min(len(kpts1), len(kpts2))):
            x1, y1 = kpts1[i]
            x2, y2 = kpts2_shifted[i]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5, alpha=0.7)
            
        # Draw keypoints
        ax.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=3, alpha=0.8)
        ax.scatter(kpts2_shifted[:, 0], kpts2_shifted[:, 1], c=color, s=3, alpha=0.8)

def normalize_scores(scores, method='minmax'):
    """Normalize scores to [0, 1] range for fair comparison."""
    if scores is None or len(scores) == 0:
        return scores
    
    if method == 'minmax':
        return (scores - scores.min()) / (scores.max() - scores.min())
    elif method == 'zscore':
        return (scores - scores.mean()) / scores.std()
    return scores

def plot_matches_comparison(image_left, next_image, detectors_data):
    """Create matches visualization with 3x1 subplot layout."""
    # Create figure with subplots (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(f'Feature Matching Comparison:', horizontalalignment="center", fontsize=16, fontweight='bold')
    
    # Define colors for each detector
    colors = ["lime", "deepskyblue", "orange"]
    detector_names = [detector1, detector2, detector3]
    
    # Plot matches for each detector (one per row)
    for i, (detector_name, (kpts0, kpts1, scores), color) in enumerate(zip(detector_names, detectors_data, colors)):
        ax = axes[i]
        if kpts0 is None or kpts1 is None:
            ax.text(0.5, 0.5, f'No data for {detector_name}', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{detector_name} - No Data")
            continue
            
        draw_matches_matplotlib(image_left, next_image, kpts0, kpts1, ax, 
                              color=color, title=f"{detector_name}")
    
    plt.tight_layout()
    return fig

def plot_score_histograms(detectors_data):
    """Create score histograms with 3x1 subplot layout."""
    # Create figure with subplots (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Score Distributions: ', horizontalalignment="center", fontsize=16, fontweight='bold')
    
    # Define colors for each detector
    colors = ["lime", "deepskyblue", "orange"]
    detector_names = [detector1, detector2, detector3]
    
    # Plot individual histograms - each with its own scale
    for i, (detector_name, (kpts0, kpts1, scores), color) in enumerate(zip(detector_names, detectors_data, colors)):
        ax = axes[i]
        if scores is None or len(scores) == 0:
            ax.text(0.5, 0.5, f'No scores for {detector_name}', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{detector_name} Scores - No Data")
            continue
        
        # Plot histogram
        ax.hist(scores, bins=30, alpha=0.7, color=color, edgecolor="black")
        ax.set_title(f"{detector_name}", loc="left", fontsize=12, fontweight='bold')
        ax.set_xlabel("Confidence Score", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ax.text(0.02, 0.98, f'μ={mean_score:.3f}\nσ={std_score:.3f}\nrange=[{np.min(scores):.3f}, {np.max(scores):.3f}]', 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_normalized_comparison(detectors_data, detector_names, colors):
    """Create a separate plot comparing normalized scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Score Distribution Comparison', fontsize=14, fontweight='bold')
    
    # Left plot: Original scores (separate y-axes if needed)
    valid_scores = []
    valid_detectors = []
    valid_colors = []
    
    for detector_name, (kpts0, kpts1, scores), color in zip(detector_names, detectors_data, colors):
        if scores is not None and len(scores) > 0:
            valid_scores.append(scores)
            valid_detectors.append(detector_name)
            valid_colors.append(color)
    
    if len(valid_scores) > 0:
        # Check if Harris-SIFT has very different range
        harris_idx = None
        for i, name in enumerate(valid_detectors):
            if 'harris' in name.lower() or 'sift' in name.lower():
                harris_idx = i
                break
        
        if harris_idx is not None and len(valid_scores) > 1:
            harris_scores = valid_scores[harris_idx]
            other_scores = [s for i, s in enumerate(valid_scores) if i != harris_idx]
            other_names = [n for i, n in enumerate(valid_detectors) if i != harris_idx]
            other_colors = [c for i, c in enumerate(valid_colors) if i != harris_idx]
            
            # Check if ranges are very different
            harris_range = harris_scores.max() - harris_scores.min()
            other_ranges = [s.max() - s.min() for s in other_scores]
            
            if len(other_ranges) > 0 and harris_range > 10 * max(other_ranges):
                # Plot Harris-SIFT separately with secondary y-axis
                ax1_twin = ax1.twinx()
                ax1_twin.hist(harris_scores, bins=30, alpha=0.6, 
                            color=valid_colors[harris_idx], edgecolor="black",
                            label=valid_detectors[harris_idx])
                ax1_twin.set_ylabel(f"{valid_detectors[harris_idx]} Frequency", color=valid_colors[harris_idx])
                ax1_twin.tick_params(axis='y', labelcolor=valid_colors[harris_idx])
                
                # Plot others on main axis
                for scores, name, color in zip(other_scores, other_names, other_colors):
                    ax1.hist(scores, bins=30, alpha=0.6, label=name, color=color, edgecolor="black")
                
                ax1.set_ylabel("Frequency (LightGlue & LoFTR)")
                ax1.legend(loc='upper left')
                ax1_twin.legend(loc='upper right')
            else:
                # Plot all together
                for scores, name, color in zip(valid_scores, valid_detectors, valid_colors):
                    ax1.hist(scores, bins=30, alpha=0.6, label=name, color=color, edgecolor="black")
                ax1.legend()
        else:
            # Plot all together
            for scores, name, color in zip(valid_scores, valid_detectors, valid_colors):
                ax1.hist(scores, bins=30, alpha=0.6, label=name, color=color, edgecolor="black")
            ax1.legend()
    
    ax1.set_title("Original Score Distributions")
    ax1.set_xlabel("Confidence Score")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Normalized scores for fair comparison
    for scores, name, color in zip(valid_scores, valid_detectors, valid_colors):
        normalized_scores = normalize_scores(scores)
        ax2.hist(normalized_scores, bins=30, alpha=0.6, label=name, color=color, edgecolor="black")
    
    ax2.set_title("Normalized Score Distributions")
    ax2.set_xlabel("Normalized Confidence Score [0, 1]")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_match_statistics(detectors_data, detector_names):
    """Print detailed statistics for each detector."""
    print("\n" + "="*80)
    print(f"MATCH STATISTICS")
    print("="*80)
    
    for detector_name, (kpts0, kpts1, scores) in zip(detector_names, detectors_data):
        if kpts0 is None or scores is None:
            print(f"{detector_name:15}: No data available")
            continue
            
        print(f"{detector_name:15}: {len(scores):3d} matches | "
              f"Score: {np.mean(scores):.4f}±{np.std(scores):.4f} | "
              f"Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print("="*80)

if __name__ == "__main__":
    try:
        # Load matches for all detectors
        print(f"Loading matches for frame {idx}...")
        keypoint_left_first_1, keypoint_left_next_1, scores1 = load_matches(detector1)
        keypoint_left_first_2, keypoint_left_next_2, scores2 = load_matches(detector2)
        keypoint_left_first_3, keypoint_left_next_3, scores3 = load_matches(detector3)
        print(len(scores3))
        
        # Load images
        print("Loading images...")
        image_left = load_image(img_0_path)
        next_image = load_image(img_1_path)
        
        # Prepare data for plotting
        detectors_data = [
            (keypoint_left_first_1, keypoint_left_next_1, scores1),
            (keypoint_left_first_2, keypoint_left_next_2, scores2),
            (keypoint_left_first_3, keypoint_left_next_3, scores3)
        ]
        detector_names = [detector1, detector2, detector3]
        colors = ["lime", "deepskyblue", "orange"]
        
        # Print statistics
        print_match_statistics(detectors_data, detector_names)
        
        # Create match visualization plot (3x1)
        print("Creating matches visualization...")
        fig1 = plot_matches_comparison(image_left, next_image, detectors_data)
        
        # Create score histograms plot (3x1)
        print("Creating score histograms...")
        fig2 = plot_score_histograms(detectors_data)
        
        if show:
            plt.show()
        else:
            # Save the plots if not showing
            output_path1 = f"matches_visualization_frame_{idx}.png"
            output_path2 = f"score_histograms_frame_{idx}.png"
            fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
            fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {output_path1}, {output_path2}")
        
        plt.close('all')
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()