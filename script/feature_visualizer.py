import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set parameters
detector1 = "lightglue"
detector2 = "LoFTR"
threshold = 100  # Top-k matches
idx = 5          # Frame index to visualize
show = True

# Append LightGlue path
current_dir = os.path.dirname(__file__)
lightglue_path = os.path.abspath(os.path.join(current_dir, '..', 'LightGlue'))
sys.path.append(lightglue_path)
from lightglue import viz2d
from lightglue.utils import load_image

# Image paths
sequence_dir = "../datasets/BIEL/00/"
img_0_path = os.path.join(sequence_dir, "image_0", sorted(os.listdir(os.path.join(sequence_dir, "image_0")))[idx])
img_1_path = os.path.join(sequence_dir, "image_0", sorted(os.listdir(os.path.join(sequence_dir, "image_0")))[idx + 1])

def load_matches(detector):
    cache_dir = f"../datasets/predicted/prefiltered_matches/00/{detector}"
    cache_path = os.path.join(cache_dir, f"matches_{idx}.npz")
    
    if not os.path.exists(cache_path):
        print(f"[ERROR] Cache file not found: {cache_path}")
        return None, None

    data = np.load(cache_path, allow_pickle=True)
    kpts0 = data["keypoint_left_first"]
    kpts1 = data["keypoint_left_next"]
    matches = data["matches"] if "matches" in data.files else None
    scores = data["scores"] if "scores" in data.files else None

    if detector == "lightglue":
        topk = np.argsort(-scores)[:threshold]
        filtered_matches = matches[topk]
        kpts0 = kpts0[filtered_matches[:, 0]]
        kpts1 = kpts1[filtered_matches[:, 1]]
    elif detector == "LoFTR":
        topk = np.argsort(-scores)[:threshold]
        kpts0 = kpts0[topk]
        kpts1 = kpts1[topk]

    return kpts0, kpts1

if __name__ == "__main__":

    # Load matches
    keypoint_left_first_1, keypoint_left_next_1 = load_matches(detector1)
    keypoint_left_first_2, keypoint_left_next_2 = load_matches(detector2)

    # Load images
    image_left = load_image(img_0_path)
    next_image = load_image(img_1_path)

     # LightGlue 
    _ = viz2d.plot_images([image_left, next_image])
    viz2d.plot_matches(keypoint_left_first_1, keypoint_left_next_1, color="lime", lw=0.2)
    plt.title(f"Top-{threshold} Matches with {detector1}")

    # LoFTR 
    _ = viz2d.plot_images([image_left, next_image])
    viz2d.plot_matches(keypoint_left_first_2, keypoint_left_next_2, color="deepskyblue", lw=0.2)
    plt.title(f"Top-{threshold} Matches with {detector2}")
    plt.show()

    plt.close()
