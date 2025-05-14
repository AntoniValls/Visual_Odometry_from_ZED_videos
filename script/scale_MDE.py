import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import numpy as np
import torch
from tqdm import tqdm

# === FLAGS ===
PROCESS_ALL = True
VISUALIZE = False
SDE_REF = "Complex" # ZED or Complex

"""
This file contains the functions and the code to scale the MDE from "Distill Any Depth" using the SGBM Matcher (Complex version) SDE.
It first computes the global scale factor and then applies it to the MDEs
"""

def load_depth_tensor(path, device='cuda'):
    """Loads a .npy depth map as a PyTorch tensor on the desired device."""
    depth = np.load(path)
    return torch.from_numpy(depth).float().to(device)

def scale_monocular_to_metric_torch(mono_depth, stereo_depth, mask=False, only_scale=False, global_scale=None):
    epsilon = 1e-6
    mono_depth = 1.0 / (mono_depth + epsilon)

    depth_min, depth_max = 0.5, 20 
    device = mono_depth.device
    valid_mask = torch.ones_like(mono_depth, dtype=torch.bool, device=device)

    if mask:
        valid_mask = (
            (stereo_depth > depth_min) &
            (stereo_depth < depth_max) &
            torch.isfinite(stereo_depth) &
            (mono_depth > 0)
        )
        if valid_mask.sum() < 100:
            raise ValueError("Too few valid points to compute scale.")

    if global_scale is None:
        scale = torch.median(stereo_depth[valid_mask] / mono_depth[valid_mask])
    else:
        scale = torch.tensor(global_scale, device=device)

    if only_scale:
        return None, scale.item()

    scaled = mono_depth * scale
    mono_depth_scaled = (
        torch.clamp(scaled, min=depth_min, max=depth_max) if mask else scaled
    )

    return mono_depth_scaled, scale.item()

def compute_global_scale_torch(distill_dir, SDE_dir, device='cuda'):
    n_files = len([f for f in os.listdir(distill_dir) if f.startswith("depth_map_") and f.endswith(".npy")])
    scale_vector = []
    for i in tqdm(range(n_files), desc="Computing global scale"):
        distill_path = os.path.join(distill_dir, f"depth_map_{i}.npy")
        SDE_path = os.path.join(SDE_dir, f"depth_map_{i}.npy")

        distill_depth = load_depth_tensor(distill_path, device)
        SDE_depth = load_depth_tensor(SDE_path, device)
        
        try:
            _, scale = scale_monocular_to_metric_torch(distill_depth, SDE_depth, mask=False, only_scale=True)
        except ValueError:
            scale = 1.0
        if not np.isnan(scale):
            scale_vector.append(scale)

    return float(np.mean(scale_vector))

def scale_all_depth_maps_torch(distill_dir, SDE_dir, save_dir, global_scale, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    n_files = len([f for f in os.listdir(distill_dir) if f.startswith("depth_map_") and f.endswith(".npy")])

    for i in tqdm(range(n_files), desc="Scaling depth maps"):
        distill_path = os.path.join(distill_dir, f"depth_map_{i}.npy")
        SDE_path = os.path.join(SDE_dir, f"depth_map_{i}.npy")

        distill_depth = load_depth_tensor(distill_path, device)
        SDE_depth = load_depth_tensor(SDE_path, device)

        scaled_depth, _ = scale_monocular_to_metric_torch(distill_depth, SDE_depth, mask=False, global_scale=global_scale)

        # Move back to CPU and convert to numpy for saving
        np.save(os.path.join(save_dir, f"scaled_depth_map_{i}.npy"), scaled_depth.cpu().numpy())

if __name__ == '__main__':

    if PROCESS_ALL:
        distill_dir = "../datasets/predicted/depth_maps/00/Distill"
        if SDE_REF == "Complex":
            SDE_dir = "../datasets/predicted/depth_maps/00/Complex"
        elif SDE_REF == "ZED":
            SDE_dir = "../datasets/BIEL/00/depths"
        else:
            print("Error!")

        save_dir = os.path.join(distill_dir, f"scaled_by_{SDE_REF}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        global_scale = compute_global_scale_torch(distill_dir, SDE_dir, device=device)
        print(f"Global scale = {global_scale:.4f}")

        scale_all_depth_maps_torch(distill_dir, SDE_dir, save_dir, global_scale, device=device)


    if VISUALIZE:
        #indices = [0, 200, 600, 1000, 1500, 2500]
        indices = [1500, 1700]
        fig, axs = plt.subplots(len(indices), 3, figsize=(15, 3 * len(indices)))

        for row, i in enumerate(indices):
            original_path = os.path.join("../datasets/BIEL/00/image_0/", f"{str(i).zfill(6)}.png")
            if SDE_REF == "Complex":
                SDE_path = os.path.join("../datasets/predicted/depth_maps/00/Complex/", f"depth_map_{i}.npy")
            elif SDE_REF == "ZED":
                SDE_path = os.path.join("../datasets/BIEL/00/depths", f"depth_map_{i}.npy")
            else:
                print("Error!")
            SDE_path = os.path.join("../datasets/predicted/depth_maps/00/Complex/", f"depth_map_{i}.npy")
            distill_path = os.path.join("../datasets/predicted/depth_maps/00/Distill/", f"depth_map_{i}.npy")

            original = cv2.imread(original_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            SDE_depth = load_depth_tensor(SDE_path, device)
            distill_depth = load_depth_tensor(distill_path, device)

            try:
                distill_scaled, scale = scale_monocular_to_metric_torch(distill_depth, SDE_depth, mask=False, global_scale=444.2727)
                print(f"[{i}] Scale factor: {scale}")
            except ValueError as e:
                print(f"[{i}] Error scaling depth maps: {e}")
                distill_scaled = distill_depth

            # Convert tensors to numpy for visualization
            SDE_np = SDE_depth.cpu().numpy()
            distill_np = distill_scaled.cpu().numpy()

            axs[row, 0].imshow(original)
            axs[row, 0].set_title(f'Original Image (i={i})')
            axs[row, 0].axis('off')

            im1 = axs[row, 1].imshow(SDE_np, cmap='plasma',
                                    vmin=np.percentile(SDE_np, 5),
                                    vmax=np.percentile(SDE_np, 90))
            axs[row, 1].set_title('SDE Depth Map')
            fig.colorbar(im1, ax=axs[row, 1])

            im2 = axs[row, 2].imshow(distill_np, cmap='plasma',
                                    vmin=np.percentile(distill_np, 5),
                                    vmax=np.percentile(distill_np, 90))
            axs[row, 2].set_title('Scaled Mono Depth Map')
            fig.colorbar(im2, ax=axs[row, 2])

        plt.tight_layout()
        plt.show()
