import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import scale_monocular_to_metric


if __name__ == '__main__':
    
    for i in tqdm(range(2440, 3015)):
        # Load the depth maps
        complex_path = os.path.join("../datasets/predicted/depth_maps/00/Complex/", f"depth_map_{i}.npy")
        distill_path = os.path.join("../datasets/predicted/depth_maps/00/Distill/", f"depth_map_{i}.npy")

        complex_depth = np.load(complex_path)
        distill_depth = np.load(distill_path)
    
        # Scale the monocular depth map to match the stereo depth map
        try:
            scaled_mono_depth, scale = scale_monocular_to_metric(distill_depth, complex_depth)
            print(f"Scale factor: {scale}")
        except ValueError as e:
            print(f"Error scaling depth maps: {e}")
            scaled_mono_depth = distill_depth  # Fallback to original depth map
            scale = 1.0     

        # Save the scaled depth map
        save_dir = "../datasets/predicted/depth_maps/00/Distill/scaled"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"scaled_depth_map_{i}.npy"), scaled_mono_depth)

    # # Plotting
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # vmin = np.percentile(complex_depth, 5)
    # vmax = np.percentile(complex_depth, 95)
    # im0 = axs[0].imshow(complex_depth, cmap='plasma', vmin=vmin, vmax=vmax)
    # axs[0].set_title('Complex Depth Map')
    # plt.colorbar(im0, ax=axs[0])

    # im1 = axs[1].imshow(scaled_mono_depth, cmap='plasma')
    # axs[1].set_title('Scaled Mono Depth Map')
    # plt.colorbar(im1, ax=axs[1])

    # plt.tight_layout()
    # plt.show()
    