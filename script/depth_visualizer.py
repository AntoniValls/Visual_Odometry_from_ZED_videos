import os
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from PIL import Image

"Script for visualizing and comparing the different stereo depth estimation algorithms used."
if __name__ == "__main__":

    original_images = "../datasets/BIEL/00/image_0/"
    zed_depth = "../datasets/BIEL/00/depths/"
    simple_depth = "../datasets/predicted/depth_maps/00/Simple"
    complex_depth = "../datasets/predicted/depth_maps/00/Complex"
    hitnet_depth = "../datasets/predicted/depth_maps/00/HitNet"

    titles = ["Original", "ZED", "Simple", "Complex", "HitNet"]

    indices = [1500,2000]
    fig, axs = plt.subplots(len(indices), len(titles), figsize=(17, 3 * len(indices)))

    vmin = 0
    vmax = 10
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'inferno'

    for row, i in enumerate(indices):
        try:
            # Load original image
            rgb_img = Image.open(os.path.join(original_images, f"{i:06d}.png"))

            # Load depth maps
            depth1 = np.load(os.path.join(zed_depth, f"depth_map_{i}.npy"))
            depth2 = np.load(os.path.join(simple_depth, f"depth_map_{i}.npy"))
            depth3 = np.load(os.path.join(complex_depth, f"depth_map_{i}.npy"))
            depth5 = np.load(os.path.join(hitnet_depth, f"depth_map_{i}.npy"))

        except FileNotFoundError as e:
            print(f"[ERROR] Missing file for frame {i}: {e}")
            continue

        # Plot original image
        axs[row, 0].imshow(rgb_img)
        axs[row, 0].set_title(titles[0])
        axs[row, 0].axis('off')

        # Plot depth maps
        for j, depth in enumerate([depth1, depth2, depth3, depth5]):
            im = axs[row, j + 1].imshow(depth, cmap=cmap, norm=norm)
            axs[row, j + 1].set_title(titles[j + 1])
            axs[row, j + 1].axis('off')

    # Single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Depth (m)', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.show()