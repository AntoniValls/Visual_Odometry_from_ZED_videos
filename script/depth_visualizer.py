import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # === CONFIGURE FOLDERS ===
    zed_depth = "../datasets/BIEL/00/depths/"
    simple_depth = "../datasets/predicted/depth_maps/00/Simple"
    complex_depth = "../datasets/predicted/depth_maps/00/Complex"
    distill_depth_c = "../datasets/predicted/depth_maps/00/Distill/scaled_by_Complex"
    distill_depth_ZED = "../datasets/predicted/depth_maps/00/Distill/scaled_by_ZED"

    titles = ["ZED", "Simple SDE", "Complex SDE", "Distill MDE (C)", "Distill MDE (ZED)"]

    # === List of frame indices you want to visualize ===
    indices = [20, 800, 1500, 2082]
    fig, axs = plt.subplots(len(indices), 5, figsize=(15, 3 * len(indices)))

    for row, i in enumerate(indices):
        try:
            depth1 = np.load(os.path.join(zed_depth, f"depth_map_{i}.npy"))
            depth2 = np.load(os.path.join(complex_depth, f"depth_map_{i}.npy"))
            depth3 = np.load(os.path.join(distill_depth_c, f"scaled_depth_map_{i}.npy"))
            depth4 = np.load(os.path.join(distill_depth_ZED, f"scaled_depth_map_{i}.npy"))
            depth5 = np.load(os.path.join(simple_depth, f"depth_map_{i}.npy"))
        except FileNotFoundError as e:
            print(f"[ERROR] Missing file for frame {i}: {e}") 
            break 
        
        for j, depth in enumerate([depth1, depth5, depth2, depth3, depth4]):
            if j == 0:
                vmin, vmax = 0, 10
            else:
                vmin = np.percentile(depth, 5)
                vmax = np.percentile(depth, 90)
                print(vmin, vmax)
            img = axs[row, j].imshow(depth, cmap='inferno',
                            vmin=vmin,
                            vmax=vmax)
            axs[row, j].set_title(titles[j])
            fig.colorbar(img, ax=axs[row, j])
    plt.tight_layout()
    plt.show()

    

