import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # === CONFIGURE FOLDERS ===
    zed_depth = "../datasets/BIEL/00/depths/"
    simple_depth = "../datasets/predicted/depth_maps/00/Simple"
    complex_depth = "../datasets/predicted/depth_maps/00/Complex"
    highres_depth = "../datasets/predicted/depth_maps/00/HighRes"
    hitnet_depth = "../datasets/predicted/depth_maps/00/HitNet"
    fastacv_depth = "../datasets/predicted/depth_maps/00/FastACV"

    titles = ["ZED", "Simple", "Complex", "HighRes", "HitNet", "FastACV"]

    # === List of frame indices you want to visualize ===
    indices = [20, 1300, 1490, 3000]
    fig, axs = plt.subplots(len(indices), len(titles), figsize=(15, 3 * len(indices)))

    for row, i in enumerate(indices):
        try:
            depth1 = np.load(os.path.join(zed_depth, f"depth_map_{i}.npy"))
            depth3 = np.load(os.path.join(complex_depth, f"depth_map_{i}.npy"))
            depth4 = np.load(os.path.join(highres_depth, f"depth_map_{i}.npy"))
            depth2 = np.load(os.path.join(simple_depth, f"depth_map_{i}.npy"))
            depth5 = np.load(os.path.join(hitnet_depth, f"depth_map_{i}.npy"))
            depth6 = np.load(os.path.join(fastacv_depth, f"depth_map_{i}.npy"))


        except FileNotFoundError as e:
            print(f"[ERROR] Missing file for frame {i}: {e}") 
            break 
        
        for j, depth in enumerate([depth1, depth2, depth3, depth4, depth5, depth6]):
            vmin = 0
            vmax = 25
            img = axs[row, j].imshow(depth, cmap='inferno',
                            vmin=vmin,
                            vmax=vmax)
            axs[row, j].set_title(titles[j])
            #fig.colorbar(img, ax=axs[row, j])
    plt.tight_layout()
    plt.show()

    

