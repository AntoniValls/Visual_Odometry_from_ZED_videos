import yaml
from dataloader import DataLoader
from utils import stereo_depth
from preprocessing import rectify_images
import os
from tqdm import tqdm
import numpy as np

"""
This script computes and optionally stores depth maps from a stereo image dataset. 
It supports processing all frames or a single frame and includes an option to rectify images
"""

# Driver Code
if __name__ == '__main__':

    # Load Config File
    with open("../config/IRI_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)

    # Declare Necessary Variables
    sequence = config['data']
    rectify = config['parameters']['rectified']
        
    # Create Instances
    data_handler = DataLoader(sequence=sequence)

    # Reset frames to start from the beginning of the image list on a new run. Because we are using generators
    data_handler.reset_frames()

    # Obtain left and right images, and the camera parameters, given an index. "all" means all images. Set to an integer for single-frame processing
    index = "all"

    if index != "all":
        # --- Single Frame Processing ---
        left_image, right_image = data_handler.get_two_images(index)

        # Apply rectification if needed
        if rectify:
            # Rectify without modifying intrinsic matrices P0 and P1
            left_image, right_image, *_ = rectify_images(left_image, right_image) 
        
        # Compute and visualize the depth/disparity maps
        stereo_depth(left_image, right_image, data_handler.P0, data_handler.P1, config, stereo_complex=True, plot=True)

    else:
        # --- Full sequence processing ---
        num_frames = data_handler.frames
        iterator = range(num_frames - 1)
        iterator = tqdm(iterator, desc="Processing frames")

        if data_handler.low_memory:
            data_handler.reset_frames()
            next_image = next(data_handler.left_images)

        # Initialize rectification maps (only computed once)
        map1, map2 = None, None

        for i in iterator:
            # Retrieve current stereo pair
            if data_handler.low_memory:
                image_left = next_image
                image_right = next(data_handler.right_images)
                next_image = next(data_handler.left_images)
            else:
                image_left = data_handler.left_images[i]
                image_right = data_handler.right_images[i]
                next_image = data_handler.left_images[i+1]

            # Apply rectification if needed
            if rectify: 
                if i == 0:
                    # First time we need to obtain the rectification maps
                    image_left, image_right, _, _, map1, map2 = rectify_images(image_left, image_right) # We not change the camera intrinsic matrices P0 and P1
                else:
                    image_left, image_right, *_ = rectify_images(image_left, image_right, i, map1, map2)

            # Compute depth map (no visualization)
            depth_map, _ = stereo_depth(image_left, image_right, data_handler.P0, data_handler.P1, config, stereo_complex=True, plot=False)

            # Save the computed depth map
            save_dir = f"../datasets/predicted/depth_maps/{sequence['type']}"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"depth_map_{i}.npy"), depth_map)

    # --- Experimental configurations (for debugging/benchmarking) ---
    # stereo_depth(recleft_image, recright_image, nP0, nP1, config, stereo_complex=False, plot=True, title="R-NP-NC")
    # stereo_depth(recleft_image, recright_image, data_handler.P0, data_handler.P1, config,stereo_complex=False, plot=True, title="R-P-NC")
    # stereo_depth(left_image, right_image, data_handler.P0, data_handler.P1, config,stereo_complex=False, plot=True, title="NR-P-NC")

    # stereo_depth(recleft_image, recright_image, nP0, nP1, config, stereo_complex=True, plot=True, title="R-NP-C")
    # stereo_depth(recleft_image, recright_image, data_handler.P0, data_handler.P1, config,stereo_complex=True, plot=True, title="R-P-C")
    # stereo_depth(left_image, right_image, data_handler.P0, data_handler.P1, config,stereo_complex=True, plot=True, title="NR-P-C") # BEST ONE

