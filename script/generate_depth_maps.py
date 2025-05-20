import yaml
from dataloader import DataLoader
from sde import StereoDepthEstimator
from preprocessing import rectify_images
import os
from tqdm import tqdm
import numpy as np

"""
This script computes and optionally stores depth maps from a stereo image dataset. 
It supports processing all frames or a single frame and includes an option to rectify images.
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
    depth_model = config['parameters']['depth_model']
        
    # Create Instances
    data_handler = DataLoader(sequence=sequence)

    # Reset frames to start from the beginning of the image list on a new run. Because we are using generators
    data_handler.reset_frames()
    
    # Initialize the SD estimator
    sde = StereoDepthEstimator(config, data_handler.P0, data_handler.P1)

    # Obtain left and right images, and the camera parameters, given an index. "all" means all images. Set to an integer for single-frame processing
    index = 200

    if index != "all":
        # --- Single Frame Processing ---
        left_image, right_image = data_handler.get_two_images(index)

        # Compute and visualize the depth/disparity maps
        depth, _ = sde.estimate_depth(left_image, right_image, plot=True)
        
    else:
        # --- Full sequence processing ---
        num_frames = data_handler.frames
        iterator = range(num_frames - 1)
        iterator = tqdm(iterator, desc=f"Processing frames:")

        if data_handler.low_memory:
            data_handler.reset_frames()
            next_image = next(data_handler.left_images)

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

            # Compute depth map 
            depth_map, _ = sde.estimate_depth(image_left, image_right)
            
            # Save the computed depth map
            save_dir = f"../datasets/predicted/depth_maps/{sequence['type']}/{depth_model}"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"depth_map_{i}.npy"), depth_map)
