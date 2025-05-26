import yaml
from dataloader import DataLoader
from visual_odometry import visual_odometry
import numpy as np
import cv2
import os

"""
Main script for motion estimation project based on stereo images.
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

    max_depths = [9.9]
    depth_methods = ['ZED']
    for depth_method in depth_methods:
        config['parameters']['depth_model'] = depth_method
        for j in range(len(max_depths)):
            try: 
                config['parameters']['max_depth'] = max_depths[j]
                print(f"Working on: \nMax Depth = {config['parameters']['max_depth']}")

                # Create Instances
                data_handler = DataLoader(sequence=sequence)
                data_handler.reset_frames()
                
                # Estimated trajectory by our algorithm pipeline
                trajectory = visual_odometry(data_handler, config, precomputed_depth_maps=True, plot=False, plotframes=False, verbose=False)
                
                # Saving the trajectory in a .txt file
                positions = trajectory[:, [0, 2, 1], 3]  
                save_dir = f"../datasets/predicted/trajectories/{sequence['type']}"
                os.makedirs(save_dir, exist_ok=True)
                np.savetxt(os.path.join(save_dir, f"{config['parameters']['detector']}_{config['parameters']['depth_model']}_{config['parameters']['ransac_method']}_maxdepth{config['parameters']['max_depth']}.txt"), positions, fmt="%.16f")

            except cv2.error as e:
                print(f"Failed for current threshold/max_depth combination:\n {e}")
                continue  # Skip to the next iteration of the loop
            
        