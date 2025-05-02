import yaml
from dataloader import DataLoader
from visual_odometry import visual_odometry
import numpy as np
import os

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

    # Create Instances
    data_handler = DataLoader(sequence=sequence)

    # Reset frames to start from the beginning of the image list on a new run. Because we are using generators
    data_handler.reset_frames()
    
    # Estimated trajectory by our algorithm pipeline
    trajectory = visual_odometry(data_handler, config, plot=False, plotframes=False, verbose=False)
    
    # Saving the trajectory in a .txt file
    positions = trajectory[:, [0, 2, 1], 3]  # [x, y, z] ordering (to match your plot)
    save_dir = f"../datasets/predicted/trajectories/{sequence['type']}"
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, f"{config['parameters']['detector']}_{config['parameters']['threshold']}_MD_20.txt"), positions, fmt="%.16f")

