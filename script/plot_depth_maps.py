import yaml
from dataloader import DataLoader
from utils import plot_depth_map_from_two
from preprocessing import rectify_images

"""
Plot depth maps
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
        
    # Create Instances
    data_handler = DataLoader(sequence=sequence)

    # Reset frames to start from the beginning of the image list on a new run. Because we are using generators
    data_handler.reset_frames()

    # Obtain left and right images, and the camera parameters, given an index
    index = 200
    left_image, right_image = data_handler.get_two_images(index)

    #gitdis(left_image, right_image)
    rectify_images(left_image, right_image)
    P0 = data_handler.P0
    P1 = data_handler.P1

    plot_depth_map_from_two(left_image, right_image, P0, P1, config)



