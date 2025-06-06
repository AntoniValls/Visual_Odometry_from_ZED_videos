import cv2
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

class DataLoader(object):
    def __init__(self, sequence, low_memory=True):
        """
        :params str sequence: Image path.
        :params bool lidar: lidar data.
        :params bool low_memory: If you have low memory in your laptop(e.g. Your RAM < 32GB), set the value to True.    
        """

        self.low_memory = low_memory

        # Set the directories for images and ground truth
        self.sequence_dir = os.path.join(sequence['main_path'], sequence['type']) + "/"
        if sequence['ground_truth']:
            self.poses_dir = sequence['pose_path'] + f"{sequence['type']}_IRI.txt"
            poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)

        # Get the list of images in the left and right camera directories
        self.left_camera_images = sorted(
            os.listdir(self.sequence_dir + 'image_0'))
        self.right_camera_images = sorted(
            os.listdir(self.sequence_dir + 'image_1'))

        # Verify counts match (for stereo alignment)
        assert len(self.left_camera_images) == len(self.right_camera_images), \
            "Mismatch between left/right image counts!"

        # Left frame are our reference frames
        self.frames = len(self.left_camera_images)

        # Extract the calibration parameters from P matrix
        self.calibration_dir = sequence["main_path"]
        calibration = pd.read_csv(
            self.calibration_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)

        # Projection Matrices
        self.P0 = np.array(calibration.loc['P0:']).reshape((3, 4))
        self.P1 = np.array(calibration.loc['P1:']).reshape((3, 4))

        # Extract timestamps from the file
        self.timestamps = self.get_timestamps()
        # Image loader
        if self.low_memory:

            # Utilzing generators for low RAM
            self.reset_frames()

            # Image dimensions
            self.image_height = 720
            self.image_width = 1280

        else:
            # Store all the images into the memory
            self.left_images = []
            self.right_images = []

            # Iterate through all the images to store in the above defined variables
            for i, left in enumerate(self.left_camera_images):
                right = self.right_camera_images[i]
                self.left_images.append(cv2.imread(
                    self.sequence_dir + 'image_0/' + left))
                self.right_images.append(cv2.imread(
                    self.sequence_dir + 'image_1/' + right))

            self.image_height = 720
            self.image_width = 1280

    def get_timestamps(self):
        """
        Get the timestamps of the images.
        
        :return: Numpy array of timestamps.
        """
        timestamps = []

        with open(self.sequence_dir + 'vislam_data.txt', 'r') as file:
            for line in file:
                data = json.loads(line)
                timestamps.append(data['timestamp'])

        return np.array(timestamps)
    def reset_frames(self):
        """ 
        Reset generators, not lists. These generators 
        yield  images (cv2.imread(...)) one at a time 
        when accessed, rather than loading the entire dataset into RAM
        """
        self.left_images = (cv2.imread(self.sequence_dir + 'image_0/' + left)
                            for left in self.left_camera_images)
        self.right_images = (cv2.imread(self.sequence_dir + 'image_1/' + right)
                             for right in self.right_camera_images)
        pass
    
    def get_two_images(self, index=0):
        """
        Load and return the left and right images at a given index.
        
        :param index: Index of the images to retrieve.
        :return: Tuple of (left_image, right_image)
        """
        if self.low_memory:
            left_path = os.path.join(self.sequence_dir, 'image_0', self.left_camera_images[index])
            right_path = os.path.join(self.sequence_dir, 'image_1', self.right_camera_images[index])
            left_image = cv2.imread(left_path)
            right_image = cv2.imread(right_path)
        else:
            left_image = self.left_images[index]
            right_image = self.right_images[index]

        return left_image, right_image