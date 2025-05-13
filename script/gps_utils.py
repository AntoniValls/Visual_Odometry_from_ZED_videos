import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_slam_keyframes(slam_keyfrm, theta):
    #Defining the rotation matrix
    rot_matrix = np.array([[np.cos(theta),-np.sin(theta)],
                           [np.sin(theta),np.cos(theta)]])
    
    #Rotate the slam keyframes
    slam_rot = np.dot(slam_keyfrm[:,[0,1]],rot_matrix)
    return slam_rot

def align_trajectories(slam_keyfrm):
    #theta = np.arctan2(gps_data[4,4],gps_data[4,3])
    # theta = np.absolute(np.arctan2(gps_data[:20,4],gps_data[:20,3]))
    # theta = np.mean(theta)
    theta = 0.1665810075*np.pi # Hardcoded value for theta in radians coming the previous formula in a file with the gps_data 
    print(f'Theta angle: {theta*180/np.pi} deg')
    scale_factor = 1
    #gps_start = gps_data[0,:2]
    gps_start = [455395.37362745,5425694.47262261] # Hardcoded value for the initial GPS point from a file with the gps_data
    # gps_start = [426069.90, 4581718.85] # Harcoded value for the initial GPS point for IRI_00
    print(f'Initial GPS point: {gps_start}\n')

    # Rotate the SLAM keyframes using x (first) and z (third)
    slam_rotated = rotate_slam_keyframes(slam_keyfrm, theta)
    
    # Apply scaling to the rotated SLAM keyframes
    scaled_slam = slam_rotated * scale_factor

    slam_start = scaled_slam[0,:2]
    print(f'Initial SLAM point: {slam_start}\n')

    offset = gps_start - slam_start  # Calculate translation offset
    print(f'Computed offset: {offset}\n')

    aligned_slam = scaled_slam[:,:2] + offset  # Apply translation to SLAM keyframes
    return aligned_slam