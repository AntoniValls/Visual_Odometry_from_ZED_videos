import os
import sys
import cv2
import json
import enum
import argparse
import numpy as np
import pyzed.sl as sl
import pathlib as Path


def progress_bar(percent_done, bar_length=50):
    #Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def IMUDataExtraction(imu_data):
    out = {}
    out["is_available"] = imu_data.is_available
    out["timestamp"] = imu_data.timestamp.get_nanoseconds()
    out["pose"] = {}
    pose = sl.Transform()
    imu_data.get_pose(pose)
    out["pose"]["translation"] = [0, 0, 0]
    out["pose"]["translation"][0] = pose.get_translation().get()[0]
    out["pose"]["translation"][1] = pose.get_translation().get()[1]
    out["pose"]["translation"][2] = pose.get_translation().get()[2]
    out["pose"]["quaternion"] = [0, 0, 0, 0]
    out["pose"]["quaternion"][0] = pose.get_orientation().get()[0]
    out["pose"]["quaternion"][1] = pose.get_orientation().get()[1]
    out["pose"]["quaternion"][2] = pose.get_orientation().get()[2]
    out["pose"]["quaternion"][3] = pose.get_orientation().get()[3]
    out["pose_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(3):
            out["pose_covariance"][i * 3 + j] = imu_data.get_pose_covariance().r[i][j] 

    out["angular_velocity"] = [0, 0, 0]
    out["angular_velocity"][0] = imu_data.get_angular_velocity()[0]
    out["angular_velocity"][1] = imu_data.get_angular_velocity()[1]
    out["angular_velocity"][2] = imu_data.get_angular_velocity()[2]

    out["linear_acceleration"] = [0, 0, 0]
    out["linear_acceleration"][0] = imu_data.get_linear_acceleration()[0]
    out["linear_acceleration"][1] = imu_data.get_linear_acceleration()[1]
    out["linear_acceleration"][2] = imu_data.get_linear_acceleration()[2]

    out["angular_velocity_uncalibrated"] = [0, 0, 0]
    out["angular_velocity_uncalibrated"][0] = imu_data.get_angular_velocity_uncalibrated()[
        0]
    out["angular_velocity_uncalibrated"][1] = imu_data.get_angular_velocity_uncalibrated()[
        1]
    out["angular_velocity_uncalibrated"][2] = imu_data.get_angular_velocity_uncalibrated()[
        2]

    out["linear_acceleration_uncalibrated"] = [0, 0, 0]
    out["linear_acceleration_uncalibrated"][0] = imu_data.get_linear_acceleration_uncalibrated()[
        0]
    out["linear_acceleration_uncalibrated"][1] = imu_data.get_linear_acceleration_uncalibrated()[
        1]
    out["linear_acceleration_uncalibrated"][2] = imu_data.get_linear_acceleration_uncalibrated()[
        2]

    out["angular_velocity_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(3):
            out["angular_velocity_covariance"][i * 3 +j] = imu_data.get_angular_velocity_covariance().r[i][j]

    out["linear_acceleration_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(3):
            out["linear_acceleration_covariance"][i * 3 +
                                                  j] = imu_data.get_linear_acceleration_covariance().r[i][j]

    out["effective_rate"] = imu_data.effective_rate
    return out

def main():
    # ------------------------- OPTIONS ------------------------------------
    save_images = False
    save_depth = True
    save_pointcloud = False
    save_imu = True
    # ----------------------------------------------------------------------
    
    # Get input parameters
    svo_input_path = opt.input_svo_file
    output_dir = opt.output_path_dir

    if not os.path.isdir(output_dir):
        sys.stdout.write("Output directory doesn't exist. Check permission or create it.\n",
                         output_dir, "\n")
        exit()

    # ZED init
    zed = sl.Camera()
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.set_from_svo_file(svo_input_path)  # ← Set this path
    init.svo_real_time_mode = False # Don't convert in realtime
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD  # KITTI-like
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Better quality
    init.enable_right_side_measure = False


    # Open the SVO file 
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED initialization failed")
        exit(1)
    
    runtime = sl.RuntimeParameters()

    # Prepare output paths
    os.makedirs(output_dir, exist_ok=True)
    if save_images:
        os.makedirs(os.path.join(output_dir, "image_0"), exist_ok=True) # left images
        os.makedirs(os.path.join(output_dir, "image_1"), exist_ok=True) # right images
    if save_depth:
        os.makedirs(os.path.join(output_dir, "depths"), exist_ok=True)
    if save_pointcloud:
        os.makedirs(os.path.join(output_dir, "pointclouds"), exist_ok=True)
    if save_imu:
        imu_file = os.path.join(output_dir, "imu_data.txt")

    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    old_imu_timestamp = 0

    # Start SVO conversion
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")
    
    nb_frames = zed.get_svo_number_of_frames()
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            frame = zed.get_svo_position()
            # Retrieve images
            if save_images:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                
                left_img_file = output_dir + "/image_0/" + ("%s.png" % str(frame).zfill(6))
                right_img_file = output_dir + "/image_1/" + ("%s.png" % str(frame).zfill(6)) 
                
                # Save images
                cv2.imwrite(str(left_img_file),left_image.get_data())
                cv2.imwrite(str(right_img_file),right_image.get_data())
            
            # Retrieve depth
            if save_depth:
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # x = round(left_image.get_width() / 2)
                # y = round(left_image.get_height() / 2)
                depth_np = depth.get_data()
                # print(f"Distance to Camera at ({x}, {y}): {depth_value} m. Err {err}")
                # Save as .npy in meters
                np.save(os.path.join(output_dir, "depths", f"depth_map_{frame}.npy"), depth_np)

            # Retrieve point cloud
            if save_pointcloud:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                pc_np = point_cloud.get_data()
                pc_valid = pc_np[:, :, :3].reshape(-1, 3)
                pc_valid = pc_valid[~np.isnan(pc_valid).any(axis=1)]
                pc_path = os.path.join(output_dir, "pointclouds", f"{frame:06d}.xyz")
                np.savetxt(pc_path, pc_valid, fmt="%.3f")
            
            # Retrieve IMU Data
            sensor_data = sl.SensorsData()
            if(zed.get_sensors_data(sensor_data,sl.TIME_REFERENCE.CURRENT)):
                if(old_imu_timestamp != sensor_data.get_imu_data().timestamp):
                    old_imu_timestamp = sensor_data.get_imu_data().timestamp
                    sensor_data_serialized = IMUDataExtraction(sensor_data.get_imu_data())

                    # Save IMU Data
                    with open(imu_file, 'a') as file:
                        file.write(json.dumps(sensor_data_serialized) + "\n")
            
            # Display progress  
            progress_bar((frame + 1) / nb_frames * 100, 30)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100 , 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    zed.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    parser.add_argument('--output_path_dir', type = str, help = 'Path to a directory, where .png will be written, if mode includes image sequence export', default = '')
    opt = parser.parse_args()
    if not opt.input_svo_file.endswith(".svo") and not opt.input_svo_file.endswith(".svo2"): 
        print("--input_svo_file parameter should be a .svo file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    if not os.path.isfile(opt.input_svo_file):
        print("--input_svo_file parameter should be an existing file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    if len(opt.output_path_dir)==0 :
        print("In mode ",opt.mode,", output_path_dir parameter needs to be specified.")
        exit()
    if not os.path.isdir(opt.output_path_dir):
        print("--output_path_dir parameter should be an existing folder but is not : ",opt.output_path_dir,"Exit program.")
        exit()
    main()

