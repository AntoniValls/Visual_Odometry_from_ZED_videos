import os
import sys
import cv2
import json
import numpy as np
import pyzed.sl as sl
import shutil

def progress_bar(percent_done, bar_length=50):
    #Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def main(svo_input_path,
        output_dir,
        save_images=False,
        save_depth=False,
        save_pointcloud=False,
        save_imu=False):
    """
    Processes SVO file and optionally saves images, depth, pointclouds, and IMU data.

    Parameters:
        svo_input_path (str): Path to the SVO file.
        output_dir (str): Directory to save outputs.
        save_images (bool): Save RGB images if True.
        save_depth (bool): Save depth maps if True.
        save_pointcloud (bool): Save pointclouds if True.
        save_imu (bool): Save IMU and pose data if True.
    """

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
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # Right-handed, y-down
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Better quality
    init.enable_right_side_measure = False

    # Open the SVO file 
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED initialization failed")
        exit(1)
    
    runtime = sl.RuntimeParameters()

    # Prepare output directory
    # Rewrite the output file if it exists
    os.makedirs(output_dir, exist_ok=True)
    if save_images:
        for subfolder in ["image_0", "image_1"]:
            folder_path = os.path.join(output_dir, subfolder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    if save_depth:
        depth_path = os.path.join(output_dir, "depths")
        if os.path.exists(depth_path):
            shutil.rmtree(depth_path)
            os.makedirs(depth_path)
    if save_pointcloud:
        pc_path = os.path.join(output_dir, "pointclouds")
        if os.path.exists(pc_path):
            shutil.rmtree(pc_path)
        os.makedirs(pc_path)
    if save_imu:
        imu_file = os.path.join(output_dir, "imu_data.txt")
        if os.path.exists(imu_file):
            os.remove(imu_file)

    # Prepare single image containers
    if save_images:
        left_image = sl.Mat()
        right_image = sl.Mat()

    if save_depth:
        depth = sl.Mat()
  
    if save_pointcloud:
        point_cloud = sl.Mat()
  
    if save_imu:
        zed_pose = sl.Pose() # Visual-Inertial SLAM pose
        zed_sensors = sl.SensorsData() # Accelerometer and Gyroscope data
        py_transform = sl.Transform()  # Transform object for TrackingParameters object
        tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
        err = zed.enable_positional_tracking(tracking_parameters)
        if (err != sl.ERROR_CODE.SUCCESS):
            exit(-1)
    
    old_imu_timestamp = 0
    nb_frames = zed.get_svo_number_of_frames()

    # Start SVO conversion
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

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
                # x = left_image.get_width() / 2)
                # y = left_image.get_height() / 2)
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
            if save_imu:
                                
                # Retrieve the pose of the camera from the Visual-Inertial SLAM System
                zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)

                # Pose data
                translation = zed_pose.get_translation()
                orientation = zed_pose.get_orientation()

                # Retrieve the IMU data (accel + gyro)
                if(zed.get_sensors_data(zed_sensors,sl.TIME_REFERENCE.IMAGE)):
                    if(old_imu_timestamp != zed_sensors.get_imu_data().timestamp):
                        old_imu_timestamp = zed_sensors.get_imu_data().timestamp

                        # IMU data
                        imu_data = zed_sensors.get_imu_data()
                        linear_acc = imu_data.get_linear_acceleration()
                        angular_vel = imu_data.get_angular_velocity()
                        timestamp_ns = imu_data.timestamp.get_nanoseconds()

                        # Save IMU data in a text file
                        out = {}
                        out["is_available"] = imu_data.is_available
                        out["timestamp"] = timestamp_ns
                        out["pose"] = {}
                        out["pose"]["translation"] = [translation.get()[0], translation.get()[1], translation.get()[2]]
                        out["pose"]["quaternion"] = [orientation.get()[0], orientation.get()[1], orientation.get()[2], orientation.get()[3]]
                        out["pose_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0] # Covariance matrix of the pose (3x3)
                        for i in range(3):
                            for j in range(3):
                                out["pose_covariance"][i * 3 + j] = imu_data.get_pose_covariance().r[i][j]
                        out["angular_velocity"] = [angular_vel[0], angular_vel[1], angular_vel[2]]
                        out["linear_acceleration"] = [linear_acc[0], linear_acc[1], linear_acc[2]]
                        out["angular_velocity_uncalibrated"] = [imu_data.get_angular_velocity_uncalibrated()[0],
                                                                imu_data.get_angular_velocity_uncalibrated()[1],
                                                                imu_data.get_angular_velocity_uncalibrated()[2]]
                        out["linear_acceleration_uncalibrated"] = [imu_data.get_linear_acceleration_uncalibrated()[0],
                                                                    imu_data.get_linear_acceleration_uncalibrated()[1],
                                                                    imu_data.get_linear_acceleration_uncalibrated()[2]]
                        out["angular_velocity_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0] # Covariance matrix of the angular velocity (3x3)
                        for i in range(3):
                            for j in range(3):
                                out["angular_velocity_covariance"][i * 3 + j] = imu_data.get_angular_velocity_covariance().r[i][j]
                        out["linear_acceleration_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0] # Covariance matrix of the linear acceleration (3x3)
                        for i in range(3):
                            for j in range(3):
                                out["linear_acceleration_covariance"][i * 3 + j] = imu_data.get_linear_acceleration_covariance().r[i][j]
                        out["effective_rate"] = imu_data.effective_rate  # Effective rate of the IMU in Hz
                        
                        # Save IMU Data
                        with open(imu_file, 'a') as file:
                            file.write(json.dumps(out) + "\n")
            
            # Display progress  
            progress_bar((frame + 1) / nb_frames * 100, 30)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100 , 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    zed.close()
    return 0

if __name__ == "__main__":
    seqs = [str(i).zfill(2) for i in range(23)]

    for seq in seqs:
        print(f"Processing sequence {seq}...")
        input_svo_path = f"../datasets/BIEL/svofiles/IRI_{seq}.svo2"
        output_directory = f"../datasets/BIEL/{seq}/"  

        main(input_svo_path, 
            output_directory,
            save_images=True,
            save_depth=True,
            save_pointcloud=False,
            save_imu=True)
    

