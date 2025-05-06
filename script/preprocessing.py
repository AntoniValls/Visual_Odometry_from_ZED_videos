import cv2
import numpy as np
import matplotlib.pyplot as plt

def rectify_images(img_left, img_right, idx=0, map1=None, map2=None, plot=False):
    """
    Rectifies a pair of stereo images from a ZED camera using pre-calibrated camera parameters.
    
    This function performs stereo rectification which transforms the images so that corresponding points 
    are on the same horizontal scanline (epipolar lines become horizontal). This is essential for 
    stereo matching algorithms. The function uses hardcoded intrinsic and extrinsic parameters 
    specific to a ZED stereo camera in HD mode.

    Notes:
    ------
    - Uses hardcoded calibration parameters for ZED camera (HD mode) (in the SN...conf file)
    - Includes lens distortion correction and stereo rectification
    - Epipolar lines are drawn to visualize the rectification quality
    - Images are converted from BGR to RGB format for display purposes
    - The rectification process includes:
        1. Undistortion using camera intrinsic parameters
        2. Stereo rectification to align epipolar lines
        3. Image remapping to new rectified coordinates
        4. Obtaining the rectified left and right camera matrices (P1, P2)

    """

    # We only need to rectify the images once, so we can use the maps for subsequent calls
    if idx == 0:
        if img_left is None or img_right is None:
            raise ValueError("Check image paths: One or both images could not be loaded.")

        h, w = img_left.shape[:2]
        image_size = (w, h)

        # == Intrinsics HD ==
        # Camera matrices
        K1 = np.array([[706.391, 0, 631.506],
                    [0, 705.071, 387.479],
                    [0, 0, 1]])
        
        K2 = np.array([[709.177, 0, 631.227],
                    [0, 708.343, 381.786],
                    [0, 0, 1]])
        
        # Distortion vectors [k1, k2, p1, p2, k3]
        D1 = np.array([-0.164644, 0.012281, 0.007764, 0.000446, 0.000032])  # 5 terms only
        D2 = np.array([-0.166799, 0.012723, 0.008387, 0.000536, -0.000078])  # Again, only 5 terms
        
        # == Extrinsics (HD) ==
        # Rotation matrix
        RX = 0.00225504
        RY = 0.0
        RZ = -0.00325592
        rvec = np.array([RX, RY, RZ])
        R, _ = cv2.Rodrigues(rvec)
        
        # Translation vector
        T = np.array([[-0.0627311], # Baseline
                    [0.0776366],  # TY
                    [0.164198]])  # TZ
        R = np.eye(3)
        T = np.array([[-0.1], [0.0], [0.0]])  # 10cm baseline

        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # Init undistort rectify maps
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
        map1 = (map1x, map1y)
        map2 = (map2x, map2y)
    else:
        map1x, map1y = map1
        map2x, map2y = map2

    # Remap images
    rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

    # Plot
    if plot:
        # Convert BGR to RGB for matplotlib
        img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
        rectified_left_rgb = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB)
        rectified_right_rgb = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2RGB)

        # Draw Epipolar Lines
        def draw_epipolar_lines(image, color=(255, 0, 0), step=50):
            img_copy = image.copy()
            for y in range(0, image.shape[0], step):
                cv2.line(img_copy, (0, y), (image.shape[1], y), color, 1)
            return img_copy

        img_left_rgb = draw_epipolar_lines(img_left_rgb, color=(0, 255, 0))
        img_right_rgb = draw_epipolar_lines(img_right_rgb, color=(0, 255, 0))
        rect_left_epilines = draw_epipolar_lines(rectified_left_rgb, color=(0, 255, 0))
        rect_right_epilines = draw_epipolar_lines(rectified_right_rgb, color=(0, 255, 0))

        
        _, axs = plt.subplots(2, 2, figsize=(14, 8))
        axs[0, 0].imshow(img_left_rgb)
        axs[0, 0].set_title("Original Left")
        axs[0, 1].imshow(img_right_rgb)
        axs[0, 1].set_title("Original Right")
        axs[1, 0].imshow(rect_left_epilines)
        axs[1, 0].set_title("Rectified Left + Epipolar Lines")
        axs[1, 1].imshow(rect_right_epilines)
        axs[1, 1].set_title("Rectified Right + Epipolar Lines")

        for ax in axs.ravel():
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    if idx == 0:
        return rectified_left, rectified_right, P1, P2, map1, map2
    else:
        return rectified_left, rectified_right, map1, map2


