import cv2
import numpy as np
import matplotlib.pyplot as plt

def rectify_images(img_left, img_right):
    
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
    
    # Distortion vectors [k1, k2, p1, p2, k3, k4, k5, k6,]
    D1 = np.array([11.67746437638404, -4.847547466048848,
                   0.0004966451731231306, -9.039033628208174e-05,
                   -0.19014947028934145, 11.765739618583925,
                   -2.679278099599428, -1.1749281418984923])
    
    D2 = np.array([3.846453489420972, 17.190073727386558,
                   0.0004947103549008237, -3.341436041060913e-05,
                   1.489075549979669, 4.013565142010989,
                   17.761309005590853, 4.747755641309874])
    
    # == Extrinsics ==
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

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # Trying (NOT SURE)
    R1 = R1.dot(np.linalg.inv(K1))
    R2 = R2.dot(np.linalg.inv(K2))

    # Init undistort rectify maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    # Remap images
    rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Convert BGR to RGB for matplotlib
    # img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    # img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    # rectified_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB)
    # rectified_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2RGB)

    # Draw Epipolar Lines
    def draw_epipolar_lines(image, color=(255, 0, 0), step=50):
        img_copy = image.copy()
        for y in range(0, image.shape[0], step):
            cv2.line(img_copy, (0, y), (image.shape[1], y), color, 1)
        return img_copy

    img_left = draw_epipolar_lines(img_left, color=(0, 255, 0))
    img_right = draw_epipolar_lines(img_right, color=(0, 255, 0))
    rect_left_epilines = draw_epipolar_lines(rectified_left, color=(0, 255, 0))
    rect_right_epilines = draw_epipolar_lines(rectified_right, color=(0, 255, 0))

    # Plot
    _, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs[0, 0].imshow(img_left)
    axs[0, 0].set_title("Original Left")
    axs[0, 1].imshow(img_right)
    axs[0, 1].set_title("Original Right")
    axs[1, 0].imshow(rect_left_epilines)
    axs[1, 0].set_title("Rectified Left + Epipolar Lines")
    axs[1, 1].imshow(rect_right_epilines)
    axs[1, 1].set_title("Rectified Right + Epipolar Lines")

    for ax in axs.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return rectified_left, rectified_right


