import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import decomposition

class StereoDepthEstimator:
    """
    Stereo Depth Estimation for low-texture images. 

    This class computes the disparity maps and then recreates the depths using intrinsic camera parameters.

    Supported algorithms:
    - "Simple": Basic SGBM Matcher without pre/post processing. 
    - "Complex": Preprocessing (enhance texture and contrast) + SGBM Matcher + WSL filter (noise reduction)
    """

    def __init__(self, config, P0, P1):
        self.rgb_value = config['parameters']['rgb']
        self.model = config['parameters']['depth_model']

        # Left camera instrinsic matrix
        K_left, *_ = decomposition(P0)
        self.focal_length = K_left[0,0]
        self.baseline = abs(P1[0,3]/ P1[0,0])

    def SGBM_based_disparity_map(self, left_image, right_image):
        '''
        Takes a stereo pair of images from the sequence and
        computes the SGBM-based disparity map (Simple or Complex).

        Args:
            left_image: image from left camera (Gray or BGR)
            right_image: image from right camera (Gray or BGR)
        
        Returns:
            disparity map
        '''

        if self.rgb_value:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            num_channels = 3
        else:
            left_gray = left_image
            right_gray = right_image
            num_channels = 1

        if self.model == "Complex":
            # --- Pre-processing: Enhance texture ---
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            left_gray = clahe.apply(left_gray)
            right_gray = clahe.apply(right_gray)

            # Bilateral filter to reduce noise but keep edges
            left_gray = cv2.bilateralFilter(left_gray, d=5, sigmaColor=75, sigmaSpace=75)
            right_gray = cv2.bilateralFilter(right_gray, d=5, sigmaColor=75, sigmaSpace=75)

        # --- SGBM Parameters ---
        min_disp = 0
        num_disp = 16 * 8  # Increase disparity range (must be divisible by 16)
        block_size = 5

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * num_channels * block_size ** 2,
            P2=32 * num_channels * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=1,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute left disparity
        left_disp = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

        if self.model == "Simple":
            return left_disp

        elif self.model == "Complex":
            # --- Post-processing: WLS Filter ---
            # Reduces noise while preserving edges
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            right_disp = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0

            # WLS filter
            lmbda = 8000
            sigma = 1.5
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
            filtered_disp = wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp)

            return filtered_disp

    def compute_depth(self, disparity_map):
        """
        Convert disparity to depth with sanity checks.
        """

        disparity_map[disparity_map <= 0] = 0.1  # Avoid division by zero
        depth_map = (self.focal_length * self.baseline) / disparity_map

        return depth_map


    def plot_depth_results(self, left_img, right_img=None, depth_map=None, disparity_map=None, title_suffix=""):
        """ 
        Visualize stereo inputs, disparity, and depth results.
        """

        cols = 2 if right_img is not None else 1
        rows = 2 if depth_map is not None or disparity_map is not None else 1
        fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))

        if rows == 1 and cols == 1:
            axs = [[axs]]
        elif rows == 1:
            axs = [axs]
        elif cols == 1:
            axs = [[ax] for ax in axs]

        axs[0][0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        axs[0][0].set_title(f"Left Image {title_suffix}")
        axs[0][0].axis("off")

        if right_img is not None:
            axs[0][1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
            axs[0][1].set_title(f"Right Image {title_suffix}")
            axs[0][1].axis("off")
    
        if disparity_map is not None:  
            disp_plot = axs[1][0].imshow(disparity_map, cmap="viridis")
            axs[1][0].set_title("Disparity Map")
            axs[1][0].axis("off")
            fig.colorbar(disp_plot, ax=axs[1][0])

        if depth_map is not None:
            vmin = np.percentile(depth_map, 5)
            vmax = np.percentile(depth_map, 95)
            depth_plot = axs[1][1 if right_img is not None else 0].imshow(depth_map, cmap="plasma", vmin=vmin, vmax=vmax)
            axs[1][1 if right_img is not None else 0].set_title("Depth Map")
            axs[1][1 if right_img is not None else 0].axis("off")
            fig.colorbar(depth_plot, ax=axs[1][1 if right_img is not None else 0])

        plt.tight_layout()
        plt.show()

        return

    def estimate_depth(self, left_image, right_image, plot=False):
        '''
        Ultimate function to predict depth maps from stereo vision.

        Args:
            left_image: Left camera image
            right_image: Right camera image
            plot: Boolean about if to plot the depth maps

        Returns:
            Tuple of (depth_map, disparity_map or None)
        '''

        # Compute the disparity map
        disparity_map = self.SGBM_based_disparity_map(left_image, right_image)
        
        # Compute the depth map
        depth_map = self.compute_depth(disparity_map)

        # Plot the depth map
        if plot:
            self.plot_depth_results(left_image, right_image, depth_map, disparity_map, title_suffix=str(self.model))

        return depth_map, disparity_map
