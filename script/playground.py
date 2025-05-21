import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches

def preprocess_cv2_gray(img: np.ndarray) -> torch.Tensor:
    """Convert a grayscale OpenCV image to a normalized torch tensor for Kornia."""
    img_t = torch.from_numpy(img).float() / 255.0  # [H, W], float32, [0,1]
    return img_t[None, None]  # Add batch and channel dimensions -> [1, 1, H, W]

if __name__ == "__main__":

    # Load image in color and convert to float32 in [0, 1]
    img_0_path = "../datasets/BIEL/00/image_0/000200.png"
    img_1_path = "../datasets/BIEL/00/image_1/000200.png"
    
    image_left = K.io.load_image(img_0_path, K.io.ImageLoadType.RGB32)[None, ...]
    next_image = K.io.load_image(img_1_path, K.io.ImageLoadType.RGB32)[None, ...]

    input_dict = {"image0": K.color.rgb_to_grayscale(image_left), # LoFTR only works in grayscale
                    "image1": K.color.rgb_to_grayscale(next_image)
                    }

    matcher = KF.LoFTR(pretrained="outdoor")

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    # Now let’s clean-up the correspondences with modern RANSAC and 
    # estimate fundamental matrix between two images

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    



