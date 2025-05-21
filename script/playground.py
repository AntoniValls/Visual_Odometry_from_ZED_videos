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

    image0_np = cv2.cvtColor(cv2.imread("../datasets/BIEL/00/image_0/000200.png"), cv2.COLOR_BGR2GRAY)
    image1_np = cv2.cvtColor(cv2.imread("../datasets/BIEL/00/image_0/000201.png"), cv2.COLOR_BGR2GRAY)

     # Preprocess for LoFTR
    image0 = preprocess_cv2_gray(image0_np)
    image1 = preprocess_cv2_gray(image1_np)

    matcher = KF.LoFTR(pretrained="outdoor")

    input_dict = {"image0": image0,
                "image1": image1
                }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    # Now let’s clean-up the correspondences with modern RANSAC and 
    # estimate fundamental matrix between two images

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0

    # Draw the matches using Kornia Moons
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            torch.from_numpy(mkpts0).view(1, -1, 2),
            torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts0.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            torch.from_numpy(mkpts1).view(1, -1, 2),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1),
        ),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(image0),
        K.tensor_to_image(image1),
        inliers,
        draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
    )



