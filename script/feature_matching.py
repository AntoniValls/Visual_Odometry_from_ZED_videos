import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import matplotlib.pyplot as plt
import torch
import os, sys
from feature_visualizer import load_matches

current_dir = os.path.dirname(__file__)
lightglue_path = os.path.abspath(os.path.join(current_dir, '..', 'LightGlue'))
sys.path.append(lightglue_path)
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

class FeatureMatcher:
    """
    A class for feature extraction and matching between sequential images using deep learning-based
    LightGlue with SuperPoint detector.

    Also accepts loading precomputed matches from cache.
    """

    def __init__(self, config):

        self.config = config
        self.data_name = config['data']['type']
        self.detector = config['parameters']['detector']
        self.threshold = config['parameters']['threshold']
        self.cache_dir = f"../datasets/predicted/prefiltered_matches/{self.data_name}/{self.detector}/"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the models
        if self.detector == "lightglue":
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        elif self.detector == "LoFTR":
            self.extractor = None
            self.matcher = KF.LoFTR(pretrained="outdoor")
        
        elif self.detector == "Harris-SIFT":
            self.extractor = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        else:
            raise ValueError("Detector must be 'lightglue', 'LoFTR' or 'Harris-SIFT")

    def compute(self, image_left, next_image, idx, plot=False, show=False):
        """
        Compute feature extraction and matching of sequential images or load
        precomputed ones. Also plot.

        Args:
            image_left: first frame
            next_image: second frame
            idx: index of the first frame
            plot: boolean about if to plot every 1000 frames
            show: boolean about if to show the plots.

        Return:
            keypoint_left_first, keypoint_left_next

        """

        cache_path = os.path.join(self.cache_dir, f"matches_{idx}.npz")
        # Special loadings
        sequence_dir = os.path.join(self.config['data']['main_path'], self.config['data']['type']) + "/"
        img_0_path = sequence_dir + 'image_0/' + sorted(os.listdir(sequence_dir + 'image_0'))[idx]
        img_1_path = sequence_dir + 'image_0/' + sorted(os.listdir(sequence_dir + 'image_0'))[idx + 1]

        # Load precomputed matches from cache
        if os.path.exists(cache_path):
            if idx == 0:
                print(f"Loading cached matches from {self.cache_dir}")
            data = np.load(cache_path, allow_pickle=True)
            keypoint_left_first = data["keypoint_left_first"]
            keypoint_left_next = data["keypoint_left_next"]
            matches = data["matches"] if "matches" in data.files else None
            scores = data["scores"] if "scores" in data.files else None

            mask=np.argsort(-scores)[:1000]
            if self.detector == "lightglue":
                #mask = scores > 0.990
                filtered_matches = matches[mask]
                keypoint_left_first= keypoint_left_first[filtered_matches[:, 0]]
                keypoint_left_next = keypoint_left_next[filtered_matches[:, 1]]
            elif self.detector == "LoFTR":
                #mask = scores > 0.990
                keypoint_left_first = keypoint_left_first[mask]
                keypoint_left_next = keypoint_left_next[mask]
            elif self.detector == "Harris-SIFT":
                threshold = 1000
                keypoint_left_first = keypoint_left_first[:threshold]
                keypoint_left_next = keypoint_left_next[:threshold]
        
        # Compute and save (THE THRESHOLD NEEDS TO BE CORRECTED)
        else:
            if idx == 0:
                print(f"Computing matches and saving to {self.cache_dir}")
            
            # LightGlue feature matching
            if self.detector == "lightglue":
                image_left = load_image(img_0_path)
                next_image = load_image(img_1_path)

                descriptor_left_first = self.extractor.extract(image_left.to(self.device))
                descriptor_left_next = self.extractor.extract(next_image.to(self.device))
                matches01 = self.matcher({
                "image0": descriptor_left_first, 
                "image1": descriptor_left_next
                })

                # Remove batch dimension
                descriptor_left_first, descriptor_left_next, matches01 = [
                    rbd(x) for x in [descriptor_left_first, descriptor_left_next, matches01]
                ]  
                
                # Convert to numpy arrays
                keypoint_left_first = descriptor_left_first["keypoints"].cpu().numpy()
                keypoint_left_next = descriptor_left_next["keypoints"].cpu().numpy()
                matches = matches01["matches"].cpu().numpy()
                scores = matches01["scores"].cpu().detach().numpy()

                # Save raw data
                np.savez(cache_path,
                        keypoint_left_first=keypoint_left_first,
                        keypoint_left_next=keypoint_left_next,
                        matches=matches,
                        scores=scores)

                # Apply top-k filtering
                topk = np.argsort(-scores)[:self.threshold]
                filtered_matches = matches[topk]
                keypoint_left_first = keypoint_left_first[filtered_matches[:, 0]]
                keypoint_left_next = keypoint_left_next[filtered_matches[:, 1]]

            # LoFTR feature matching (using Kornia)
            elif self.detector == "LoFTR":
                
                # Load image in color and convert to float32 in [0, 1]
                image_left = K.io.load_image(img_0_path, K.io.ImageLoadType.RGB32)[None, ...]
                next_image = K.io.load_image(img_1_path, K.io.ImageLoadType.RGB32)[None, ...]

                input_dict = {"image0": K.color.rgb_to_grayscale(image_left), # LoFTR only works in grayscale
                              "image1": K.color.rgb_to_grayscale(next_image)
                              }
                
                with torch.inference_mode():        
                    correspondences = self.matcher(input_dict)

                # Convert to numpy arrays
                keypoint_left_first = correspondences["keypoints0"].cpu().numpy()
                keypoint_left_next = correspondences["keypoints1"].cpu().numpy()
                scores = correspondences["confidence"].cpu().numpy()

                # Save the raw data
                np.savez(cache_path,
                        keypoint_left_first=keypoint_left_first,
                        keypoint_left_next=keypoint_left_next,
                        scores=scores)
                
                # Apply top-k filtering
                topk = np.argsort(-scores)[:self.threshold]
                keypoint_left_first = keypoint_left_first[topk]
                keypoint_left_next = keypoint_left_next[topk]
        
            elif self.detector == "Harris-SIFT":
        
                # Load grayscale images
                img1 = cv2.imread(img_0_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img_1_path, cv2.IMREAD_GRAYSCALE)

                # Harris Corner Detection
                def harris_keypoints(img, block_size=2, ksize=3, k=0.04, threshold=0.01):
                    harris = cv2.cornerHarris(img, block_size, ksize, k)
                    harris = cv2.dilate(harris, None)
                    keypoints = np.argwhere(harris > threshold * harris.max())
                    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]
                    return keypoints

                kp1 = harris_keypoints(img1)
                kp2 = harris_keypoints(img2)

                # Extract SIFT descriptors
                kp1, des1 = self.extractor.compute(img1, kp1)
                kp2, des2 = self.extractor.compute(img2, kp2)

                # Match descriptors using BFMatcher with default L2 norm
                all_matches = self.matcher.match(des1, des2)
                all_matches = sorted(all_matches, key=lambda x: x.distance)

                # Save all keypoints and matches
                keypoint_left_first = np.array([kp1[m.queryIdx].pt for m in all_matches])
                keypoint_left_next = np.array([kp2[m.trainIdx].pt for m in all_matches])
                scores = np.array([m.distance for m in all_matches])

                # Save all matches and scores for later filtering
                np.savez(cache_path,
                        keypoint_left_first=keypoint_left_first,
                        keypoint_left_next=keypoint_left_next,
                        scores=scores)

                # Apply threshold filtering (already sortered by distance)
                # lower distance = better match
                keypoint_left_first = keypoint_left_first[:self.threshold]
                keypoint_left_next = keypoint_left_next[:self.threshold]
        
        # Plot matches every 1000 frames
        if not plot and idx % 1000 == 0:

            image_left = load_image(img_0_path)
            next_image = load_image(img_1_path)

            save_dir = f"../datasets/predicted/matches/{self.data_name}/{self.detector}_{self.threshold}"
            os.makedirs(save_dir, exist_ok=True)

            _ = viz2d.plot_images([image_left, next_image])
            viz2d.plot_matches(keypoint_left_first, keypoint_left_next, color="lime", lw=0.2)
            # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
            plt.title(f"Matches using {self.detector}. Frames {idx} and {idx+1}")
            if show:
                plt.show()
            viz2d.save_plot(os.path.join(save_dir, f"matches_{idx}.png"))
            plt.close()

        return keypoint_left_first, keypoint_left_next
        