import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import matplotlib.pyplot as plt
import torch
import os, sys

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
            self.matcher =KF.LoFTR(pretrained="outdoor")

        else:
            raise ValueError("Detector must be 'lightglue' or 'LoFTR'")

            


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
            keypoint_left_first, keypoint_left_next, filtered_matches

        """

        cache_path = os.path.join(self.cache_dir, f"matches_{idx}.npz")

        # Load precomputed matches from cache
        if os.path.exists(cache_path):
            if idx == 0:
                print(f"Loading cached matches from {self.cache_dir}")
            data = np.load(cache_path, allow_pickle=True)
            keypoint_left_first = data["keypoint_left_first"]
            keypoint_left_next = data["keypoint_left_next"]
            descriptor_left_first = data["descriptor_left_first"]
            descriptor_left_next = data["descriptor_left_next"]
            matches = data["matches"]
            scores = data["scores"] if "scores" in data.files else None

            # Apply threshold filtering
            topk = np.argsort(-scores)[:self.threshold]
            filtered_matches = matches[topk]
            keypoint_left_first = keypoint_left_first[filtered_matches[:, 0]]
            keypoint_left_next = keypoint_left_next[filtered_matches[:, 1]]

        # Compute and save
        else:
            if idx == 0:
                print(f"Computing matches and saving to {self.cache_dir}")
            
            # Special loadings
            sequence_dir = os.path.join(self.config['main_path'], self.config['type']) + "/"
            img_0_path = sequence_dir + 'image_0/' + sorted(os.listdir(sequence_dir + 'image_0'))[idx]
            img_1_path = sequence_dir + 'image_0/' + sorted(os.listdir(sequence_dir + 'image_0'))[idx + 1]
            
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
                        descriptor_left_first=descriptor_left_first,
                        descriptor_left_next=descriptor_left_next,
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
                
        # Plot matches every 1000 frames
        if not plot and idx % 1000 == 0:
            save_dir = f"../datasets/predicted/matches/{self.data_name}/{self.detector}_{self.threshold}"
            os.makedirs(save_dir, exist_ok=True)
            _ = viz2d.plot_images([image_left, next_image])
            viz2d.plot_matches(keypoint_left_first, keypoint_left_next, color="lime", lw=0.2)
            # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
            plt.title(f"Matches using LightGlue. Frames {idx} and {idx+1}")
            if show:
                plt.show()
            viz2d.save_plot(os.path.join(save_dir, f"matches_{idx}.png"))
            plt.close()

        return keypoint_left_first, keypoint_left_next, filtered_matches
        