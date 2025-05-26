import os
import numpy as np
import cv2
from natsort import natsorted

def create_video_from_depth_maps(input_folder, output_path, fps=24, use_colormap=True):
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    files = natsorted(files)

    if not files:
        raise ValueError("No .npy files found in the input folder.")

    # Load first frame to get dimensions
    first_depth = np.load(os.path.join(input_folder, files[0]))
    height, width = first_depth.shape

    # Define whether the output will be color or grayscale
    is_color = use_colormap

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)

    for fname in files:
        depth_map = np.load(os.path.join(input_folder, fname))

        # Normalize to 0–255
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        if use_colormap:
            frame = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        else:
            frame = depth_uint8  # single-channel image

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == '__main__': 
    depth_model = "ZED"
    if depth_model != "ZED":
        input_folder = f"../datasets/predicted/depth_maps/00/{depth_model}"
    else:
        input_folder = f"../datasets/BIEL/00/depths/"
    save_dir = "../datasets/predicted/calaix_de_sastre/"
    output_path = os.path.join(save_dir, f"00_{depth_model}.mp4")
    create_video_from_depth_maps(input_folder, output_path, fps=24, use_colormap=True)