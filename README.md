# Visual Odometry from Home-Recorded Videos using a Stereolabs ZED Mini Camera

This project focuses on visual odometry (VO) from stereo video sequences recorded at home using a **Stereolabs ZED Mini** camera. It implements and compares multiple techniques across the VO pipeline—from depth estimation to motion tracking and evaluation.

---

## Project Components

### 1. Stereo Depth Estimation (SDE)
- **SGBM (Semi-Global Block Matching)**: A classical stereo matching method.
- **HITNet** (Tankovich et al.): A state-of-the-art deep learning model for real-time stereo matching.

### 2. Feature Extraction and Matching
- **Classical Approach**
  - Harris corner detector + SIFT descriptor
  - Brute-Force matcher
- **State-of-the-Art Approaches**
  - **LoFTR**: Dense matching using transformers.
  - **LightGlue + SuperPoint**: Combines learned keypoints and match refinement.

### 3. Motion Estimation / PnP Solving
Solvers used for camera pose estimation:
- **RANSAC**
- **LoRANSAC**
- **PROSAC**
- **MAGSAC++**

---

## Additional Features

- **Visual-Inertial Odometry (VIO) Extension**: Integrates inertial data when available.
- **Error Statistics & Evaluation Tools**: Assess the accuracy and robustness of each VO pipeline.
- **Exploratory Data Analysis (EDA)**: Understand and visualize VO behavior.
- **ZED VIO Benchmarking**: Compare custom pipelines with ZED’s built-in VIO output.

---

![Visual Odometry Pipeline Example](https://github.com/user-attachments/assets/33d3247d-fc53-4f7b-925c-9df6ef463dcf)

