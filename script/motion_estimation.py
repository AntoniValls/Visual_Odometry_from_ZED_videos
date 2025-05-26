import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional

class AdvancedRANSAC:
    """Advanced RANSAC implementations for robust pose estimation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.max_iterations = config.get('max_iterations', 1000)
        self.confidence = config.get('confidence', 0.99)
        self.threshold = config.get('threshold', 5.0)  # pixels
        self.min_inliers = config.get('min_inliers', 15)
        
    def magsac_plus_plus(self, points_3d: np.ndarray, points_2d: np.ndarray, 
                        intrinsic_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MAGSAC++ implementation for robust pose estimation
        Combines marginalization and spatial coherence
        """
        best_model = None
        best_score = -np.inf
        best_inliers = None
        
        # Precompute neighborhood graph for spatial coherence
        neighborhoods = self._compute_neighborhoods(points_2d)
        
        # Adaptive threshold based on noise estimation
        sigma = self._estimate_noise(points_3d, points_2d, intrinsic_matrix)
        adaptive_threshold = max(self.threshold, 3 * sigma)
        
        iterations = 0
        while iterations < self.max_iterations:
            # Sample minimal set (6 points for PnP)
            sample_indices = self._guided_sampling(points_2d, neighborhoods, 6)
            
            try:
                # Estimate pose from minimal set
                sample_3d = points_3d[sample_indices]
                sample_2d = points_2d[sample_indices]
                
                success, rvec, tvec = cv2.solvePnP(sample_3d, sample_2d, intrinsic_matrix, None)
                if not success:
                    iterations += 1
                    continue
                
                # Project all 3D points
                projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, None)
                projected_points = projected_points.reshape(-1, 2)
                
                # Compute residuals
                residuals = np.linalg.norm(points_2d - projected_points, axis=1)
                
                # MAGSAC++ scoring with marginalization
                score, inliers = self._magsac_scoring(residuals, adaptive_threshold, neighborhoods)
                
                if score > best_score and len(inliers) >= self.min_inliers:
                    best_score = score
                    best_model = (rvec.copy(), tvec.copy())
                    best_inliers = inliers.copy()
                    
                    # Early termination check
                    if self._early_termination_check(len(inliers), len(points_3d)):
                        break
                
            except cv2.error:
                pass
            
            iterations += 1
        
        if best_model is None:
            raise ValueError("MAGSAC++ failed to find a valid model")
        
        # Refine with all inliers
        refined_model = self._refine_pose(points_3d[best_inliers], points_2d[best_inliers], 
                                        intrinsic_matrix, best_model)
        
        return refined_model[0], refined_model[1], best_inliers
    
    def lo_ransac(self, points_3d: np.ndarray, points_2d: np.ndarray, 
                  intrinsic_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LO-RANSAC (Locally Optimized RANSAC) for iterative refinement
        """
        best_model = None
        best_inliers = None
        best_inlier_count = 0
        
        for iteration in range(self.max_iterations):
            # Sample minimal set
            sample_indices = np.random.choice(len(points_3d), 6, replace=False)
            
            try:
                sample_3d = points_3d[sample_indices]
                sample_2d = points_2d[sample_indices]
                
                success, rvec, tvec = cv2.solvePnP(sample_3d, sample_2d, intrinsic_matrix, None)
                if not success:
                    continue
                
                # Find inliers
                projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, None)
                projected_points = projected_points.reshape(-1, 2)
                residuals = np.linalg.norm(points_2d - projected_points, axis=1)
                inliers = np.where(residuals < self.threshold)[0]
                
                if len(inliers) < self.min_inliers:
                    continue
                
                # Local optimization: iteratively add close points and re-estimate
                current_inliers = inliers.copy()
                for _ in range(3):  # Maximum 3 LO iterations
                    # Re-estimate with current inliers
                    success, rvec, tvec = cv2.solvePnP(points_3d[current_inliers], 
                                                     points_2d[current_inliers], 
                                                     intrinsic_matrix, None)
                    if not success:
                        break
                    
                    # Find new inliers
                    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, None)
                    projected_points = projected_points.reshape(-1, 2)
                    residuals = np.linalg.norm(points_2d - projected_points, axis=1)
                    new_inliers = np.where(residuals < self.threshold)[0]
                    
                    if len(new_inliers) <= len(current_inliers):
                        break
                    current_inliers = new_inliers
                
                if len(current_inliers) > best_inlier_count:
                    best_inlier_count = len(current_inliers)
                    best_model = (rvec.copy(), tvec.copy())
                    best_inliers = current_inliers.copy()
                
            except cv2.error:
                continue
        
        if best_model is None:
            raise ValueError("LO-RANSAC failed to find a valid model")
        
        return best_model[0], best_model[1], best_inliers
    
    def prosac(self, points_3d: np.ndarray, points_2d: np.ndarray, 
               intrinsic_matrix: np.ndarray, quality_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        PROSAC (Progressive Sample Consensus) - quality-guided sampling
        """
        # Sort points by quality (descending)
        sorted_indices = np.argsort(-quality_scores)
        points_3d_sorted = points_3d[sorted_indices]
        points_2d_sorted = points_2d[sorted_indices]
        
        best_model = None
        best_inliers = None
        best_inlier_count = 0
        
        # Progressive sampling
        for n in range(6, len(points_3d_sorted) + 1):
            current_points_3d = points_3d_sorted[:n]
            current_points_2d = points_2d_sorted[:n]
            
            # Number of iterations for this subset size
            iterations_for_n = min(50, self.max_iterations // (len(points_3d_sorted) - 5))
            
            for _ in range(iterations_for_n):
                # Sample from top n points
                sample_indices = np.random.choice(n, 6, replace=False)
                
                try:
                    sample_3d = current_points_3d[sample_indices]
                    sample_2d = current_points_2d[sample_indices]
                    
                    success, rvec, tvec = cv2.solvePnP(sample_3d, sample_2d, intrinsic_matrix, None)
                    if not success:
                        continue
                    
                    # Evaluate on all points
                    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, None)
                    projected_points = projected_points.reshape(-1, 2)
                    residuals = np.linalg.norm(points_2d - projected_points, axis=1)
                    inliers = np.where(residuals < self.threshold)[0]
                    
                    if len(inliers) > best_inlier_count and len(inliers) >= self.min_inliers:
                        best_inlier_count = len(inliers)
                        best_model = (rvec.copy(), tvec.copy())
                        best_inliers = inliers.copy()
                
                except cv2.error:
                    continue
        
        if best_model is None:
            raise ValueError("PROSAC failed to find a valid model")
        
        return best_model[0], best_model[1], best_inliers
    
    def _compute_neighborhoods(self, points_2d: np.ndarray, radius: float = 50.0) -> List[np.ndarray]:
        """Compute spatial neighborhoods for each point"""
        distances = cdist(points_2d, points_2d)
        neighborhoods = []
        for i in range(len(points_2d)):
            neighbors = np.where(distances[i] < radius)[0]
            neighborhoods.append(neighbors)
        return neighborhoods
    
    def _guided_sampling(self, points_2d: np.ndarray, neighborhoods: List[np.ndarray], 
                        num_samples: int) -> np.ndarray:
        """Guided sampling considering spatial distribution"""
        if len(points_2d) <= num_samples:
            return np.arange(len(points_2d))
        
        # Start with random point
        selected = [random.randint(0, len(points_2d) - 1)]
        
        # Add points that are spatially diverse
        for _ in range(num_samples - 1):
            best_candidate = -1
            best_min_distance = -1
            
            # Try several candidates
            for _ in range(10):
                candidate = random.randint(0, len(points_2d) - 1)
                if candidate in selected:
                    continue
                
                # Compute minimum distance to selected points
                min_dist = min([np.linalg.norm(points_2d[candidate] - points_2d[s]) 
                               for s in selected])
                
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_candidate = candidate
            
            if best_candidate != -1:
                selected.append(best_candidate)
            else:
                # Fallback to random selection
                remaining = [i for i in range(len(points_2d)) if i not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
        
        return np.array(selected)
    
    def _estimate_noise(self, points_3d: np.ndarray, points_2d: np.ndarray, 
                       intrinsic_matrix: np.ndarray) -> float:
        """Estimate noise level in the data"""
        # Use median of nearest neighbor distances
        distances = cdist(points_2d, points_2d)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        return np.median(min_distances) * 0.1
    
    def _magsac_scoring(self, residuals: np.ndarray, threshold: float, 
                       neighborhoods: List[np.ndarray]) -> Tuple[float, np.ndarray]:
        """MAGSAC++ scoring with spatial coherence"""
        # Compute weights based on residuals (sigmoid function)
        weights = 1.0 / (1.0 + np.exp((residuals - threshold) / (threshold * 0.1)))
        
        # Add spatial coherence term
        spatial_weights = np.ones_like(weights)
        for i, neighbors in enumerate(neighborhoods):
            if len(neighbors) > 1:
                neighbor_weights = weights[neighbors]
                spatial_weights[i] = np.mean(neighbor_weights)
        
        # Combine photometric and spatial evidence
        final_weights = 0.7 * weights + 0.3 * spatial_weights
        
        # Score is sum of weights
        score = np.sum(final_weights)
        
        # Inliers are points with high final weights
        inliers = np.where(final_weights > 0.5)[0]
        
        return score, inliers
    
    def _early_termination_check(self, num_inliers: int, total_points: int) -> bool:
        """Check if we can terminate early"""
        inlier_ratio = num_inliers / total_points
        return inlier_ratio > 0.8  # If 80% are inliers, we probably found the right model
    
    def _refine_pose(self, points_3d: np.ndarray, points_2d: np.ndarray, 
                    intrinsic_matrix: np.ndarray, initial_pose: Tuple) -> Tuple:
        """Refine pose using all inliers with iterative optimization"""
        rvec_init, tvec_init = initial_pose
        
        # Use OpenCV's iterative refinement
        success, rvec_refined, tvec_refined = cv2.solvePnP(
            points_3d, points_2d, intrinsic_matrix, None, 
            rvec_init, tvec_init, useExtrinsicGuess=True, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return rvec_refined, tvec_refined
        else:
            return rvec_init, tvec_init


def motion_estimation(firstImage_keypoints, secondImage_keypoints, 
                             intrinsic_matrix, config, depth):
    """
    Improved motion estimation with advanced RANSAC algorithms
    """
    max_depth = config['parameters']['max_depth']
    ransac_method = config['parameters'].get('ransac_method', 'magsac++')
    
    image1_points = np.float32(firstImage_keypoints)
    image2_points = np.float32(secondImage_keypoints)
    
    # Extract camera parameters
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    
    points_3D = []
    valid_indices = []
    
    # Build 3D points with better depth handling
    for idx, (u, v) in enumerate(image1_points):
        # Use bilinear interpolation for sub-pixel depth
        u_int, v_int = int(u), int(v)
        if (0 <= u_int < depth.shape[1]-1 and 0 <= v_int < depth.shape[0]-1):
            # Bilinear interpolation
            u_frac = u - u_int
            v_frac = v - v_int
            
            z = (depth[v_int, u_int] * (1-u_frac) * (1-v_frac) +
                 depth[v_int, u_int+1] * u_frac * (1-v_frac) +
                 depth[v_int+1, u_int] * (1-u_frac) * v_frac +
                 depth[v_int+1, u_int+1] * u_frac * v_frac)
        else:
            z = depth[int(v), int(u)] if (0 <= int(u) < depth.shape[1] and 
                                        0 <= int(v) < depth.shape[0]) else 0
        
        # Filter by depth
        if 0.1 < z < max_depth:  # Avoid points too close to camera
            x = z * (u - cx) / fx
            y = z * (v - cy) / fy
            points_3D.append([x, y, z])
            valid_indices.append(idx)
    
    if len(points_3D) < 6:
        raise ValueError("Insufficient valid 3D points for pose estimation")
    
    points_3D = np.array(points_3D)
    valid_indices = np.array(valid_indices)
    image1_points = image1_points[valid_indices]
    image2_points = image2_points[valid_indices]
    
    # Configure RANSAC
    ransac_config = {
        'max_iterations': config['parameters'].get('ransac_iterations', 1000),
        'confidence': config['parameters'].get('ransac_confidence', 0.99),
        'threshold': config['parameters'].get('ransac_threshold', 3.0),
        'min_inliers': config['parameters'].get('min_inliers', 15)
    }
    
    ransac_solver = AdvancedRANSAC(ransac_config)
    
    # Choose RANSAC method
    if ransac_method == 'magsac++':
        rvec, tvec, inliers = ransac_solver.magsac_plus_plus(
            points_3D, image2_points, intrinsic_matrix)
    elif ransac_method == 'lo_ransac':
        rvec, tvec, inliers = ransac_solver.lo_ransac(
            points_3D, image2_points, intrinsic_matrix)
    elif ransac_method == 'prosac':
        # Compute quality scores based on feature response or depth confidence
        quality_scores = np.ones(len(points_3D))  # Placeholder - you can improve this
        rvec, tvec, inliers = ransac_solver.prosac(
            points_3D, image2_points, intrinsic_matrix, quality_scores)

    else:
        # Fallback to OpenCV's RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3D, image2_points, intrinsic_matrix, None,
            reprojectionError=ransac_config['threshold'],
            iterationsCount=ransac_config['max_iterations'],
            confidence=ransac_config['confidence']
        )
        if not success:
            raise ValueError("Standard RANSAC failed")
        inliers = inliers.flatten() if inliers is not None else np.arange(len(points_3D))
    
    # Convert to rotation matrix
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    
    # Filter points to return only inliers
    inlier_image1_points = image1_points[inliers]
    inlier_image2_points = image2_points[inliers]
    
    return rotation_matrix, tvec, inlier_image1_points, inlier_image2_points

