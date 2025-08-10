import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from collections import defaultdict, deque
import pickle
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm
import urllib.request
import tempfile
from scipy.linalg import fractional_matrix_power

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Structures

@dataclass
class CameraParams:
    """Camera intrinsic parameters"""
    fx: float = 718.856  # Focal length x (pixels)
    fy: float = 718.856  # Focal length y (pixels)
    cx: float = 607.193  # Principal point x (pixels)
    cy: float = 185.216  # Principal point y (pixels)
    k1: float = 0.0      # Radial distortion coefficient 1
    k2: float = 0.0      # Radial distortion coefficient 2
    p1: float = 0.0      # Tangential distortion coefficient 1
    p2: float = 0.0      # Tangential distortion coefficient 2
    k3: float = 0.0      # Radial distortion coefficient 3
    
    @property
    def K(self):
        """Camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @property
    def dist_coeffs(self):
        """Distortion coefficients"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

@dataclass
class Frame:
    """Represents a single video frame with extracted features"""
    id: int
    image: np.ndarray
    timestamp: float
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    pose: Optional[np.ndarray] = None  # 4x4 transformation matrix
    
class MapPoint:
    """3D point in the world coordinate system"""
    def __init__(self, position: np.ndarray, descriptor: np.ndarray = None):
        self.position = position  # 3D position (x, y, z)
        self.descriptor = descriptor  # Feature descriptor
        self.observations = {}  # frame_id -> keypoint_idx
        self.id = id(self)
        self.color = np.array([255, 255, 255], dtype=np.uint8)  # Default white
        self.outlier = False
        
class Map:
    """Global map containing all map points and keyframes"""
    def __init__(self):
        self.points = []  # List of MapPoint objects
        self.keyframes = []  # List of Frame objects (keyframes only)
        self.graph = defaultdict(list)  # Covisibility graph
        self.scale_factor = 1.0  # Scale factor for scale recovery
        
    def add_point(self, point: MapPoint):
        """Add a new map point"""
        self.points.append(point)
        
    def add_keyframe(self, frame: Frame):
        """Add a new keyframe"""
        self.keyframes.append(frame)
        
    def get_points_array(self):
        """Get all map points as numpy array"""
        if not self.points:
            return np.array([])
        return np.array([p.position * self.scale_factor for p in self.points if not p.outlier])
    
    def get_colors_array(self):
        """Get all point colors as numpy array"""
        if not self.points:
            return np.array([])
        return np.array([p.color for p in self.points if not p.outlier])
    
    def apply_scale_correction(self, scale_factor: float):
        """Apply scale correction to all map points and camera poses"""
        self.scale_factor *= scale_factor
        
        # Scale all map points
        for point in self.points:
            point.position *= scale_factor
            
        # Scale all keyframe poses (translation components)
        for keyframe in self.keyframes:
            if keyframe.pose is not None:
                keyframe.pose[:3, 3] *= scale_factor

# Scale Recovery Module

class ScaleRecovery:
    """Handle scale recovery for monocular SLAM"""
    
    def __init__(self):
        """Initialize scale recovery module"""
        self.reference_objects = {
            # Common object sizes in meters (width, height, depth)
            'person': (0.5, 1.7, 0.3),
            'car': (2.0, 1.5, 4.5),
            'door': (0.9, 2.1, 0.05),
            'book': (0.15, 0.23, 0.03),
            'laptop': (0.35, 0.25, 0.02),
            'phone': (0.08, 0.15, 0.01)
        }
        
        self.known_distances = []  # List of (distance, measured_distance) pairs
        self.estimated_scale = 1.0
        
    def add_known_distance(self, actual_distance: float, measured_distance: float):
        """
        Add a known distance measurement for scale estimation
        
        Args:
            actual_distance: True distance in meters
            measured_distance: Measured distance from SLAM (in arbitrary units)
        """
        if measured_distance > 0:
            scale_estimate = actual_distance / measured_distance
            self.known_distances.append((actual_distance, measured_distance))
            
            # Update scale estimate using median of all measurements
            scales = [actual / measured for actual, measured in self.known_distances]
            self.estimated_scale = np.median(scales)
            
    def estimate_scale_from_motion(self, trajectory: np.ndarray, 
                                   assumed_speed: float = 1.5) -> float:
        """
        Estimate scale from assumed motion speed (e.g., walking speed)
        
        Args:
            trajectory: Camera trajectory
            assumed_speed: Assumed speed in m/s (default: walking speed)
            
        Returns:
            scale_factor: Estimated scale factor
        """
        if len(trajectory) < 10:
            return 1.0
            
        # Calculate distance traveled
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # Assume time between frames (e.g., 30 FPS)
        total_time = len(trajectory) / 30.0
        
        # Calculate estimated speed
        estimated_speed = total_distance / total_time
        
        # Scale factor to match assumed speed
        scale_factor = assumed_speed / estimated_speed
        
        return scale_factor
    
    def get_scale_factor(self) -> float:
        """Get current scale factor estimate"""
        return self.estimated_scale

# Feature Detection & Tracking

class FeatureExtractor:
    """Extract and match ORB features between frames"""
    
    def __init__(self, n_features=2000, scale_factor=1.2, n_levels=8):
        """
        Initialize ORB feature extractor
        
        Args:
            n_features: Maximum number of features to extract
            scale_factor: Pyramid decimation ratio
            n_levels: Number of pyramid levels
        """
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # Feature matcher using brute force with Hamming distance
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def extract(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract ORB features from image
        
        Args:
            image: Grayscale image
            
        Returns:
            keypoints: List of detected keypoints
            descriptors: Feature descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match(self, desc1: np.ndarray, desc2: np.ndarray, 
              ratio_threshold: float = 0.7) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors using Lowe's ratio test
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_threshold: Threshold for Lowe's ratio test
            
        Returns:
            matches: List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Find k=2 nearest neighbors
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
                    
        return good_matches

# Motion Estimation

class PoseEstimator:
    """Estimate camera pose from feature correspondences"""
    
    def __init__(self, camera_params: CameraParams):
        """
        Initialize pose estimator
        
        Args:
            camera_params: Camera intrinsic parameters
        """
        self.camera = camera_params
        self.K = camera_params.K
        
    def estimate_pose_2d2d(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate relative pose between two frames using essential matrix
        
        Args:
            pts1: 2D points in first frame
            pts2: 2D points in second frame
            
        Returns:
            R: Rotation matrix
            t: Translation vector
            mask: Inlier mask
        """
        # Find essential matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            return None, None, None
        
        # Recover pose from essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, t, mask_pose
    
    def estimate_pose_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate camera pose using PnP (3D-2D correspondences)
        
        Args:
            points_3d: 3D world points
            points_2d: 2D image points
            
        Returns:
            rvec: Rotation vector
            tvec: Translation vector
        """
        if len(points_3d) < 4:
            return None, None
            
        # Solve PnP problem using RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d,
            self.K, self.camera.dist_coeffs,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return rvec, tvec
        return None, None

# Triangulation & Mapping

class Triangulator:
    """Triangulate 3D points from 2D correspondences"""
    
    def __init__(self, camera_params: CameraParams):
        """
        Initialize triangulator
        
        Args:
            camera_params: Camera intrinsic parameters
        """
        self.camera = camera_params
        self.K = camera_params.K
        
    def triangulate(self, pts1: np.ndarray, pts2: np.ndarray, 
                    P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from 2D correspondences
        
        Args:
            pts1: 2D points in first frame
            pts2: 2D points in second frame
            P1: Projection matrix of first camera
            P2: Projection matrix of second camera
            
        Returns:
            points_3d: Triangulated 3D points
        """
        # Ensure correct dtypes for OpenCV
        P1 = np.asarray(P1, dtype=np.float64)
        P2 = np.asarray(P2, dtype=np.float64)
        pts1 = np.asarray(pts1, dtype=np.float64)
        pts2 = np.asarray(pts2, dtype=np.float64)

        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def check_triangulation(self, points_3d: np.ndarray, P1: np.ndarray, 
                            P2: np.ndarray) -> np.ndarray:
        """
        Check if triangulated points are in front of both cameras
        
        Args:
            points_3d: 3D points
            P1: Projection matrix of first camera
            P2: Projection matrix of second camera
            
        Returns:
            mask: Boolean mask of valid points
        """
        # Ensure correct dtype
        points_3d = np.asarray(points_3d, dtype=np.float64)
        P1 = np.asarray(P1, dtype=np.float64)
        P2 = np.asarray(P2, dtype=np.float64)

        # Convert to homogeneous coordinates
        points_4d = np.hstack([points_3d, np.ones((len(points_3d), 1), dtype=np.float64)])
        
        # Check depth in first camera
        z1 = (P1[2:3] @ points_4d.T).flatten()
        
        # Check depth in second camera
        z2 = (P2[2:3] @ points_4d.T).flatten()
        
        # Points should be in front of both cameras (positive depth)
        mask = (z1 > 0) & (z2 > 0)
        
        return mask

# Loop Closure Detection

class LoopDetector:
    """Detect loop closures using bag of words"""
    
    def __init__(self, vocabulary_size=1000, min_score=0.3):
        """
        Initialize loop detector
        
        Args:
            vocabulary_size: Size of visual vocabulary
            min_score: Minimum similarity score for loop detection
        """
        self.vocabulary_size = vocabulary_size
        self.min_score = min_score
        self.database = []  # Store frame descriptors
        self.frame_ids = []  # Store corresponding frame IDs
        
    def add_frame(self, frame_id: int, descriptors: np.ndarray):
        """
        Add frame to loop detection database
        
        Args:
            frame_id: Frame ID
            descriptors: Frame descriptors
        """
        if descriptors is not None and len(descriptors) > 0:
            # Compute histogram of visual words (simplified BoW)
            hist = self._compute_bow_vector(descriptors)
            self.database.append(hist)
            self.frame_ids.append(frame_id)
    
    def detect_loop(self, descriptors: np.ndarray, current_frame_id: int, 
                   min_frames_apart: int = 30) -> Optional[int]:
        """
        Detect if current frame closes a loop
        
        Args:
            descriptors: Current frame descriptors
            current_frame_id: Current frame ID
            min_frames_apart: Minimum frames between loop closure
            
        Returns:
            loop_frame_id: ID of loop closure frame or None
        """
        if descriptors is None or len(descriptors) == 0:
            return None
            
        # Compute BoW vector for current frame
        current_hist = self._compute_bow_vector(descriptors)
        
        best_score = 0
        best_frame_id = None
        
        # Compare with previous frames
        for i, (hist, frame_id) in enumerate(zip(self.database, self.frame_ids)):
            # Skip recent frames
            if abs(current_frame_id - frame_id) < min_frames_apart:
                continue
                
            # Compute similarity score
            score = self._compute_similarity(current_hist, hist)
            
            if score > best_score and score > self.min_score:
                best_score = score
                best_frame_id = frame_id
                
        return best_frame_id
    
    def _compute_bow_vector(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Compute bag of words vector (simplified version)
        
        Args:
            descriptors: Feature descriptors
            
        Returns:
            bow_vector: Histogram of visual words
        """
        # Simple histogram based on descriptor values
        
        hist = np.zeros(self.vocabulary_size)
        
        for desc in descriptors:
            # Hash descriptor to vocabulary index
            idx = int(np.sum(desc) % self.vocabulary_size)
            hist[idx] += 1
            
        # Normalize histogram
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
            
        return hist
    
    def _compute_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compute similarity between two BoW vectors
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            
        Returns:
            similarity: Similarity score [0, 1]
        """
        # Use histogram intersection
        return np.sum(np.minimum(hist1, hist2))

# Bundle Adjustment

class BundleAdjuster:
    """Optimize camera poses and 3D points"""
    
    def __init__(self, camera_params: CameraParams):
        """
        Initialize bundle adjuster
        
        Args:
            camera_params: Camera intrinsic parameters
        """
        self.camera = camera_params
        self.K = camera_params.K
        
    def local_bundle_adjustment(self, frames: List[Frame], map_points: List[MapPoint], 
                                window_size: int = 5):
        """
        Perform local bundle adjustment on recent frames
        
        Args:
            frames: List of frames
            map_points: List of map points
            window_size: Number of frames to optimize
        """
        # Simplified BA: just refine poses using PnP
                
        if len(frames) < 2:
            return
            
        # Get recent frames
        recent_frames = frames[-window_size:]
        
        for frame in recent_frames:
            # Skip if pose is invalid
            if frame.pose is None or not isinstance(frame.pose, np.ndarray):
                continue
            if frame.pose.shape != (4, 4) or np.isnan(frame.pose).any():
                continue

            # Collect 3D-2D correspondences
            points_3d = []
            points_2d = []
            
            for point in map_points:
                if frame.id in point.observations:
                    kp_idx = point.observations[frame.id]
                    if kp_idx < len(frame.keypoints):
                        points_3d.append(point.position)
                        points_2d.append(frame.keypoints[kp_idx].pt)
            
            if len(points_3d) >= 10:
                points_3d = np.array(points_3d, dtype=np.float64)
                points_2d = np.array(points_2d, dtype=np.float64)
                
                # Refine pose
                rvec, tvec = self._refine_pose(points_3d, points_2d, frame.pose)
                
                if rvec is not None:
                    # Update frame pose
                    R_refined = cv2.Rodrigues(rvec)[0]
                    frame.pose[:3, :3] = R_refined
                    frame.pose[:3, 3] = tvec.flatten()
    
    def _refine_pose(self, points_3d: np.ndarray, points_2d: np.ndarray, 
                     initial_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Refine camera pose using PnP"""
        # Skip if pose is invalid
        if initial_pose is None or not isinstance(initial_pose, np.ndarray):
            return None, None
        if initial_pose.shape != (4, 4) or np.isnan(initial_pose).any():
            return None, None

        # Convert initial pose to rvec, tvec
        rvec_init = cv2.Rodrigues(initial_pose[:3, :3].astype(np.float64))[0]
        tvec_init = initial_pose[:3, 3].reshape(-1, 1).astype(np.float64)
        
        # Refine using PnP with initial guess
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d,
            self.K, self.camera.dist_coeffs,
            rvec_init, tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return rvec, tvec
        return None, None


# IMU/Odometry Simulation

class SyntheticIMU:
    """Generate synthetic IMU data with noise"""
    
    def __init__(self, noise_accel=0.01, noise_gyro=0.001, bias_accel=0.05, bias_gyro=0.01):
        """
        Initialize synthetic IMU
        
        Args:
            noise_accel: Accelerometer noise standard deviation (m/s²)
            noise_gyro: Gyroscope noise standard deviation (rad/s)
            bias_accel: Accelerometer bias (m/s²)
            bias_gyro: Gyroscope bias (rad/s)
        """
        self.noise_accel = noise_accel
        self.noise_gyro = noise_gyro
        self.bias_accel = bias_accel
        self.bias_gyro = bias_gyro
        
    def generate_measurements(self, true_accel: np.ndarray = None, 
                            true_gyro: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic IMU measurements
        
        Args:
            true_accel: True acceleration (m/s²)
            true_gyro: True angular velocity (rad/s)
            
        Returns:
            accel: Noisy acceleration measurement
            gyro: Noisy gyroscope measurement
        """
        if true_accel is None:
            # Default: gravity only
            true_accel = np.array([0, 0, 9.81])
            
        if true_gyro is None:
            # Default: no rotation
            true_gyro = np.array([0, 0, 0])
            
        # Add noise and bias
        accel = true_accel + np.random.normal(0, self.noise_accel, 3) + self.bias_accel
        gyro = true_gyro + np.random.normal(0, self.noise_gyro, 3) + self.bias_gyro
        
        return accel, gyro

# Main SLAM Pipeline

class MonocularSLAM:
    """Main monocular visual SLAM system"""
    
    def __init__(self, camera_params: CameraParams = None):
        """
        Initialize SLAM system
        
        Args:
            camera_params: Camera intrinsic parameters
        """
        # Use default camera parameters if not provided
        self.camera = camera_params if camera_params else CameraParams()
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.pose_estimator = PoseEstimator(self.camera)
        self.triangulator = Triangulator(self.camera)
        self.loop_detector = LoopDetector()
        self.bundle_adjuster = BundleAdjuster(self.camera)
        self.imu = SyntheticIMU()
        self.scale_recovery = ScaleRecovery()
        
        # SLAM state
        self.map = Map()
        self.frames = []
        self.current_frame_id = 0
        self.initialized = False
        
        # Tracking state
        self.reference_frame = None
        self.tracking_confidence = 0
        
        # Visualization
        self.trajectory = []
        
    def process_frame(self, image: np.ndarray, timestamp: float = None) -> bool:
        """
        Process a single frame through the SLAM pipeline
        
        Args:
            image: Input image
            timestamp: Frame timestamp
            
        Returns:
            success: Whether frame was successfully processed
        """
        if timestamp is None:
            timestamp = self.current_frame_id / 30.0  # Assume 30 FPS
            
        # Extract features
        keypoints, descriptors = self.feature_extractor.extract(image)
        
        if descriptors is None or len(keypoints) < 50:
            logger.warning(f"Insufficient features in frame {self.current_frame_id}")
            return False
            
        # Create frame object
        frame = Frame(
            id=self.current_frame_id,
            image=image,
            timestamp=timestamp,
            keypoints=keypoints,
            descriptors=descriptors
        )
        
        # Initialize or track
        if not self.initialized:
            success = self._initialize(frame)
        else:
            success = self._track(frame)
            
        if success:
            self.frames.append(frame)
            self.current_frame_id += 1
            
            # Add to trajectory
            if frame.pose is not None:
                position = frame.pose[:3, 3]
                self.trajectory.append(position)
                
            # Perform scale recovery periodically
            if len(self.trajectory) > 50 and len(self.trajectory) % 50 == 0:
                self._perform_scale_recovery()
                
            # Check for loop closure
            if self.current_frame_id % 10 == 0:  # Check every 10 frames
                loop_id = self.loop_detector.detect_loop(
                    frame.descriptors, frame.id
                )
                if loop_id is not None:
                    logger.info(f"Loop closure detected: frame {frame.id} -> {loop_id}")
                    self._close_loop(frame, self.frames[loop_id])
                    
            # Add frame to loop detector database
            self.loop_detector.add_frame(frame.id, frame.descriptors)
            
            # Local bundle adjustment
            if len(self.frames) % 5 == 0:  # Every 5 frames
                self.bundle_adjuster.local_bundle_adjustment(
                    self.frames, self.map.points
                )
                
        return success
    
    def _perform_scale_recovery(self):
        """Perform scale recovery using motion assumptions"""
        if len(self.trajectory) < 10:
            return
            
        trajectory_array = np.array(self.trajectory)
        
        # Estimate scale from assumed walking speed
        scale_factor = self.scale_recovery.estimate_scale_from_motion(
            trajectory_array, assumed_speed=1.5  # Assume 1.5 m/s walking speed
        )
        
        # Apply scale correction if reasonable
        if 0.1 < scale_factor < 10.0:  # Reasonable scale range
            self.map.apply_scale_correction(scale_factor)
            
            # Update trajectory
            self.trajectory = [pos * scale_factor for pos in self.trajectory]
            
            logger.info(f"Applied scale correction: {scale_factor:.3f}")
    
    def _initialize(self, frame: Frame) -> bool:
        """
        Initialize SLAM with first keyframe
        
        Args:
            frame: Current frame
            
        Returns:
            success: Whether initialization was successful
        """
        if self.reference_frame is None:
            # Set first frame as reference
            self.reference_frame = frame
            frame.pose = np.eye(4)  # Identity pose for first frame
            self.map.add_keyframe(frame)
            logger.info("Set first frame as reference")
            return True
            
        # Try to initialize with two-view geometry
        matches = self.feature_extractor.match(
            self.reference_frame.descriptors,
            frame.descriptors
        )
        
        if len(matches) < 100:
            logger.warning(f"Insufficient matches for initialization: {len(matches)}")
            return False
            
        # Get matched points
        pts1 = np.array([self.reference_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate relative pose
        R, t, mask = self.pose_estimator.estimate_pose_2d2d(pts1, pts2)
        
        if R is None:
            logger.warning("Failed to estimate initial pose")
            return False
            
        # Set frame pose
        frame.pose = np.eye(4)
        frame.pose[:3, :3] = R
        frame.pose[:3, 3] = t.flatten()
        
        # Triangulate initial map points
        P1 = self.camera.K @ np.eye(3, 4)
        P2 = self.camera.K @ frame.pose[:3]
        
        points_3d = self.triangulator.triangulate(pts1, pts2, P1, P2)
        
        # Check triangulation
        valid_mask = self.triangulator.check_triangulation(points_3d, P1, P2)
        
        # Add valid points to map
        for i, (point_3d, valid) in enumerate(zip(points_3d, valid_mask)):
            if valid and mask[i]:
                # Create map point
                map_point = MapPoint(point_3d, frame.descriptors[matches[i].trainIdx])
                
                # Add observations
                map_point.observations[self.reference_frame.id] = matches[i].queryIdx
                map_point.observations[frame.id] = matches[i].trainIdx
                
                # Extract color from image
                kp = frame.keypoints[matches[i].trainIdx]
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= y < frame.image.shape[0] and 0 <= x < frame.image.shape[1]:
                    if len(frame.image.shape) == 3:
                        map_point.color = frame.image[y, x][::-1]  # BGR to RGB
                
                self.map.add_point(map_point)
                
        if len(self.map.points) > 50:
            self.map.add_keyframe(frame)
            self.initialized = True
            logger.info(f"SLAM initialized with {len(self.map.points)} points")
            return True
            
        logger.warning("Insufficient triangulated points for initialization")
        return False
    
    def _track(self, frame: Frame) -> bool:
        """
        Track camera pose for current frame
        
        Args:
            frame: Current frame
            
        Returns:
            success: Whether tracking was successful
        """
        if not self.frames:
            return False
            
        # Match with previous frame
        prev_frame = self.frames[-1]
        matches = self.feature_extractor.match(
            prev_frame.descriptors,
            frame.descriptors
        )
        
        if len(matches) < 30:
            logger.warning(f"Lost tracking: insufficient matches ({len(matches)})")
            self.tracking_confidence = 0
            return False
            
        # Collect 3D-2D correspondences from map
        points_3d = []
        points_2d = []
        
        for match in matches:
            # Check if previous keypoint corresponds to a map point
            for map_point in self.map.points:
                if prev_frame.id in map_point.observations:
                    if map_point.observations[prev_frame.id] == match.queryIdx:
                        points_3d.append(map_point.position)
                        points_2d.append(frame.keypoints[match.trainIdx].pt)
                        break
                        
        if len(points_3d) >= 10:
            # Estimate pose using PnP
            points_3d = np.array(points_3d)
            points_2d = np.array(points_2d)
            
            rvec, tvec = self.pose_estimator.estimate_pose_pnp(points_3d, points_2d)
            
            if rvec is not None:
                # Set frame pose
                frame.pose = np.eye(4)
                frame.pose[:3, :3] = cv2.Rodrigues(rvec)[0]
                frame.pose[:3, 3] = tvec.flatten()
                
                self.tracking_confidence = min(1.0, len(points_3d) / 100.0)
                
                # Decide if keyframe
                if self._is_keyframe(frame, prev_frame):
                    self.map.add_keyframe(frame)
                    self._create_new_map_points(frame, prev_frame, matches)
                    
                logger.debug(f"Tracked frame {frame.id} with {len(points_3d)} points")
                return True
        else:
            # Try to recover using motion model or two-view geometry
            pts1 = np.array([prev_frame.keypoints[m.queryIdx].pt for m in matches])
            pts2 = np.array([frame.keypoints[m.trainIdx].pt for m in matches])
            
            R, t, mask = self.pose_estimator.estimate_pose_2d2d(pts1, pts2)
            
            if R is not None:
                # Apply relative transformation
                frame.pose = prev_frame.pose @ np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
                
                self.tracking_confidence = 0.5
                
                if self._is_keyframe(frame, prev_frame):
                    self.map.add_keyframe(frame)
                    self._create_new_map_points(frame, prev_frame, matches)
                    
                return True
                
        logger.warning(f"Failed to track frame {frame.id}")
        return False
    
    def _is_keyframe(self, frame: Frame, prev_frame: Frame) -> bool:
        """
        Decide if current frame should be a keyframe
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            
        Returns:
            is_keyframe: Whether frame should be a keyframe
        """
        # Keyframe selection criteria
        # 1. Minimum frame gap
        if len(self.map.keyframes) > 0:
            last_kf = self.map.keyframes[-1]
            if frame.id - last_kf.id < 5:  # Minimum 5 frames apart
                return False
                
        # 2. Sufficient motion
        if frame.pose is not None and prev_frame.pose is not None:
            # Calculate translation
            translation = np.linalg.norm(frame.pose[:3, 3] - prev_frame.pose[:3, 3])
            
            # Calculate rotation (angle from rotation matrix)
            R_rel = frame.pose[:3, :3] @ prev_frame.pose[:3, :3].T
            angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            
            # Require minimum motion
            if translation < 0.1 and angle < np.radians(5):
                return False
                
        # 3. Tracking quality
        if self.tracking_confidence < 0.3:
            return True  # Force keyframe if tracking is poor
            
        return True
    
    def _create_new_map_points(self, frame: Frame, prev_frame: Frame, matches: List[cv2.DMatch]):
        """
        Create new map points from unmatched features
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            matches: Feature matches between frames
        """
        # Get matched indices
        matched_curr = set([m.trainIdx for m in matches])
        matched_prev = set([m.queryIdx for m in matches])
        
        # Get corresponding points
        pts1 = np.array([prev_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Triangulate new points
        P1 = self.camera.K @ prev_frame.pose[:3]
        P2 = self.camera.K @ frame.pose[:3]
        
        points_3d = self.triangulator.triangulate(pts1, pts2, P1, P2)
        
        # Check triangulation
        valid_mask = self.triangulator.check_triangulation(points_3d, P1, P2)
        
        # Add valid points to map
        for i, (point_3d, valid) in enumerate(zip(points_3d, valid_mask)):
            if valid:
                # Check if point already exists in map
                exists = False
                for map_point in self.map.points:
                    if prev_frame.id in map_point.observations:
                        if map_point.observations[prev_frame.id] == matches[i].queryIdx:
                            # Add new observation
                            map_point.observations[frame.id] = matches[i].trainIdx
                            exists = True
                            break
                            
                if not exists:
                    # Create new map point
                    map_point = MapPoint(point_3d, frame.descriptors[matches[i].trainIdx])
                    map_point.observations[prev_frame.id] = matches[i].queryIdx
                    map_point.observations[frame.id] = matches[i].trainIdx
                    
                    # Extract color
                    kp = frame.keypoints[matches[i].trainIdx]
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    if 0 <= y < frame.image.shape[0] and 0 <= x < frame.image.shape[1]:
                        if len(frame.image.shape) == 3:
                            map_point.color = frame.image[y, x][::-1]  # BGR to RGB
                    
                    self.map.add_point(map_point)
    
    def _close_loop(self, current_frame: Frame, loop_frame: Frame):
        """
        Close a detected loop
        
        Args:
            current_frame: Current frame
            loop_frame: Loop closure frame
        """
        # Match features between loop frames
        matches = self.feature_extractor.match(
            current_frame.descriptors,
            loop_frame.descriptors
        )
        
        if len(matches) < 50:
            logger.warning("Insufficient matches for loop closure")
            return
            
        # Get matched points
        pts1 = np.array([current_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([loop_frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate relative pose
        R, t, mask = self.pose_estimator.estimate_pose_2d2d(pts1, pts2)
        
        if R is not None:
            # Compute pose correction
            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t.flatten()
            
            expected_pose = loop_frame.pose @ relative_pose
            pose_error = expected_pose @ np.linalg.inv(current_frame.pose)
            
            # Apply correction to trajectory 
            correction_weight = 0.5
            for frame in self.frames[loop_frame.id:]:
                frame.pose = frame.pose @ fractional_matrix_power(
                    pose_error, correction_weight
                )
                
            logger.info("Loop closure correction applied")
    
    def get_camera_trajectory(self) -> np.ndarray:
        """
        Get camera trajectory as array of positions
        
        Returns:
            trajectory: Nx3 array of camera positions
        """
        if not self.trajectory:
            return np.array([])
        return np.array(self.trajectory)
    
    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D point cloud and colors
        
        Returns:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
        """
        points = self.map.get_points_array()
        colors = self.map.get_colors_array()
        return points, colors
    
    def save_map(self, filename: str):
        """
        Save map to file
        
        Args:
            filename: Output filename (.ply or .pcd)
        """
        points, colors = self.get_point_cloud()
        
        if len(points) == 0:
            logger.warning("No points to save")
            return
            
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Save to file
        o3d.io.write_point_cloud(filename, pcd)
        logger.info(f"Saved point cloud to {filename} ({len(points)} points)")
    
    def save_trajectory(self, filename: str):
        """
        Save camera trajectory to file in simple x y z format
        
        Args:
            filename: Output filename
        """
        trajectory = self.get_camera_trajectory()
        
        if len(trajectory) == 0:
            logger.warning("No trajectory to save")
            return
        
        # Save trajectory as simple float triplets (x y z format)
        with open(filename, 'w') as f:
            f.write("# Camera trajectory (x y z format)\n")
            f.write("# Each line represents a camera position in world coordinates\n")
            for pos in trajectory:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                
        logger.info(f"Saved trajectory to {filename} ({len(trajectory)} poses)")
    
    def add_scale_reference(self, actual_distance: float, point1_id: int, point2_id: int):
        """
        Add a scale reference using known distance between two points
        
        Args:
            actual_distance: Known distance in meters
            point1_id: ID of first map point
            point2_id: ID of second map point
        """
        if point1_id < len(self.map.points) and point2_id < len(self.map.points):
            point1 = self.map.points[point1_id]
            point2 = self.map.points[point2_id]
            
            # Calculate measured distance
            measured_distance = np.linalg.norm(point1.position - point2.position)
            
            # Add to scale recovery
            self.scale_recovery.add_known_distance(actual_distance, measured_distance)
            
            # Apply scale correction
            scale_factor = self.scale_recovery.get_scale_factor()
            self.map.apply_scale_correction(scale_factor)
            
            # Update trajectory
            self.trajectory = [pos * scale_factor for pos in self.trajectory]
            
            logger.info(f"Applied scale reference: {actual_distance}m -> scale factor {scale_factor:.3f}")

# Visualization

class SLAMVisualizer:
    """Visualize SLAM results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.fig = None
        self.axes = None
        
    def plot_trajectory(self, trajectory: np.ndarray, title: str = "Camera Trajectory"):
        """
        Plot camera trajectory in 2D and 3D
        
        Args:
            trajectory: Nx3 array of camera positions
            title: Plot title
        """
        if len(trajectory) == 0:
            logger.warning("No trajectory to plot")
            return
            
        fig = plt.figure(figsize=(15, 5))
        
        # 2D trajectory (top view)
        ax1 = fig.add_subplot(131)
        ax1.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax1.scatter(trajectory[0, 0], trajectory[0, 2], c='g', s=100, marker='o', label='Start')
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 2], c='r', s=100, marker='*', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('Top View')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # 2D trajectory (side view)
        ax2 = fig.add_subplot(132)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=100, marker='o', label='Start')
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=100, marker='*', label='End')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Side View')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # 3D trajectory
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(trajectory[:, 0], trajectory[:, 2], trajectory[:, 1], 'b-', linewidth=2)
        ax3.scatter(trajectory[0, 0], trajectory[0, 2], trajectory[0, 1], 
                   c='g', s=100, marker='o', label='Start')
        ax3.scatter(trajectory[-1, 0], trajectory[-1, 2], trajectory[-1, 1], 
                   c='r', s=100, marker='*', label='End')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_zlabel('Y (m)')
        ax3.set_title('3D View')
        ax3.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('trajectory.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_point_cloud(self, points: np.ndarray, colors: np.ndarray = None,
                            trajectory: np.ndarray = None):
        """
        Visualize 3D point cloud using Open3D
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
            trajectory: Camera trajectory
        """
        if len(points) == 0:
            logger.warning("No points to visualize")
            return
            
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        else:
            # Default gray color
            pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(points), 3)) * 0.5
            )
            
        geometries = [pcd]
        
        # Add trajectory as line set
        if trajectory is not None and len(trajectory) > 1:
            lines = [[i, i+1] for i in range(len(trajectory)-1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(trajectory)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(
                [[1, 0, 0] for _ in lines]  # Red trajectory
            )
            geometries.append(line_set)
            
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        geometries.append(coord_frame)
        
        # Visualize
        o3d.visualization.draw_geometries(
            geometries,
            window_name="SLAM Point Cloud",
            width=1024,
            height=768
        )
        
    def plot_features(self, image: np.ndarray, keypoints: List[cv2.KeyPoint], 
                     title: str = "Detected Features"):
        """
        Plot detected features on image
        
        Args:
            image: Input image
            keypoints: Detected keypoints
            title: Plot title
        """
        # Draw keypoints
        img_with_kp = cv2.drawKeypoints(
            image, keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        plt.figure(figsize=(12, 8))
        if len(img_with_kp.shape) == 3:
            plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img_with_kp, cmap='gray')
        plt.title(f"{title} ({len(keypoints)} features)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('features.png', dpi=150, bbox_inches='tight')
        plt.show()

# Video Processing

def download_video(url: str, output_path: str) -> bool:
    """
    Download video from URL
    
    Args:
        url: Video URL
        output_path: Output file path
        
    Returns:
        success: Whether download was successful
    """
    try:
        logger.info(f"Downloading video from {url}")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Video downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return False

def process_video(video_path: str, slam: MonocularSLAM, 
                 max_frames: int = None, skip_frames: int = 1):
    """
    Process video through SLAM pipeline
    
    Args:
        video_path: Path to input video
        slam: SLAM system instance
        max_frames: Maximum number of frames to process
        skip_frames: Number of frames to skip
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
    
    frame_count = 0
    processed_count = 0
    
    # Process frames
    pbar = tqdm(total=min(max_frames, total_frames) if max_frames else total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
            
        if max_frames and processed_count >= max_frames:
            break
            
        # Skip frames if requested
        if frame_count % (skip_frames + 1) == 0:
            timestamp = frame_count / fps
            
            # Process frame
            success = slam.process_frame(frame, timestamp)
            
            if success:
                processed_count += 1
                
            # Visualize first frame features
            if processed_count == 1 and slam.frames:
                visualizer = SLAMVisualizer()
                visualizer.plot_features(
                    frame, slam.frames[0].keypoints,
                    "First Frame Features"
                )
                
        frame_count += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    
    logger.info(f"Processed {processed_count} frames")

# Main Function

def main():
    """Main function to run SLAM pipeline"""
    
    parser = argparse.ArgumentParser(description='Monocular Visual SLAM Pipeline')
    parser.add_argument('--video', type=str, 
        default=os.path.join(os.path.dirname(__file__), "IMG_2790.MOV"),
        help='Path or URL to input video')
    parser.add_argument('--max_frames', type=int, default=500,
        help='Maximum number of frames to process')
    parser.add_argument('--skip_frames', type=int, default=2,
        help='Number of frames to skip')
    parser.add_argument('--output_dir', type=str, default='output',
        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
        help='Enable visualization')
    parser.add_argument('--scale_hint', type=float, default=1.5,
        help='Assumed motion speed in m/s for scale recovery (default: 1.5 for walking)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download video if URL provided
    video_path = args.video
    if args.video.startswith('http'):
        video_path = os.path.join(args.output_dir, 'input_video.mp4')
        if not os.path.exists(video_path):
            # Convert Google Drive link to direct download
            if 'drive.google.com' in args.video:
                file_id = args.video.split('/d/')[1].split('/')[0]
                download_url = f'https://drive.google.com/uc?id={file_id}&export=download'
            else:
                download_url = args.video
                
            success = download_video(download_url, video_path)
            if not success:
                logger.error("Failed to download video")
                return
                
    # Initialize SLAM system
    logger.info("Initializing SLAM system...")
    slam = MonocularSLAM()
    
    # Set scale hint for scale recovery
    slam.scale_recovery = ScaleRecovery()
    
    # Process video
    logger.info("Processing video...")
    process_video(video_path, slam, args.max_frames, args.skip_frames)
    
    # Apply final scale recovery
    if len(slam.trajectory) > 10:
        trajectory_array = np.array(slam.trajectory)
        final_scale = slam.scale_recovery.estimate_scale_from_motion(
            trajectory_array, assumed_speed=args.scale_hint
        )
        if 0.1 < final_scale < 10.0:
            slam.map.apply_scale_correction(final_scale)
            slam.trajectory = [pos * final_scale for pos in slam.trajectory]
            logger.info(f"Applied final scale correction: {final_scale:.3f}")
    
    # Save results
    logger.info("Saving results...")
    
    # Save point cloud
    slam.save_map(os.path.join(args.output_dir, 'map.ply'))
    
    # Save trajectory
    slam.save_trajectory(os.path.join(args.output_dir, 'trajectory.txt'))
    
    # Visualize results
    if args.visualize:
        visualizer = SLAMVisualizer()
        
        # Plot trajectory
        trajectory = slam.get_camera_trajectory()
        if len(trajectory) > 0:
            visualizer.plot_trajectory(trajectory)
            
        # Visualize point cloud
        points, colors = slam.get_point_cloud()
        if len(points) > 0:
            visualizer.visualize_point_cloud(points, colors, trajectory)
            
    # Print statistics
    points, _ = slam.get_point_cloud()
    trajectory = slam.get_camera_trajectory()
    
    print("\n" + "="*50)
    print("SLAM Pipeline Results:")
    print("="*50)
    print(f"Total frames processed: {len(slam.frames)}")
    print(f"Total keyframes: {len(slam.map.keyframes)}")
    print(f"Total map points: {len(points)}")
    print(f"Total trajectory points: {len(trajectory)}")
    print(f"Final scale factor: {slam.map.scale_factor:.3f}")
    print(f"Output directory: {args.output_dir}")
    print("="*50)
    
    logger.info("SLAM pipeline completed successfully!")

if __name__ == "__main__":
    main()