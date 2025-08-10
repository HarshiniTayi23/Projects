# Monocular Visual SLAM

A complete implementation of Monocular Visual Simultaneous Localization and Mapping (SLAM) using ORB features and essential matrix-based pose estimation.

## Overview

This project implements a complete monocular visual SLAM pipeline that can:
- Process monocular video input
- Extract and track ORB features across frames
- Estimate camera poses using essential matrix decomposition
- Generate a 3D point cloud map of the environment
- Detect loop closures for map consistency
- Visualize the robot trajectory and 3D map

## Features

### Core SLAM Components
- **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) features
- **Feature Matching**: Brute-force matching with cross-checking
- **Pose Estimation**: Essential matrix with RANSAC
- **Triangulation**: Linear triangulation for 3D point generation
- **Loop Closure**: Simple bag-of-words based detection
- **Bundle Adjustment**: Framework for optimization (placeholder implementation)

### Outputs
- 3D trajectory visualization (2D and 3D plots)
- Point cloud in PLY format
- Feature tracking video
- Configuration and results summary

### Parameters
- `--video`: Path to input monocular video file
- `--output`: Output directory for results (default: `slam_output`)
- `--max_frames`: Maximum number of frames to process (default: 500)
- `--skip_frames`: Process every N frames (default: 1)

## Algorithm Explanation

### 1. Pipeline Overview
The SLAM pipeline follows these main steps for each frame:

```
Input Frame → Feature Extraction → Feature Matching → Pose Estimation → 
Triangulation → Map Update → Loop Closure Detection → Bundle Adjustment
```

### 2. Feature Detection and Tracking
**ORB Features**: We use ORB (Oriented FAST and Rotated BRIEF) features because:
- **Scale Invariant**: Handles different distances from objects
- **Rotation Invariant**: Robust to camera rotation
- **Fast Computation**: Real-time performance
- **Binary Descriptors**: Efficient matching using Hamming distance

```python
# Feature extraction process
keypoints, descriptors = orb.detectAndCompute(gray_image, None)
matches = matcher.match(descriptors1, descriptors2)
```

### 3. Motion Estimation (Pose Calculation)
**Essential Matrix Method**:
- Estimates relative camera motion between consecutive frames
- Uses RANSAC for robust estimation against outliers
- Decomposes essential matrix to get rotation (R) and translation (t)

```python
E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)
```

**Why Essential Matrix?**
- Works with calibrated cameras
- Handles pure rotation and translation
- RANSAC provides outlier rejection
- More stable than fundamental matrix for calibrated setup

### 4. Map Point Triangulation
**Linear Triangulation**:
- Creates 3D points from 2D correspondences across frames
- Uses projection matrices from estimated poses
- Filters points based on geometric constraints

```python
# Triangulation process
P1 = K @ pose1[:3, :]  # First camera projection matrix
P2 = K @ pose2[:3, :]  # Second camera projection matrix
points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
points_3d = points_4d[:3, :] / points_4d[3, :]  # Normalize
```

### 5. Loop Closure Detection
**Bag-of-Words Approach**:
- Compares current frame descriptors with previous frames
- Uses feature matching to find similar views
- Requires temporal gap to avoid false positives

### 6. Bundle Adjustment (Framework)
**Purpose**: Jointly optimize camera poses and 3D points
- Minimizes reprojection errors
- Improves global map consistency
- Typically uses non-linear optimization (Levenberg-Marquardt)

### Libraries and Tools
- **OpenCV**: Computer vision library for feature detection and pose estimation
- **NumPy**: Numerical computing for matrix operations
- **Matplotlib**: Visualization of results
- **PLY Format**: Point cloud storage standard
