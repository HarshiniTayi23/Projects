import open3d as o3d
import numpy as np
import os

map_file = r"C:\Users\tayis\OneDrive\Desktop\Projects\Personal\SLAM\output\map.ply"
trajectory_file = r"C:\Users\tayis\OneDrive\Desktop\Projects\Personal\SLAM\output\trajectory.txt"

if not os.path.exists(map_file):
    raise FileNotFoundError(f"Map file not found: {map_file}")
if not os.path.exists(trajectory_file):
    raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

# Load point cloud
pcd = o3d.io.read_point_cloud(map_file)

# Load trajectory as complex, then take only real values
trajectory = np.loadtxt(trajectory_file, dtype=np.complex128)
trajectory = np.real(trajectory)

if trajectory.ndim == 1:  # If only one point
    trajectory = trajectory.reshape(1, 3)

# Create trajectory line set
lines = [[i, i+1] for i in range(len(trajectory)-1)]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(trajectory)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red lines

# Add coordinate frame
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# Visualize
o3d.visualization.draw_geometries(
    [pcd, line_set, coord_frame],
    window_name="SLAM Map and Camera Trajectory",
    width=1024,
    height=768
)
