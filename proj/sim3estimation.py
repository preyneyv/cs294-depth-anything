import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import argparse
# from align import sim3_pipeline
from align_teaser import sim3_teaser_pipeline


# Step1: proj 2D depth image to 3D cloud points
# Intrinsic matrix from hw4
K = np.array([[493.9573761039707, 0.0, 517.5669208422318], # s=0
              [0.0, 493.39913655467205, 384.5477507228477],
              [0.0, 0.0, 1.0]])

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Depth image: H x W
# [WRONGLY IMPLEMENTED] should use float depth rather than a integer pixel value read from the depth img
depth_image = cv2.imread('./test_data/vanilla_depth_image_20241105_071643.png')  # Load your depth image here (shape H x W, values are depth)
assert np.array_equal(depth_image[:, :, 0], depth_image[:, :, 1])
assert np.array_equal(depth_image[:, :, 1], depth_image[:, :, 2])

depth = depth_image[:, :, 0]
# print(depth.shape)

# Create meshgrid of pixel coordinates
h, w = depth.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Flatten arrays for vectorized computation
u_flat = u.flatten()
v_flat = v.flatten()
depth_flat = depth.flatten()  # depth at each pixel

# [PREPROCESS TO THE DEPTH IMAGE, WRONLY IMPLEMENTED]
# reverse depth for non-metric depth image, make 255(white, close)->0(black, remote)
depth_flat = 255-depth_flat

# Optional: scale depth if it’s not metric
# depth_flat = depth_flat * scale_factor

# Back-project with K
X = (u_flat - cx) * depth_flat / fx
Y = (v_flat - cy) * depth_flat / fy
Z = depth_flat

# Stack into point cloud: shape (N, 3)
points_3d = np.vstack((X, Y, Z)).T
print(points_3d.shape)

# Optionally remove invalid points (where depth == 0)
valid = depth_flat > 0
points_3d = points_3d[valid]

pcd_cam = o3d.geometry.PointCloud()
pcd_cam.points = o3d.utility.Vector3dVector(points_3d)

# [optional] downsample to save memory for quick test
pcd_cam = pcd_cam.voxel_down_sample(voxel_size=8)

points_cam = np.asarray(pcd_cam.points)

# [optional] visualize the point cloud
fig = go.Figure(data=[go.Scatter3d(
    x=points_cam[:, 0],
    y=points_cam[:, 1],
    z=points_cam[:, 2],
    mode='markers',
    marker=dict(size=1, opacity=0.8)
)])

fig.update_layout(scene=dict(
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    zaxis=dict(title='Z'),
    aspectmode='data',
))
fig.show()


# Step2: Estimate s, R, T between the two 3D cloud points
# Load the LiDAR pointcloud
pcd_lidar = o3d.io.read_point_cloud('./test_data/point_cloud_20241105_071643.pcd')

# [optional] downsample to save memory for quick test
pcd_lidar = pcd_lidar.voxel_down_sample(voxel_size=1)

points_lidar = np.asarray(pcd_lidar.points)  # points in LiDAR frame

# [FLIP THE AXIS MANUALLY] Remap axes: (y, z, x) --> (x, y, z)
points_lidar = points_lidar[:, [1, 2, 0]]  # swap axes
pcd_lidar.points = o3d.utility.Vector3dVector(points_lidar)

# [optional] visualize the point cloud
fig = go.Figure(data=[go.Scatter3d(
    x=points_lidar[:, 0],
    y=points_lidar[:, 1],
    z=points_lidar[:, 2],
    mode='markers',
    marker=dict(size=1, opacity=0.8)
)])

fig.update_layout(scene=dict(
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    zaxis=dict(title='Z'),
    aspectmode='data',
))
fig.show()

# pcd_cam, pcd_lidar, args.noise, corr_cam, corr_lidar, verbose=True)
print("Starting the sim3_tease_pipeline...")
print("points_cam.shape", points_cam.shape)
print("points_lidar.shape", points_lidar.shape)

# noise_bound should be 0.05~0.1 as recommended in Teaser++
cam_aligned, metadata = sim3_teaser_pipeline(pcd_cam, pcd_lidar, 0.04, points_cam, points_lidar, verbose=True)

print("Best scale:", metadata["s"])
print("Best fitness:", metadata["fitness"], "RMSE:", metadata["rmse"])
# save
# o3d.io.write_point_cloud("cam_aligned.pcd", cam_aligned)
o3d.io.write_point_cloud("cam_aligned.pcd", cam_aligned, write_ascii=True)
print("Aligned camera cloud written to cam_aligned.pcd")
