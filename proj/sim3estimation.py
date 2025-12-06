import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import argparse
from scipy.ndimage import median_filter
# from align import sim3_pipeline
# from align_teaser import sim3_teaser_pipeline


def robust_reciprocal(depth_img, eps=1e-3,
                      pmin=0.01, pmax=99.5,
                      median_ks=3, bilateral_d=9, bilateral_sigma=75):
    # normalize to [0,1]
    inv = depth_img/depth_img.max()

    # percentile clipping (remove outliers)
    lo = np.percentile(inv, pmin)
    hi = np.percentile(inv, pmax)
    inv_clipped = np.clip(inv, lo, hi)

    # optional median filter to remove salt-and-pepper
    # inv_clipped = median_filter(inv_clipped, size=median_ks)

    # avoid exact zero
    inv_smooth = np.clip(inv_clipped, eps, 1.0)

    # invert to get relative depth (Z ∝ 1 / inv)
    z = 1.0 / inv_smooth

    # optional normalize to [0,1] for visualization or to [0,255] for point cloud scaling
    # z = (z - z.min()) / (z.max() - z.min())

    return z


# Step1: proj 2D depth image to 3D cloud points
# Intrinsic matrix from hw4
K = np.array([[493.9573761039707, 0.0, 517.5669208422318], # s=0
              [0.0, 493.39913655467205, 384.5477507228477],
              [0.0, 0.0, 1.0]])

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Depth image: H x W
depth = np.load('./test_data/vanilla_depth_raw_20241105_071643.npy') # Load your depth image here (shape H x W, values are depth)

# print(depth.shape)

# Create meshgrid of pixel coordinates
h, w = depth.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Flatten arrays for vectorized computation
u_flat = u.flatten()
v_flat = v.flatten()
depth_flat = depth.flatten()  # depth at each pixel
# depth_flat = depth

# [PREPROCESS TO THE DEPTH IMAGE, CHECK MANUALLY] reverse depth for non-metric depth image, make 255(white, close)->0(black, remote)
depth_flat = 1.0 / depth_flat
# depth_flat = np.sort(depth_flat)
# depth_flat = np.clip(depth_flat, 0, 20)


# Optional: use the packed function. Similar point cloud.
# depth_flat = robust_reciprocal(depth_flat, eps=2e-3, pmin=0.01, pmax=99.5)

# [LINEAR PROJECTION, MATHEMATICALLY WRONG] this has the best output, although mathematically it is wrong
# depth_flat = depth_flat.max() - depth_flat

# Back-project
X = (u_flat - cx) * depth_flat / fx
Y = (v_flat - cy) * depth_flat / fy
Z = depth_flat

# Stack into point cloud: shape (N, 3)
points_3d = np.vstack((X, Y, Z)).T
print(points_3d.shape)

# Remove sky points
valid = depth_flat < 0.5
points_3d = points_3d[valid]

pcd_cam = o3d.geometry.PointCloud()
pcd_cam.points = o3d.utility.Vector3dVector(points_3d)

# [optional] downsample to save memory for quick test
pcd_cam = pcd_cam.voxel_down_sample(voxel_size=1e-4)

points_cam = np.asarray(pcd_cam.points).copy()
points_cam[:, 1] = -points_cam[:, 1]   # invert y axis

# Filter by x-range
mask = (points_cam[:, 0] > -0.01) & (points_cam[:, 0] < 0.01)
points_cam = points_cam[mask]

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
    yaxis=dict(title='Y', autorange='reversed'),
    zaxis=dict(title='Z'),
    aspectmode='data',
))
fig.show()


# Step2: Estimate s, R, T between the two 3D cloud points
# Load the LiDAR pointcloud
pcd_lidar = o3d.io.read_point_cloud('./test_data/point_cloud_20241105_071643.pcd')

# [optional] downsample to save memory for quick test
# pcd_lidar = pcd_lidar.voxel_down_sample(voxel_size=1)

points_lidar = np.asarray(pcd_lidar.points)  # points in LiDAR frame

# [CHECK THE AXIS MANUALLY] Remap axes: (y, z, x) --> (x, y, z)
points_lidar = points_lidar[:, [1, 2, 0]]  # swap axes
points_lidar[:, 0] = -points_lidar[:, 0]   # invert x axis
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
    xaxis=dict(title='X', autorange='reversed'),
    yaxis=dict(title='Y'),
    zaxis=dict(title='Z'),
    aspectmode='data',
))
fig.show()

# pcd = o3d.io.read_point_cloud('./test_data/cam_aligned.pcd')  # Update the path
# points = np.asarray(pcd.points)  # Extract XYZ coordinates
#
# # Plot using Plotly
# fig = go.Figure(data=[go.Scatter3d(
#     x=points[:, 0],
#     y=points[:, 1],
#     z=points[:, 2],
#     mode='markers',
#     marker=dict(size=2, opacity=0.8)  # Adjust size if needed
# )])
#
# fig.update_layout(
#     title="Camera Aligned Result Visualization",
#     scene=dict(
#         xaxis=dict(title='X'),
#         yaxis=dict(title='Y'),
#         zaxis=dict(title='Z'),
#         aspectmode='data',
#     )
# )
# fig.show()

# pcd_cam, pcd_lidar, args.noise, corr_cam, corr_lidar, verbose=True)
cam_aligned, metadata = sim3_teaser_pipeline(pcd_cam, pcd_lidar, 0.05, points_cam, points_lidar, verbose=True)

print("Best scale:", metadata["s"])
print("Best fitness:", metadata["fitness"], "RMSE:", metadata["rmse"])
# save
o3d.io.write_point_cloud("cam_aligned.ply", cam_aligned)
print("Aligned camera cloud written to cam_aligned.ply")


