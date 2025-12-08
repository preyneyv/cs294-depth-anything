import os
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from align_teaser import sim3_teaser_pipeline
import json

# === Batch SIM(3) Alignment Pipeline ===
# Folder structure expected:
# dataset/
#   depth/
#     depth_00001.npy
#     depth_00002.npy
#   lidar/
#     lidar_00001.pcd
#     lidar_00002.pcd

DATA_ROOT = Path("../dataset")
DEPTH_DIR = DATA_ROOT / "depth"
LIDAR_DIR = DATA_ROOT / "lidar"
OUTPUT_DIR = DATA_ROOT / "aligned"
OUTPUT_DIR.mkdir(exist_ok=True)

META_PATH = OUTPUT_DIR / "metadata.jsonl"  # store full metadata per index

# Camera intrinsics
K = np.array([
    [491.331107883326, 0.0, 515.3434363622374],
    [0.0, 492.14998153326013, 388.93983736974667],
    [0.0, 0.0, 1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]


def backproject_depth(depth_img):
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    depth_flat = depth_img.flatten().astype(np.float32)
    u_flat = u.flatten()
    v_flat = v.flatten()

    depth_flat = 1.0 / depth_flat

    X = (u_flat - cx) * depth_flat / fx
    Y = (v_flat - cy) * depth_flat / fy
    Z = depth_flat

    pts = np.vstack((X, Y, Z)).T

    # Remove sky points. 0.5's setting is important
    valid = depth_flat < 0.5
    pts = pts[valid]

    # [CHECK THIS MANUALLY]
    # post-process to get x-y-z axis in the same direction with LiDAR pcd
    pts[:, 1] = -pts[:, 1] # invert y axis
    # Filter by x-range, rm noisy points
    mask = (pts[:, 0] > -0.01) & (pts[:, 0] < 0.01)
    pts = pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # [optional] downsample to save memory for quick test
    pcd = pcd.voxel_down_sample(voxel_size=1e-4)

    return pcd


def process_pair(idx):
    depth_path = DEPTH_DIR / f"depth_{idx}.npy"
    lidar_path = LIDAR_DIR / f"lidar_{idx}.pcd"

    if not depth_path.exists() or not lidar_path.exists():
        print(f"[SKIP] Missing pair for index {idx}")
        return

    depth = np.load(depth_path)
    pcd_cam = backproject_depth(depth)
    points_cam = np.asarray(pcd_cam.points).copy()

    pcd_lidar = o3d.io.read_point_cloud(str(lidar_path))

    # [CHECK THE AXIS MANUALLY] post-process to get x-y-z axis
    pts_lidar = np.asarray(pcd_lidar.points)[:, [1, 2, 0]] # Remap axes: (y, z, x) --> (x, y, z)
    pts_lidar[:, 0] = -pts_lidar[:, 0] # invert x axis

    pcd_lidar.points = o3d.utility.Vector3dVector(pts_lidar)

    # [optional] downsample to save memory for quick test
    pcd_lidar = pcd_lidar.voxel_down_sample(voxel_size=1)

    cam_aligned, metadata = sim3_teaser_pipeline(
        pcd_cam,
        pcd_lidar,
        0.05,
        points_cam,
        pts_lidar,
        verbose=False
    )

    save_path = OUTPUT_DIR / f"cam_aligned_{idx}.ply"
    o3d.io.write_point_cloud(str(save_path), cam_aligned)

    metadata["index"] = idx
    with open(META_PATH, "a") as f:
        f.write(json.dumps(metadata, default=str) + "\n")

    print(f"[DONE] {idx} saved → cam_aligned_{idx}.ply")


if __name__ == "__main__":
    all_depth = sorted([f.name.split("_")[1].split(".")[0] for f in DEPTH_DIR.glob("*.npy")])

    with open(META_PATH, "w") as f:
        pass

    for idx in tqdm(all_depth, desc="Aligning batch", ncols=100):
        process_pair(idx)

    print("Batch complete! Full metadata stored in metadata.jsonl")
