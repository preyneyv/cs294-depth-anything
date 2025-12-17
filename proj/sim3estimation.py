import os
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from align_teaser import sim3_teaser_pipeline
import json
import argparse

# === Batch SIM(3) Alignment Pipeline ===
# Folder structure expected:
# dataset/
#   depth/
#   depth_marigold/
#   depth_midas/
#   depth_midas_small/
#   depth_depth_pro/
#     depth_00001.npy, depth_00002.npy, ...
#   lidar/
#     lidar_00001.pcd, lidar_00002.pcd, ...

DATA_ROOT = Path("../dataset")
LIDAR_DIR = DATA_ROOT / "lidar"
OUTPUT_DIR = DATA_ROOT / "aligned"
OUTPUT_DIR.mkdir(exist_ok=True)

# All available depth directories
DEPTH_DIRS = {
    "depth": DATA_ROOT / "depth",
    "marigold": DATA_ROOT / "depth_marigold",
    "midas": DATA_ROOT / "depth_midas",
    "midas_small": DATA_ROOT / "depth_midas_small",
    "depth_pro": DATA_ROOT / "depth_depth_pro",
}

# Camera intrinsics
K = np.array([
    [491.331107883326, 0.0, 515.3434363622374],
    [0.0, 492.14998153326013, 388.93983736974667],
    [0.0, 0.0, 1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Point cloud downsampling parameters 
# CAMERA_PCD_VOXEL_SIZE = 1e-4 # earlier default setting
# CAMERA_PCD_VOXEL_SIZE = 0.2 # for regular eval
CAMERA_PCD_VOXEL_SIZE = 0.2 # for fast test
LIDAR_PCD_VOXEL_SIZE = 2.0 # 1.0 default
MAX_POINTS_AFTER_DOWNSAMPLE = 5000  # Cap point count after downsampling for guaranteed runtime



def backproject_depth(depth_img):
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    depth_flat = depth_img.flatten().astype(np.float32)
    u_flat = u.flatten()
    v_flat = v.flatten()

    depth_flat = np.clip(depth_flat, 1e-6, None) # prevent division by zero errors
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

    # Downsample to reduce point count for faster processing
    n_pre = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=CAMERA_PCD_VOXEL_SIZE)
    n_post = len(pcd.points)
    
    # Cap point count for guaranteed runtime
    if n_post > MAX_POINTS_AFTER_DOWNSAMPLE:
        # Randomly sample to max points
        indices = np.random.choice(n_post, MAX_POINTS_AFTER_DOWNSAMPLE, replace=False)
        pcd = pcd.select_by_index(indices)
        n_post = MAX_POINTS_AFTER_DOWNSAMPLE

    return pcd, n_pre, n_post


def process_pair(idx, depth_dir, depth_method, meta_path, verbose=False, estimate_scaling=True):
    """
    Process a single depth-LiDAR pair.
    
    Args:
        idx: Index string (e.g., "00000")
        depth_dir: Path to depth directory
        depth_method: Name of depth method (e.g., "marigold")
        meta_path: Path to metadata file
        verbose: If True, print detailed progress
    """
    if verbose:
        print(f"  [{idx}] Loading depth and LiDAR files...")
    
    depth_path = depth_dir / f"depth_{idx}.npy"
    lidar_path = LIDAR_DIR / f"lidar_{idx}.pcd"

    if not depth_path.exists() or not lidar_path.exists():
        print(f"[SKIP] Missing pair for index {idx} (method: {depth_method})")
        return None

    depth = np.load(depth_path)
    if verbose:
        print(f"  [{idx:05d}] Depth shape: {depth.shape}, backprojecting to point cloud...")
    
    pcd_cam, n_cam_pre, n_cam_post = backproject_depth(depth)
    points_cam = np.asarray(pcd_cam.points).copy()
    
    print(f"  [{idx:05d}] cam points pre-voxel: {n_cam_pre}")
    print(f"  [{idx:05d}] cam points post-voxel: {n_cam_post}")
    
    n_cam = len(pcd_cam.points)
    if n_cam < 100:
        print(f"  [{idx:05d}] SKIP: too few camera points (cam={n_cam})")
        return None

    pcd_lidar = o3d.io.read_point_cloud(str(lidar_path))
    
    n_lidar_pre = len(pcd_lidar.points)
    if verbose:
        print(f"  [{idx:05d}] LiDAR PCD: {n_lidar_pre} points, processing axes...")

    # [CHECK THE AXIS MANUALLY] post-process to get x-y-z axis
    pts_lidar = np.asarray(pcd_lidar.points)[:, [1, 2, 0]] # Remap axes: (y, z, x) --> (x, y, z)
    pts_lidar[:, 0] = -pts_lidar[:, 0] # invert x axis

    pcd_lidar.points = o3d.utility.Vector3dVector(pts_lidar)

    # Downsample to reduce point count for faster processing
    print(f"  [{idx:05d}] lidar points pre-voxel: {n_lidar_pre}")
    pcd_lidar = pcd_lidar.voxel_down_sample(voxel_size=LIDAR_PCD_VOXEL_SIZE)
    n_lidar_post = len(pcd_lidar.points)
    
    # Cap point count for guaranteed runtime
    if n_lidar_post > MAX_POINTS_AFTER_DOWNSAMPLE:
        indices = np.random.choice(n_lidar_post, MAX_POINTS_AFTER_DOWNSAMPLE, replace=False)
        pcd_lidar = pcd_lidar.select_by_index(indices)
        n_lidar_post = MAX_POINTS_AFTER_DOWNSAMPLE
    
    print(f"  [{idx:05d}] lidar points post-voxel: {n_lidar_post}")
    
    n_lidar = len(pcd_lidar.points)
    if n_lidar < 100:
        print(f"  [{idx:05d}] SKIP: too few LiDAR points (lidar={n_lidar})")
        return None
    
    # Guard: ensure we have enough points before TEASER++
    if n_cam < 100 or n_lidar < 100:
        print(f"  [{idx:05d}] SKIP: too few points (cam={n_cam}, lidar={n_lidar})")
        return None
    
    # Guard: ensure at least 3 correspondences (minimum for 3D transformation)
    n_corr = min(len(points_cam), len(pts_lidar))
    if n_corr < 3:
        print(f"  [{idx:05d}] SKIP: too few correspondences (n={n_corr}, need >= 3)")
        return None
    
    if verbose:
        print(f"  [{idx:05d}] Running TEASER++ alignment (this may take a moment)...")

    cam_aligned, metadata = sim3_teaser_pipeline(
        pcd_cam,
        pcd_lidar,
        0.05,
        points_cam,
        pts_lidar,
        verbose=verbose,
        estimate_scaling=estimate_scaling
    )
    
    if verbose:
        print(f"  [{idx}] TEASER++ complete: scale={metadata['s']:.4f}, fitness={metadata['fitness']:.4f}, RMSE={metadata['rmse']:.6f}")
        print(f"  [{idx}] Saving aligned point cloud...")

    # Save with method name to avoid conflicts
    save_path = OUTPUT_DIR / f"cam_aligned_{depth_method}_{idx}.ply"
    o3d.io.write_point_cloud(str(save_path), cam_aligned)

    metadata["index"] = idx
    metadata["depth_method"] = depth_method
    with open(meta_path, "a") as f:
        f.write(json.dumps(metadata, default=str) + "\n")

    if verbose:
        print(f"  [{idx}] ✓ Complete: {save_path.name}")
    
    return save_path


def process_depth_method(depth_method, depth_dir, limit=None, test_mode=False, estimate_scaling=True):
    """
    Process all files for a given depth method.
    
    Args:
        depth_method: Name of depth method (e.g., "marigold")
        depth_dir: Path to depth directory
        limit: Optional limit on number of files to process (for testing)
        test_mode: If True, only process first 3 files
    """
    print(f"\n{'='*60}")
    print(f"Processing depth method: {depth_method}")
    print(f"{'='*60}")
    
    if not depth_dir.exists():
        print(f"[SKIP] Directory does not exist: {depth_dir}")
        return 0
    
    # Find all depth files
    all_depth_files = sorted(depth_dir.glob("depth_*.npy"))
    if not all_depth_files:
        print(f"[SKIP] No depth files found in {depth_dir}")
        return 0
    
    # Extract indices
    all_indices = sorted([f.name.split("_")[1].split(".")[0] for f in all_depth_files])
    
    # Apply limits for testing
    if test_mode:
        all_indices = all_indices[:3]
        print(f"[TEST MODE] Processing only first 3 files for {depth_method}")
    elif limit:
        all_indices = all_indices[:limit]
        print(f"[LIMIT] Processing only first {limit} files for {depth_method}")
    
    # Create method-specific metadata file
    meta_path = OUTPUT_DIR / f"metadata_{depth_method}.jsonl"
    with open(meta_path, "w") as f:
        pass  # Clear/create file
    
    processed = 0
    for idx in tqdm(all_indices, desc=f"Aligning {depth_method}", ncols=100, disable=test_mode):
        result = process_pair(idx, depth_dir, depth_method, meta_path, verbose=test_mode, estimate_scaling=estimate_scaling)
        if result:
            processed += 1
    
    print(f"[DONE] {depth_method}: {processed}/{len(all_indices)} files processed")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch SIM(3) alignment for multiple depth estimation methods"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(DEPTH_DIRS.keys()) + ["all"],
        default=["all"],
        help="Depth methods to process (default: all). Options: depth, marigold, midas, midas_small, depth_pro, all"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process per method (for testing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first 3 files per method"
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable scale estimation in TEASER++ (faster, more stable for debugging)"
    )
    args = parser.parse_args()
    
    # Determine which methods to process
    if "all" in args.methods:
        methods_to_process = list(DEPTH_DIRS.keys())
    else:
        methods_to_process = args.methods
    
    print(f"Processing depth methods: {', '.join(methods_to_process)}")
    if args.test:
        print("TEST MODE: Processing only first 3 files per method")
    elif args.limit:
        print(f"LIMIT MODE: Processing only first {args.limit} files per method")
    if args.no_scale:
        print("SCALE ESTIMATION: DISABLED (faster, more stable)")
    
    estimate_scaling = not args.no_scale
    
    total_processed = 0
    for method in methods_to_process:
        depth_dir = DEPTH_DIRS[method]
        processed = process_depth_method(
            method, 
            depth_dir, 
            limit=args.limit,
            test_mode=args.test,
            estimate_scaling=estimate_scaling
        )
        total_processed += processed
    
    print(f"\n{'='*60}")
    print(f"Batch complete! Total files processed: {total_processed}")
    print(f"Metadata files stored in: {OUTPUT_DIR}/metadata_*.jsonl")
    print(f"Aligned point clouds stored in: {OUTPUT_DIR}/cam_aligned_*.ply")
