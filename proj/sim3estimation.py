import os
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from align_teaser import sim3_teaser_pipeline
import json
import argparse
import time
import copy

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
    "depth": DATA_ROOT / "depth",  # Depth Anything V2 (backward compatible)
    "depth_v3": DATA_ROOT / "depth_depth_anything_v3",  # Depth Anything V3
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
# Per-method camera voxel sizes (adjust based on depth quality)
CAMERA_PCD_VOXEL_SIZES = {
    "depth": 1e-3,
    "depth_v3": 1e-3,
    "marigold": 1e-3,
    "midas": 1e-3,
    "midas_small": 1e-3,
    "depth_pro": 1e-3,
}
# Default fallback
CAMERA_PCD_VOXEL_SIZE = 1e-3
LIDAR_PCD_VOXEL_SIZE = 0.2 # 1.0 default
MAX_POINTS_AFTER_DOWNSAMPLE = 5000  # Cap point count after downsampling for guaranteed runtime

# Depth range filtering parameters
# For metric methods (depth_pro, depth_v3): use fixed range in meters
DEPTH_RANGE_METRIC = {
    "MIN_Z": 0.1,   # Minimum depth in meters (10cm)
    "MAX_Z": 100.0, # Maximum depth in meters (100m)
}
# For relative methods: use percentile clipping
DEPTH_RANGE_PERCENTILES = {
    "MIN_PERCENTILE": 1.0,  # 1st percentile
    "MAX_PERCENTILE": 95.0, # 95th percentile
}

# Per-method default depth modes
DEFAULT_DEPTH_MODES = {
    "depth": "raw",           # Depth Anything v2: raw (percentile clip)
    "depth_v3": "raw",        # Depth Anything v3: raw (metric range clip)
    "midas": "raw",           # MiDaS: raw (relative depth; percentile clip)
    "midas_small": "raw",     # MiDaS Small: raw (relative depth; percentile clip)
    "depth_pro": "raw",       # Depth Pro: raw (metric clip)
    "marigold": "raw",        # Marigold: configurable (default raw)
}



def preprocess_depth(depth: np.ndarray, method: str, depth_mode: str = "auto") -> np.ndarray:
    """
    Preprocess depth map based on method and mode.
    
    Args:
        depth: Raw depth array (H, W)
        method: Depth method name (e.g., "depth", "depth_pro", "marigold")
        depth_mode: Processing mode - "auto", "raw", "inv", "one_minus_inv"
    
    Returns:
        Preprocessed depth array (metric depth in meters, ready for backprojection)
    """
    depth = depth.astype(np.float32).copy()
    
    # Mask invalid values first
    valid_mask = np.isfinite(depth) & (depth > 0)
    
    # Method-specific masking
    if method == "depth_pro":
        # Depth Pro has saturation cap at 10000
        valid_mask = valid_mask & (depth < 9999)
    elif method == "marigold":
        # Marigold should be in [0, 1] if normalized
        if depth_mode == "auto" or depth_mode == "one_minus_inv":
            # Check if values are in [0, 1] range
            if depth.max() <= 1.0 and depth.min() >= 0:
                valid_mask = valid_mask & (depth >= 0) & (depth <= 1.0)
            else:
                # Fall back to treating as inverse depth
                if depth_mode == "auto":
                    depth_mode = "inv"
    
    # Apply valid mask
    depth[~valid_mask] = np.nan
    
    # Determine processing mode
    if depth_mode == "auto":
        # Use per-method defaults
        depth_mode = DEFAULT_DEPTH_MODES.get(method, "raw")
    
    # Apply preprocessing
    if depth_mode == "raw":
        # Use depth directly (already metric)
        z = depth.copy()
    elif depth_mode == "inv":
        # Inverse depth: z = 1 / depth
        z = np.zeros_like(depth)
        valid = np.isfinite(depth) & (depth > 1e-6)
        z[valid] = 1.0 / depth[valid]
        z[~valid] = np.nan
    elif depth_mode == "one_minus_inv":
        # For normalized inverse depth: z = 1 / (1 - depth)
        # This handles visualization outputs where larger values = closer
        z = np.zeros_like(depth)
        valid = np.isfinite(depth) & (depth >= 0) & (depth < 1.0)
        z[valid] = 1.0 / (1.0 - depth[valid] + 1e-6)  # Add small epsilon to avoid division by zero
        z[~valid] = np.nan
    else:
        raise ValueError(f"Unknown depth_mode: {depth_mode}")
    
    return z


def backproject_depth(depth_img, voxel_size=None, method=None, verbose=False):
    """
    Backproject depth image to point cloud.
    
    Args:
        depth_img: Preprocessed depth image (metric depth in meters)
        voxel_size: Voxel size for downsampling (if None, uses default)
        method: Depth method name (for method-aware filtering)
        verbose: If True, print filtering statistics
    
    Returns:
        pcd: Open3D point cloud
        n_pre: Point count before downsampling
        n_post: Point count after downsampling
        n_filtered: Number of points removed by depth range filter
    """
    if voxel_size is None:
        voxel_size = CAMERA_PCD_VOXEL_SIZE
    
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    depth_flat = depth_img.flatten().astype(np.float32)
    u_flat = u.flatten()
    v_flat = v.flatten()

    # Mask invalid values (NaN, inf, non-positive)
    valid = np.isfinite(depth_flat) & (depth_flat > 1e-6)
    
    if np.sum(valid) == 0:
        # Return empty point cloud
        pcd = o3d.geometry.PointCloud()
        return pcd, 0, 0, 0
    
    depth_valid = depth_flat[valid]
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]

    # Backproject to 3D
    X = (u_valid - cx) * depth_valid / fx
    Y = (v_valid - cy) * depth_valid / fy
    Z = depth_valid

    pts = np.vstack((X, Y, Z)).T
    n_before_filter = len(pts)

    # Debug: Print Z percentiles before filtering
    if verbose:
        valid_z = Z[Z > 1e-6]
        if len(valid_z) > 0:
            z_percentiles = np.percentile(valid_z, [1, 5, 25, 50, 75, 95, 99])
            print(f"    Z percentiles before filtering: [1, 5, 25, 50, 75, 95, 99] = {z_percentiles}")

    # Apply method-aware depth range filtering
    if method in ["depth_pro", "depth_v3"]:
        # Metric methods: use fixed range in meters
        MIN_Z = DEPTH_RANGE_METRIC["MIN_Z"]
        MAX_Z = DEPTH_RANGE_METRIC["MAX_Z"]
        valid_pts = (Z > MIN_Z) & (Z < MAX_Z)
        if verbose:
            print(f"    Depth range filter (metric): {MIN_Z:.2f}m < Z < {MAX_Z:.2f}m")
    else:
        # Relative methods: use percentile clipping
        MIN_PERCENTILE = DEPTH_RANGE_PERCENTILES["MIN_PERCENTILE"]
        MAX_PERCENTILE = DEPTH_RANGE_PERCENTILES["MAX_PERCENTILE"]
        MIN_Z = np.percentile(Z, MIN_PERCENTILE)
        MAX_Z = np.percentile(Z, MAX_PERCENTILE)
        valid_pts = (Z > MIN_Z) & (Z < MAX_Z)
        if verbose:
            print(f"    Depth range filter (percentile): {MIN_PERCENTILE}th-{MAX_PERCENTILE}th percentile")
            print(f"    Effective range: {MIN_Z:.6f}m < Z < {MAX_Z:.6f}m")
    
    pts = pts[valid_pts]
    n_filtered = n_before_filter - len(pts)
    
    if verbose:
        print(f"    Points before depth filter: {n_before_filter}")
        print(f"    Points removed by depth filter: {n_filtered}")
        print(f"    Points after depth filter: {len(pts)}")

    if len(pts) == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd, 0, 0, n_filtered

    # Keep points in standard camera coordinates (X right, Y down, Z forward)
    # Do not flip axes here - any frame transforms should be done explicitly and consistently

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Downsample to reduce point count for faster processing
    n_pre = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    n_post = len(pcd.points)
    
    # Cap point count for guaranteed runtime
    if n_post > MAX_POINTS_AFTER_DOWNSAMPLE:
        # Randomly sample to max points
        indices = np.random.choice(n_post, MAX_POINTS_AFTER_DOWNSAMPLE, replace=False)
        pcd = pcd.select_by_index(indices)
        n_post = MAX_POINTS_AFTER_DOWNSAMPLE

    if verbose:
        print(f"    Points kept after depth filtering: {len(pts)}")
        print(f"    Final point count after downsampling: {n_post}")
    
    return pcd, n_pre, n_post, n_filtered


def extract_fpfh_features(pcd, voxel_size):
    """
    Extract FPFH features from a point cloud.
    
    Args:
        pcd: Open3D point cloud
        voxel_size: Voxel size for feature radius calculation
    
    Returns:
        FPFH features as numpy array (N, 33)
    """
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T


def find_correspondences_fpfh(feats0, feats1, mutual_filter=True):
    """
    Find correspondences using FPFH features via nearest neighbor search.
    Uses chunked numpy computation to avoid scipy dependency and memory issues.
    
    Args:
        feats0: Source features (N, 33)
        feats1: Target features (M, 33)
        mutual_filter: If True, apply mutual nearest neighbor filter
    
    Returns:
        corres_idx0, corres_idx1: Corresponding indices
    """
    # Find nearest neighbors: feats0 -> feats1
    # Use chunked computation to avoid memory issues with large feature sets
    chunk_size = 1000
    n0 = len(feats0)
    nn_inds01 = np.zeros(n0, dtype=np.int64)
    
    for i in range(0, n0, chunk_size):
        end_idx = min(i + chunk_size, n0)
        chunk = feats0[i:end_idx]
        # Compute squared distances: (chunk_size, M)
        dists = np.sum((chunk[:, np.newaxis, :] - feats1[np.newaxis, :, :]) ** 2, axis=2)
        nn_inds01[i:end_idx] = np.argmin(dists, axis=1)
    
    corres01_idx0 = np.arange(len(feats0))
    corres01_idx1 = nn_inds01
    
    if not mutual_filter:
        return corres01_idx0, corres01_idx1
    
    # Mutual filter: also check feats1 -> feats0
    n1 = len(feats1)
    nn_inds10 = np.zeros(n1, dtype=np.int64)
    
    for i in range(0, n1, chunk_size):
        end_idx = min(i + chunk_size, n1)
        chunk = feats1[i:end_idx]
        dists = np.sum((chunk[:, np.newaxis, :] - feats0[np.newaxis, :, :]) ** 2, axis=2)
        nn_inds10[i:end_idx] = np.argmin(dists, axis=1)
    
    corres10_idx1 = np.arange(len(feats1))
    corres10_idx0 = nn_inds10
    
    # Keep only mutual correspondences
    # A correspondence (i, j) is mutual if:
    # - j is the nearest neighbor of i in feats1, AND
    # - i is the nearest neighbor of j in feats0
    mutual_mask = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_mask]
    corres_idx1 = corres01_idx1[mutual_mask]
    
    return corres_idx0, corres_idx1


def process_pair(idx, depth_dir, depth_method, meta_path, verbose=False, estimate_scaling=True, depth_mode="auto", is_first_frame=False):
    """
    Process a single depth-LiDAR pair.
    
    Args:
        idx: Index string (e.g., "00000")
        depth_dir: Path to depth directory
        depth_method: Name of depth method (e.g., "marigold")
        meta_path: Path to metadata file
        verbose: If True, print detailed progress
        depth_mode: Depth preprocessing mode ("auto", "raw", "inv", "one_minus_inv")
        is_first_frame: If True, print detailed statistics (for test mode)
    """
    if verbose:
        print(f"  [{idx}] Loading depth and LiDAR files...")
    
    depth_path = depth_dir / f"depth_{idx}.npy"
    lidar_path = LIDAR_DIR / f"lidar_{idx}.pcd"

    if not depth_path.exists() or not lidar_path.exists():
        print(f"[SKIP] Missing pair for index {idx} (method: {depth_method})")
        return None

    # Load raw depth
    depth_raw = np.load(depth_path)
    
    # Print raw statistics for first frame in test mode
    if is_first_frame:
        print(f"\n  [{idx}] Raw depth statistics:")
        print(f"    Shape: {depth_raw.shape}, Dtype: {depth_raw.dtype}")
        print(f"    Min:   {depth_raw.min():.6f}")
        print(f"    Max:   {depth_raw.max():.6f}")
        print(f"    Mean:  {depth_raw.mean():.6f}")
        percentiles = np.percentile(depth_raw, [1, 5, 95, 99])
        print(f"    Percentiles [1, 5, 95, 99]: {percentiles}")
    
    # Preprocess depth based on method
    depth_processed = preprocess_depth(depth_raw, depth_method, depth_mode)
    
    # Print post-preprocess statistics for first frame
    if is_first_frame:
        valid_processed = np.isfinite(depth_processed) & (depth_processed > 0)
        n_valid = np.sum(valid_processed)
        print(f"  [{idx}] Post-preprocess statistics:")
        if n_valid > 0:
            depth_valid = depth_processed[valid_processed]
            print(f"    Valid pixels: {n_valid} / {depth_processed.size}")
            print(f"    Min:   {depth_valid.min():.6f}")
            print(f"    Max:   {depth_valid.max():.6f}")
            print(f"    Mean:  {depth_valid.mean():.6f}")
            z_percentiles = np.percentile(depth_valid, [1, 5, 25, 50, 75, 95, 99])
            print(f"    Z percentiles [1, 5, 25, 50, 75, 95, 99]: {z_percentiles}")
        else:
            print(f"    WARNING: No valid pixels after preprocessing!")
    
    if verbose:
        print(f"  [{idx}] Depth shape: {depth_processed.shape}, backprojecting to point cloud...")
    
    # Get method-specific voxel size
    voxel_size = CAMERA_PCD_VOXEL_SIZES.get(depth_method, CAMERA_PCD_VOXEL_SIZE)
    
    # Backproject with method-aware filtering
    pcd_cam, n_cam_pre, n_cam_post, n_filtered = backproject_depth(
        depth_processed, 
        voxel_size=voxel_size,
        method=depth_method,
        verbose=is_first_frame
    )
    
    if len(pcd_cam.points) == 0:
        print(f"  [{idx}] SKIP: Empty point cloud after backprojection")
        return None
    
    points_cam = np.asarray(pcd_cam.points).copy()
    
    print(f"  [{idx}] cam points pre-voxel: {n_cam_pre}")
    print(f"  [{idx}] cam points post-voxel: {n_cam_post}")
    if is_first_frame and n_filtered > 0:
        print(f"  [{idx}] cam points removed by depth filter: {n_filtered}")
    
    n_cam = len(pcd_cam.points)
    if n_cam < 100:
        print(f"  [{idx}] SKIP: too few camera points (cam={n_cam})")
        return None

    pcd_lidar = o3d.io.read_point_cloud(str(lidar_path))
    
    n_lidar_pre = len(pcd_lidar.points)
    if verbose:
        print(f"  [{idx}] LiDAR PCD: {n_lidar_pre} points, processing axes...")

    # [CHECK THE AXIS MANUALLY] post-process to get x-y-z axis
    pts_lidar_raw = np.asarray(pcd_lidar.points)[:, [1, 2, 0]] # Remap axes: (y, z, x) --> (x, y, z)
    pts_lidar_raw[:, 0] = -pts_lidar_raw[:, 0] # invert x axis

    pcd_lidar.points = o3d.utility.Vector3dVector(pts_lidar_raw)

    # Downsample to reduce point count for faster processing
    print(f"  [{idx}] lidar points pre-voxel: {n_lidar_pre}")
    pcd_lidar = pcd_lidar.voxel_down_sample(voxel_size=LIDAR_PCD_VOXEL_SIZE)
    n_lidar_post = len(pcd_lidar.points)
    
    # Cap point count for guaranteed runtime
    if n_lidar_post > MAX_POINTS_AFTER_DOWNSAMPLE:
        indices = np.random.choice(n_lidar_post, MAX_POINTS_AFTER_DOWNSAMPLE, replace=False)
        pcd_lidar = pcd_lidar.select_by_index(indices)
        n_lidar_post = MAX_POINTS_AFTER_DOWNSAMPLE
    
    print(f"  [{idx}] lidar points post-voxel: {n_lidar_post}")
    
    # Extract LiDAR points AFTER downsampling/capping to match the point cloud
    pts_lidar = np.asarray(pcd_lidar.points).copy()
    
    n_lidar = len(pcd_lidar.points)
    if n_lidar < 100:
        print(f"  [{idx}] SKIP: too few LiDAR points (lidar={n_lidar})")
        return None
    
    # Guard: ensure we have enough points before TEASER++
    if n_cam < 100 or n_lidar < 100:
        print(f"  [{idx}] SKIP: too few points (cam={n_cam}, lidar={n_lidar})")
        return None
    
    if verbose:
        print(f"  [{idx}] Extracting FPFH features for correspondence matching...")
    
    # Extract FPFH features for both point clouds
    start_time = time.time()
    cam_voxel_size = CAMERA_PCD_VOXEL_SIZES.get(depth_method, CAMERA_PCD_VOXEL_SIZE)
    feats_cam = extract_fpfh_features(pcd_cam, cam_voxel_size)
    feats_lidar = extract_fpfh_features(pcd_lidar, LIDAR_PCD_VOXEL_SIZE)
    
    # Find correspondences using FPFH features
    corres_idx0, corres_idx1 = find_correspondences_fpfh(feats_cam, feats_lidar, mutual_filter=True)
    n_corr = len(corres_idx0)
    feature_time = time.time() - start_time
    
    if verbose:
        print(f"  [{idx}] FPFH feature extraction and matching: {feature_time:.3f}s")
        print(f"  [{idx}] Found {n_corr} correspondences (mutual nearest neighbor)")
    
    if n_corr < 3:
        print(f"  [{idx}] SKIP: too few correspondences (n={n_corr}, need >= 3)")
        return None
    
    # Extract corresponding points
    points_cam_corr = np.asarray(pcd_cam.points)[corres_idx0]
    pts_lidar_corr = pts_lidar[corres_idx1]
    
    if verbose:
        print(f"  [{idx}] Running TEASER++ alignment with {n_corr} correspondences...")
    
    # Run TEASER++ with feature-based correspondences
    teaser_start = time.time()
    cam_aligned, metadata = sim3_teaser_pipeline(
        pcd_cam,
        pcd_lidar,
        0.05,
        points_cam_corr,
        pts_lidar_corr,
        verbose=verbose,
        estimate_scaling=estimate_scaling
    )
    teaser_time = time.time() - teaser_start
    
    # Calculate inlier ratio (points within noise_bound after transformation)
    if 'final_T' in metadata:
        # Transform camera correspondences using final transformation
        T = metadata['final_T']
        R = T[:3, :3]
        t = T[:3, 3]
        # Apply transformation: transformed = R @ points.T + t
        points_cam_corr_transformed = (R @ points_cam_corr.T).T + t
        # Compute distances between transformed camera points and LiDAR correspondences
        dists = np.linalg.norm(points_cam_corr_transformed - pts_lidar_corr, axis=1)
        noise_bound = 0.05
        inlier_ratio = np.sum(dists < noise_bound * 1.5) / len(dists) if len(dists) > 0 else 0.0
        metadata['correspondences_count'] = n_corr
        metadata['inlier_ratio'] = float(inlier_ratio)
        metadata['teaser_runtime'] = teaser_time
        metadata['feature_extraction_time'] = feature_time
        
        if verbose:
            print(f"  [{idx}] TEASER++ runtime: {teaser_time:.3f}s")
            print(f"  [{idx}] Inlier ratio: {inlier_ratio:.4f} ({np.sum(dists < noise_bound * 1.5)}/{n_corr})")
    else:
        metadata['correspondences_count'] = n_corr
        metadata['teaser_runtime'] = teaser_time
        metadata['feature_extraction_time'] = feature_time
    
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


def process_depth_method(depth_method, depth_dir, limit=None, test_mode=False, estimate_scaling=True, depth_mode_dict=None):
    """
    Process all files for a given depth method.
    
    Args:
        depth_method: Name of depth method (e.g., "marigold")
        depth_dir: Path to depth directory
        limit: Optional limit on number of files to process (for testing)
        test_mode: If True, only process first 3 files
        estimate_scaling: Whether to estimate scale in TEASER++
        depth_mode_dict: Dict mapping method names to depth modes (or None for auto)
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
    
    # Get depth mode for this method
    if depth_mode_dict and depth_method in depth_mode_dict:
        depth_mode = depth_mode_dict[depth_method]
    else:
        depth_mode = "auto"
    
    processed = 0
    for i, idx in enumerate(tqdm(all_indices, desc=f"Aligning {depth_method}", ncols=100, disable=test_mode)):
        is_first = (i == 0) and test_mode
        result = process_pair(
            idx, depth_dir, depth_method, meta_path, 
            verbose=test_mode, 
            estimate_scaling=estimate_scaling,
            depth_mode=depth_mode,
            is_first_frame=is_first
        )
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
        help="Depth methods to process (default: all). Options: depth, depth_v3, marigold, midas, midas_small, depth_pro, all"
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
    parser.add_argument(
        "--depth-mode",
        type=str,
        default=None,
        help="Per-method depth mode override in format METHOD1=mode1,METHOD2=mode2 (e.g., 'depth=raw,marigold=inv'). Modes: auto, raw, inv, one_minus_inv"
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
    
    # Parse per-method depth mode overrides
    depth_mode_dict = {}
    if args.depth_mode:
        print(f"DEPTH MODE OVERRIDES: {args.depth_mode}")
        for override in args.depth_mode.split(','):
            if '=' in override:
                method_name, mode = override.strip().split('=')
                method_name = method_name.strip()
                mode = mode.strip()
                if method_name in DEPTH_DIRS:
                    if mode in ["auto", "raw", "inv", "one_minus_inv"]:
                        depth_mode_dict[method_name] = mode
                        print(f"  {method_name} -> {mode}")
                    else:
                        print(f"  WARNING: Invalid mode '{mode}' for {method_name}, ignoring")
                else:
                    print(f"  WARNING: Unknown method '{method_name}', ignoring")
            else:
                print(f"  WARNING: Invalid format '{override}', expected METHOD=mode")
    
    # Print default depth modes
    if not depth_mode_dict:
        print("DEPTH MODES: Using method-specific defaults")
        for method in methods_to_process:
            default_mode = DEFAULT_DEPTH_MODES.get(method, "raw")
            print(f"  {method} -> {default_mode} (default)")
    
    estimate_scaling = not args.no_scale
    
    total_processed = 0
    for method in methods_to_process:
        depth_dir = DEPTH_DIRS[method]
        processed = process_depth_method(
            method, 
            depth_dir, 
            limit=args.limit,
            test_mode=args.test,
            estimate_scaling=estimate_scaling,
            depth_mode_dict=depth_mode_dict
        )
        total_processed += processed
    
    print(f"\n{'='*60}")
    print(f"Batch complete! Total files processed: {total_processed}")
    print(f"Metadata files stored in: {OUTPUT_DIR}/metadata_*.jsonl")
    print(f"Aligned point clouds stored in: {OUTPUT_DIR}/cam_aligned_*.ply")
