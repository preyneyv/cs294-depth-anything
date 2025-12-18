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
# Initial voxel sizes for adaptive downsampling (per method) - single source of truth
CAMERA_PCD_VOXEL_SIZES = {
    "depth": 0.05,      # Dense relative method
    "depth_v3": 0.05,   # Dense relative method (or metric if raw mode)
    "marigold": 0.05,   # Dense relative method
    "midas": 0.05,      # Dense relative method
    "midas_small": 0.05, # Dense relative method
    "depth_pro": 0.05,  # Metric method - keep at 1e-3 to preserve behavior
}
# Default fallback
CAMERA_PCD_VOXEL_SIZE = 1e-3
LIDAR_PCD_VOXEL_SIZE = 0.2 # 1.0 default

# Point count management (for debugging alignment quality)
# TARGET_MAX_POINTS: Active control - adaptive voxel sizing will iteratively increase voxel size
#                    until point count <= this value. This is what we actively try to achieve.
TARGET_MAX_POINTS = 20000

# MAX_POINTS_WARNING_THRESHOLD: Passive warning - if point count exceeds this AFTER adaptive sizing,
#                               prints a warning but takes no action. Should be higher than TARGET.
#                               Indicates the initial voxel size might be too small or data is very dense.
MAX_POINTS_WARNING_THRESHOLD = int(TARGET_MAX_POINTS * 1.25)

# Maximum camera voxel size cap to prevent destroying local geometry
MAX_CAM_VOXEL = 0.30  # meters - maximum voxel size for adaptive downsampling

# Use CAMERA_PCD_VOXEL_SIZES as the single source of truth for initial voxel sizes
CAMERA_VOXEL_INITIAL = CAMERA_PCD_VOXEL_SIZES

# Note: Camera cropping is now controlled via CLI arguments (--crop-z-min, --crop-z-max, --crop-fov-scale)
# LiDAR cropping is disabled in this version (focus on camera crop only)

# Feature extraction voxel size for RANSAC-based correspondence matching
# Use same size for both source and target clouds
FEATURE_VOXEL_SIZE = 0.05  # 0.05m for feature matching

# Depth range filtering parameters
# For metric methods (depth_pro, depth_v3): use fixed range in meters
DEPTH_RANGE_METRIC = {
    "MIN_Z": 0.1,   # Minimum depth in meters (10cm)
    "MAX_Z": 100.0, # Maximum depth in meters (100m) - can be overridden per method
}
# Per-method max depth overrides (for debugging)
DEPTH_RANGE_METRIC_OVERRIDE = {
    "depth_pro": 50.0,  # Limit depth_pro to 50m to match LiDAR forward range
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



def resolve_depth_mode(method: str, depth_mode: str) -> str:
    """
    Resolve depth mode from "auto" to actual mode.
    
    Args:
        method: Depth method name
        depth_mode: Processing mode - "auto", "raw", "inv", "one_minus_inv"
    
    Returns:
        Resolved depth mode (one of: "raw", "inv", "one_minus_inv")
    """
    if depth_mode == "auto":
        return DEFAULT_DEPTH_MODES.get(method, "raw")
    return depth_mode


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


def crop_camera_by_z(pcd, z_min, z_max):
    """
    Crop camera point cloud by depth (Z) range.
    
    Args:
        pcd: Open3D point cloud in camera coordinates
        z_min: Minimum Z (depth) in meters
        z_max: Maximum Z (depth) in meters
    
    Returns:
        Cropped point cloud
    """
    if len(pcd.points) == 0:
        return pcd
    
    pts = np.asarray(pcd.points)
    z_vals = pts[:, 2]  # Z is the third column (depth)
    mask = (z_vals > z_min) & (z_vals < z_max)
    
    return pcd.select_by_index(np.where(mask)[0])


def crop_camera_by_fov(pcd, fx, fy, fov_scale, verbose=False):
    """
    Crop camera point cloud by angular FOV using percentile-based approach.
    
    Args:
        pcd: Open3D point cloud in camera coordinates
        fx: Camera focal length in X
        fy: Camera focal length in Y
        fov_scale: Fraction of FOV to keep (0.6 = keep central 60%, 1.0 = keep all)
        verbose: If True, print thresholds used
    
    Returns:
        Cropped point cloud
    """
    if len(pcd.points) == 0 or fov_scale >= 1.0:
        return pcd
    
    pts = np.asarray(pcd.points)
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]
    
    # Only consider points with valid positive Z
    valid = Z > 1e-6
    if np.sum(valid) == 0:
        return pcd
    
    # Compute x_over_z and y_over_z for valid points
    x_over_z = X[valid] / Z[valid]
    y_over_z = Y[valid] / Z[valid]
    
    # Compute 95th percentile of |X/Z| and |Y/Z|
    p95_x = np.percentile(np.abs(x_over_z), 95)
    p95_y = np.percentile(np.abs(y_over_z), 95)
    
    # Thresholds: keep points within fov_scale * 95th percentile
    x_threshold = fov_scale * p95_x
    y_threshold = fov_scale * p95_y
    
    if verbose:
        print(f"    FOV crop: p95(|X/Z|)={p95_x:.4f}, p95(|Y/Z|)={p95_y:.4f}")
        print(f"    FOV thresholds: |X/Z| < {x_threshold:.4f}, |Y/Z| < {y_threshold:.4f}")
    
    # Apply FOV crop to all points
    x_over_z_all = X / Z
    y_over_z_all = Y / Z
    fov_mask = (np.abs(x_over_z_all) <= x_threshold) & (np.abs(y_over_z_all) <= y_threshold) & valid
    
    return pcd.select_by_index(np.where(fov_mask)[0])


def apply_camera_crops_metric(pcd_cam, crop_z_min, crop_z_max, crop_fov_scale, fx, fy, verbose=False):
    """
    Apply Z and optional FOV crops to camera point cloud (METRIC methods only).
    
    Args:
        pcd_cam: Camera point cloud
        crop_z_min: Minimum Z (depth) in meters
        crop_z_max: Maximum Z (depth) in meters
        crop_fov_scale: FOV scale factor (1.0 = no FOV crop)
        fx: Camera focal length in X
        fy: Camera focal length in Y
        verbose: If True, print crop statistics
    
    Returns:
        Cropped point cloud
    """
    n_before = len(pcd_cam.points)
    
    # Apply Z crop first
    pcd_cam = crop_camera_by_z(pcd_cam, crop_z_min, crop_z_max)
    n_after_z = len(pcd_cam.points)
    
    if verbose:
        print(f"    Camera crop: {n_before} -> {n_after_z} points (Z: {crop_z_min:.1f}m < Z < {crop_z_max:.1f}m)")
    
    # Apply FOV crop if requested
    if crop_fov_scale < 1.0:
        n_before_fov = n_after_z
        pcd_cam = crop_camera_by_fov(pcd_cam, fx, fy, crop_fov_scale, verbose=verbose)
        n_after_fov = len(pcd_cam.points)
        if verbose:
            print(f"    Camera crop: {n_before_fov} -> {n_after_fov} points (FOV scale: {crop_fov_scale:.2f})")
    
    return pcd_cam


def apply_camera_crops_relative(pcd_cam, rel_min_pct, rel_max_pct, crop_fov_scale, fx, fy, invert=False, verbose=False):
    """
    Apply Z and optional FOV crops to camera point cloud (RELATIVE methods only).
    Z crop is based on percentiles of Z values in the point cloud.
    
    Args:
        pcd_cam: Camera point cloud
        rel_min_pct: Minimum Z percentile (e.g., 2 = 2nd percentile)
        rel_max_pct: Maximum Z percentile (e.g., 95 = 95th percentile)
        crop_fov_scale: FOV scale factor (1.0 = no FOV crop)
        fx: Camera focal length in X
        fy: Camera focal length in Y
        invert: If True, invert the percentile interpretation (for "larger = closer" cases)
        verbose: If True, print crop statistics
    
    Returns:
        Cropped point cloud
    """
    n_before = len(pcd_cam.points)
    
    if len(pcd_cam.points) == 0:
        return pcd_cam
    
    # Get Z values from point cloud
    pts = np.asarray(pcd_cam.points)
    Z = pts[:, 2]  # Z is the third column (depth)
    valid_z = Z[Z > 1e-6]  # Only consider valid positive Z
    
    if len(valid_z) == 0:
        return pcd_cam
    
    # Compute percentile thresholds
    if invert:
        # For "larger value = closer": invert percentile interpretation
        # In this case, higher Z values = closer, so we swap the percentile interpretation
        # Keep points where Z is between (100-rel_max_pct) and (100-rel_min_pct) percentiles
        z_min_threshold = np.percentile(valid_z, 100 - rel_max_pct)
        z_max_threshold = np.percentile(valid_z, 100 - rel_min_pct)
    else:
        # Standard: lower percentile = closer, higher percentile = farther
        z_min_threshold = np.percentile(valid_z, rel_min_pct)
        z_max_threshold = np.percentile(valid_z, rel_max_pct)
    
    # Apply Z crop based on percentiles
    mask = (Z > z_min_threshold) & (Z < z_max_threshold)
    pcd_cam = pcd_cam.select_by_index(np.where(mask)[0])
    n_after_z = len(pcd_cam.points)
    
    if verbose:
        print(f"    Relative Z crop: pct[{rel_min_pct},{rel_max_pct}] => z_min={z_min_threshold:.6f}m, z_max={z_max_threshold:.6f}m, points {n_before} -> {n_after_z}")
    
    # Apply FOV crop if requested
    if crop_fov_scale < 1.0:
        n_before_fov = n_after_z
        pcd_cam = crop_camera_by_fov(pcd_cam, fx, fy, crop_fov_scale, verbose=verbose)
        n_after_fov = len(pcd_cam.points)
        if verbose:
            print(f"    Camera crop: {n_before_fov} -> {n_after_fov} points (FOV scale: {crop_fov_scale:.2f})")
    
    return pcd_cam


def backproject_depth(depth_img, fx, fy, cx, cy, method=None, is_metric_depth=False, verbose=False):
    """
    Backproject depth image to point cloud (no voxel downsampling - returns full cloud after depth filtering).
    
    Args:
        depth_img: Preprocessed depth image (metric depth in meters)
        fx: Camera focal length in X
        fy: Camera focal length in Y
        cx: Camera principal point X
        cy: Camera principal point Y
        method: Depth method name (for logging)
        is_metric_depth: If True, use metric depth range filtering; if False, use percentile filtering
        verbose: If True, print filtering statistics
    
    Returns:
        pcd: Open3D point cloud (after depth range filtering, no voxel downsampling)
        n_before_filter: Point count before depth range filter
        n_after_filter: Point count after depth range filter
        n_filtered: Number of points removed by depth range filter
    """
    
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

    # Apply depth range filtering based on is_metric_depth flag
    if is_metric_depth:
        # Metric methods: use fixed range in meters (with per-method override if available)
        MIN_Z = DEPTH_RANGE_METRIC["MIN_Z"]
        MAX_Z = DEPTH_RANGE_METRIC_OVERRIDE.get(method, DEPTH_RANGE_METRIC["MAX_Z"])
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
        return pcd, n_before_filter, 0, n_filtered

    # Keep points in standard camera coordinates (X right, Y down, Z forward)
    # Do not flip axes here - any frame transforms should be done explicitly and consistently

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    n_after_filter = len(pcd.points)
    
    if verbose:
        print(f"    Points kept after depth filtering: {n_after_filter}")
    
    return pcd, n_before_filter, n_after_filter, n_filtered


def adaptive_voxel_downsample(pcd, method=None, verbose=False):
    """
    Apply adaptive voxel downsampling to point cloud with max voxel cap.
    
    Args:
        pcd: Open3D point cloud
        method: Depth method name (for method-specific initial voxel size)
        verbose: If True, print statistics
    
    Returns:
        pcd_down: Downsampled point cloud
        final_voxel_size: Final voxel size used
        n_pre: Point count before downsampling
        n_post: Point count after downsampling
    """
    if len(pcd.points) == 0:
        return pcd, 0.0, 0, 0
    
    n_pre = len(pcd.points)
    
    # Get initial voxel size
    initial_voxel_size = CAMERA_VOXEL_INITIAL.get(method, CAMERA_PCD_VOXEL_SIZE)
    current_voxel_size = initial_voxel_size
    
    # Keep increasing voxel size until target is reached or max cap is hit
    # Do not stop at max_iterations - keep going until convergence or cap
    while True:
        pcd_down = pcd.voxel_down_sample(voxel_size=current_voxel_size)
        n_post = len(pcd_down.points)
        
        if n_post <= TARGET_MAX_POINTS:
            # Target reached, use this voxel size
            pcd = pcd_down
            break
        elif current_voxel_size >= MAX_CAM_VOXEL:
            # Hit max voxel cap, stop increasing
            pcd = pcd_down
            if verbose:
                print(f"    WARNING: Stopped at max voxel cap ({MAX_CAM_VOXEL:.3f}m) with {n_post} points (> {TARGET_MAX_POINTS})")
            break
        else:
            # Increase voxel size and try again (but don't exceed cap)
            next_voxel_size = current_voxel_size * 1.5
            if next_voxel_size > MAX_CAM_VOXEL:
                current_voxel_size = MAX_CAM_VOXEL
            else:
                current_voxel_size = next_voxel_size
            if verbose:
                print(f"    Voxel size {current_voxel_size/1.5:.6f}m -> {current_voxel_size:.6f}m (points: {n_post} > {TARGET_MAX_POINTS})")
    
    final_voxel_size = current_voxel_size
    n_post = len(pcd.points)
    
    # Warn if still over threshold (but don't cap)
    if n_post > MAX_POINTS_WARNING_THRESHOLD:
        print(f"    WARNING: camera cloud has {n_post} points (> {MAX_POINTS_WARNING_THRESHOLD}) after adaptive voxel sizing")
    
    if verbose:
        print(f"    Adaptive voxel downsampling: {n_pre} -> {n_post} points (voxel: {final_voxel_size:.6f}m)")
    
    return pcd, final_voxel_size, n_pre, n_post


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


def find_correspondences_ransac_fpfh(pcd_src, pcd_tgt, voxel_size, mutual_filter=True):
    """
    Find correspondences using Open3D RANSAC-based global registration on FPFH features.
    
    Args:
        pcd_src: Source point cloud (Open3D) - already downsampled
        pcd_tgt: Target point cloud (Open3D) - already downsampled
        voxel_size: Voxel size for feature extraction (clouds may already be downsampled)
        mutual_filter: If True, use mutual filter in RANSAC
    
    Returns:
        corr_src_pts: Corresponding source points (N, 3) from downsampled cloud
        corr_tgt_pts: Corresponding target points (N, 3) from downsampled cloud
        ransac_result: Open3D RANSAC registration result
    """
    # Only downsample if the clouds are too dense (more than 10000 points)
    # Otherwise use them as-is to preserve more points for feature matching
    if len(pcd_src.points) > 10000:
        pcd_src_down = pcd_src.voxel_down_sample(voxel_size)
    else:
        pcd_src_down = pcd_src
    if len(pcd_tgt.points) > 10000:
        pcd_tgt_down = pcd_tgt.voxel_down_sample(voxel_size)
    else:
        pcd_tgt_down = pcd_tgt
    
    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    radius_feature = voxel_size * 5
    fpfh_src = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh_tgt = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_tgt_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # RANSAC-based global registration
    # Use larger max_correspondence_distance to be less aggressive with filtering
    max_correspondence_distance = voxel_size * 2.0
    
    # Set up checkers - use less strict edge length checker to allow more correspondences
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),  # Less strict (was 0.9)
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)
    ]
    
    # RANSAC parameters
    ransac_n = 3  # Minimum correspondences for a hypothesis
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    
    # Run RANSAC with more iterations for better results
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_src_down, pcd_tgt_down,
        fpfh_src, fpfh_tgt,
        mutual_filter=mutual_filter,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=estimation_method,
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 500) #was 4000000
    )
    
    # Extract correspondences from RANSAC result
    if ransac_result.correspondence_set is None or len(ransac_result.correspondence_set) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), ransac_result
    
    # Get correspondence indices
    corres_indices = np.asarray(ransac_result.correspondence_set)
    src_indices = corres_indices[:, 0]
    tgt_indices = corres_indices[:, 1]
    
    # Extract corresponding points from downsampled clouds
    src_pts = np.asarray(pcd_src_down.points)
    tgt_pts = np.asarray(pcd_tgt_down.points)
    
    corr_src_pts = src_pts[src_indices]
    corr_tgt_pts = tgt_pts[tgt_indices]
    
    return corr_src_pts, corr_tgt_pts, ransac_result


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


def process_pair(idx, depth_dir, depth_method, meta_path, verbose=False, estimate_scaling=True, depth_mode="auto", is_first_frame=False,
                 crop_z_min=2.0, crop_z_max=50.0, crop_fov_scale=0.8, viz=False, viz_after=False, viz_frame_idx=0, k_scale=2.0,
                 rel_crop_min_pct=2, rel_crop_max_pct=95, rel_crop_invert=False):
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
    
    # Resolve effective depth mode and determine if this is metric depth
    effective_mode = resolve_depth_mode(depth_method, depth_mode)
    is_metric_depth = (depth_method in ["depth_pro", "depth_v3"]) and (effective_mode == "raw")
    
    # Log depth mode resolution for verification
    if is_first_frame or verbose:
        print(f"  [{idx}] Depth mode resolved: requested={depth_mode}, effective={effective_mode}, is_metric_depth={is_metric_depth}")
    
    # Preprocess depth based on method (pass resolved mode)
    depth_processed = preprocess_depth(depth_raw, depth_method, effective_mode)
    
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
    
    # Compute scaled intrinsics
    # Note: k_scale is passed as parameter to process_pair
    fx_eff = fx * k_scale
    fy_eff = fy * k_scale
    cx_eff = cx * k_scale
    cy_eff = cy * k_scale
    
    # Log effective intrinsics in test/verbose mode
    if is_first_frame or verbose:
        print(f"  [{idx}] Camera intrinsics:")
        print(f"    Original: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"    Scaled (k_scale={k_scale:.2f}): fx={fx_eff:.2f}, fy={fy_eff:.2f}, cx={cx_eff:.2f}, cy={cy_eff:.2f}")
    
    # Backproject depth (no voxel downsampling yet - returns full cloud after depth filtering)
    pcd_cam_full, n_cam_before_filter, n_cam_after_filter, n_filtered = backproject_depth(
        depth_processed,
        fx_eff, fy_eff, cx_eff, cy_eff,
        method=depth_method,
        is_metric_depth=is_metric_depth,
        verbose=is_first_frame
    )
    
    if len(pcd_cam_full.points) == 0:
        print(f"  [{idx}] SKIP: Empty point cloud after backprojection")
        return None
    
    # Visualization: raw clouds before any modifications
    should_viz_raw = viz and (is_first_frame or (not is_first_frame and idx == f"{viz_frame_idx:05d}"))
    if should_viz_raw:
        # Store raw camera cloud before cropping
        pcd_cam_raw = copy.deepcopy(pcd_cam_full)
    
    # Apply principled camera crops (Z range + optional FOV) using scaled intrinsics
    # Crop BEFORE adaptive voxel sizing to avoid absurd voxel sizes from far points
    # Route to metric or relative crop function based on is_metric_depth flag
    n_cam_pre_crop = len(pcd_cam_full.points)
    if is_metric_depth:
        # Metric methods: use absolute Z range in meters
        pcd_cam_crop = apply_camera_crops_metric(pcd_cam_full, crop_z_min, crop_z_max, crop_fov_scale, fx_eff, fy_eff, verbose=verbose or is_first_frame)
    else:
        # Relative methods: use percentile-based Z cropping
        pcd_cam_crop = apply_camera_crops_relative(pcd_cam_full, rel_crop_min_pct, rel_crop_max_pct, crop_fov_scale, fx_eff, fy_eff, invert=rel_crop_invert, verbose=verbose or is_first_frame)
    n_cam_post_crop = len(pcd_cam_crop.points)
    
    if n_cam_post_crop < 100:
        print(f"  [{idx}] SKIP: too few camera points after cropping (cam={n_cam_post_crop})")
        return None
    
    # Apply adaptive voxel downsampling on CROPPED cloud only
    pcd_cam, final_voxel_size, n_cam_pre_voxel, n_cam_post_voxel = adaptive_voxel_downsample(
        pcd_cam_crop, method=depth_method, verbose=verbose or is_first_frame
    )
    
    # Log point counts at each stage
    if verbose or is_first_frame:
        print(f"  [{idx}] Camera point cloud stages:")
        print(f"    Full backprojection: {n_cam_before_filter} points")
        print(f"    After depth filter: {n_cam_after_filter} points (removed {n_filtered})")
        print(f"    After crop: {n_cam_post_crop} points")
        print(f"    After adaptive voxel ({final_voxel_size:.6f}m): {n_cam_post_voxel} points")
    
    n_cam = len(pcd_cam.points)
    
    if n_cam < 100:
        print(f"  [{idx}] SKIP: too few camera points after adaptive voxel (cam={n_cam})")
        return None

    pcd_lidar = o3d.io.read_point_cloud(str(lidar_path))
    
    n_lidar_pre = len(pcd_lidar.points)
    if verbose:
        print(f"  [{idx}] LiDAR PCD: {n_lidar_pre} points, processing axes...")
    
    # Visualization: raw clouds before any modifications
    if should_viz_raw:
        # Store raw LiDAR cloud before axis remapping and downsampling
        pcd_lidar_raw_viz = copy.deepcopy(pcd_lidar)
        try:
            pcd_cam_raw_viz = copy.deepcopy(pcd_cam_raw)
            pcd_cam_raw_viz.paint_uniform_color([1.0, 0.0, 0.0])  # Red for raw camera
            pcd_lidar_raw_viz.paint_uniform_color([0.0, 1.0, 0.0])  # Green for raw LiDAR
            print(f"  [{idx}] Visualizing RAW clouds (before any modifications) (camera=red, LiDAR=green)...")
            o3d.visualization.draw_geometries([pcd_cam_raw_viz, pcd_lidar_raw_viz],
                                             window_name=f"Raw clouds (pre-processing) [{idx}]")
        except Exception as e:
            print(f"  [{idx}] WARNING: Raw visualization failed (headless mode?): {e}")

    # [CHECK THE AXIS MANUALLY] post-process to get x-y-z axis
    pts_lidar_raw = np.asarray(pcd_lidar.points)[:, [1, 2, 0]] # Remap axes: (y, z, x) --> (x, y, z)
    pts_lidar_raw[:, 0] = -pts_lidar_raw[:, 0] # invert x axis

    pcd_lidar.points = o3d.utility.Vector3dVector(pts_lidar_raw)

    # Downsample to reduce point count for faster processing
    if verbose:
        print(f"  [{idx}] lidar points pre-voxel: {n_lidar_pre}")
    pcd_lidar = pcd_lidar.voxel_down_sample(voxel_size=LIDAR_PCD_VOXEL_SIZE)
    n_lidar_post = len(pcd_lidar.points)
    
    # Warn if over threshold (but don't cap)
    if n_lidar_post > MAX_POINTS_WARNING_THRESHOLD:
        print(f"  [{idx}] WARNING: LiDAR cloud has {n_lidar_post} points (> {MAX_POINTS_WARNING_THRESHOLD}); consider increasing voxel size")
    
    if verbose:
        print(f"  [{idx}] lidar points post-voxel: {n_lidar_post}")
    
    # Note: LiDAR cropping removed in this change (focus on camera crop only)
    n_lidar_post_crop = len(pcd_lidar.points)
    
    # Extract LiDAR points AFTER downsampling/cropping to match the point cloud
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
        print(f"  [{idx}] Running RANSAC-based FPFH correspondence matching...")
    
    # Use RANSAC-based global registration for correspondences
    start_time = time.time()
    points_cam_corr, pts_lidar_corr, ransac_result = find_correspondences_ransac_fpfh(
        pcd_cam, pcd_lidar, FEATURE_VOXEL_SIZE, mutual_filter=True
    )
    feature_time = time.time() - start_time
    
    n_corr = len(points_cam_corr)
    
    # TEMP VERIFICATION: Check coordinate frame consistency
    if verbose and n_corr > 0:
        # Check point cloud bounds to verify coordinate frames
        cam_pts_array = np.asarray(pcd_cam.points)
        lidar_pts_array = np.asarray(pcd_lidar.points)
        print(f"  [{idx}] Coordinate frame check:")
        print(f"    Camera cloud bounds: X[{cam_pts_array[:, 0].min():.2f}, {cam_pts_array[:, 0].max():.2f}], "
              f"Y[{cam_pts_array[:, 1].min():.2f}, {cam_pts_array[:, 1].max():.2f}], "
              f"Z[{cam_pts_array[:, 2].min():.2f}, {cam_pts_array[:, 2].max():.2f}]")
        print(f"    LiDAR cloud bounds: X[{lidar_pts_array[:, 0].min():.2f}, {lidar_pts_array[:, 0].max():.2f}], "
              f"Y[{lidar_pts_array[:, 1].min():.2f}, {lidar_pts_array[:, 1].max():.2f}], "
              f"Z[{lidar_pts_array[:, 2].min():.2f}, {lidar_pts_array[:, 2].max():.2f}]")
        print(f"    Correspondence points - Camera: {points_cam_corr.shape}, LiDAR: {pts_lidar_corr.shape}")
        # Check if correspondence points are within original cloud bounds
        if n_corr > 0:
            corr_cam_in_bounds = np.all(
                (points_cam_corr >= cam_pts_array.min(axis=0)) & 
                (points_cam_corr <= cam_pts_array.max(axis=0))
            )
            corr_lidar_in_bounds = np.all(
                (pts_lidar_corr >= lidar_pts_array.min(axis=0)) & 
                (pts_lidar_corr <= lidar_pts_array.max(axis=0))
            )
            print(f"    Correspondence points within original bounds: cam={corr_cam_in_bounds}, lidar={corr_lidar_in_bounds}")
    
    # Determine if we should visualize this frame
    # In test mode, visualize first frame (is_first_frame). Otherwise, check frame index.
    should_viz = viz and (is_first_frame or (not is_first_frame and idx == f"{viz_frame_idx:05d}"))
    
    # Visualization: pre-registration
    if should_viz:
        try:
            pcd_cam_viz = copy.deepcopy(pcd_cam)
            pcd_lidar_viz = copy.deepcopy(pcd_lidar)
            pcd_cam_viz.paint_uniform_color([1.0, 0.0, 0.0])  # Red for camera
            pcd_lidar_viz.paint_uniform_color([0.0, 1.0, 0.0])  # Green for LiDAR
            print(f"  [{idx}] Visualizing pre-registration clouds (camera=red, LiDAR=green)...")
            o3d.visualization.draw_geometries([pcd_cam_viz, pcd_lidar_viz], 
                                             window_name=f"Pre-registration [{idx}]")
        except Exception as e:
            print(f"  [{idx}] WARNING: Visualization failed (headless mode?): {e}")
    
    # Consolidated debug print per frame
    print(f"  [{idx}] === Frame Summary ===")
    print(f"    Camera: crop=Z[{crop_z_min:.1f},{crop_z_max:.1f}]m FOV[{crop_fov_scale:.2f}], voxel={final_voxel_size:.6f}m, points={n_cam_post_crop}->{n_cam_post_voxel}")
    print(f"    LiDAR: points={n_lidar_post_crop}")
    print(f"    RANSAC: {feature_time:.3f}s, {n_corr} correspondences", end="")
    if hasattr(ransac_result, 'fitness'):
        print(f", fitness={ransac_result.fitness:.4f}, inlier_rmse={ransac_result.inlier_rmse:.6f}")
    else:
        print()
    
    if n_corr < 3:
        print(f"  [{idx}] SKIP: too few correspondences from RANSAC (n={n_corr}, need >= 3)")
        return None
    
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
    
    # Calculate inlier ratio (points within threshold after transformation)
    if 'final_T' in metadata:
        # Transform camera correspondences using final composed transformation
        T_total = metadata['final_T']
        # Apply transformation: transformed = T_total @ [points; 1]
        points_cam_corr_homogeneous = np.hstack([points_cam_corr, np.ones((len(points_cam_corr), 1))])
        points_cam_corr_transformed = (T_total @ points_cam_corr_homogeneous.T).T[:, :3]
        # Compute distances between transformed camera points and LiDAR correspondences
        dists = np.linalg.norm(points_cam_corr_transformed - pts_lidar_corr, axis=1)
        inlier_threshold = 0.3  # 0.3m threshold for inlier ratio
        inlier_ratio = np.sum(dists < inlier_threshold) / len(dists) if len(dists) > 0 else 0.0
        
        # VERIFICATION: Debug inlier ratio calculation
        if verbose and n_corr > 0:
            print(f"  [{idx}] Inlier ratio verification:")
            print(f"    Distance stats: min={dists.min():.4f}m, max={dists.max():.4f}m, mean={dists.mean():.4f}m, median={np.median(dists):.4f}m")
            print(f"    Points within {inlier_threshold}m: {np.sum(dists < inlier_threshold)}/{n_corr}")
            # Show sample of transformed vs target points
            if n_corr >= 3:
                print(f"    Sample (first 3 correspondences):")
                for i in range(min(3, n_corr)):
                    print(f"      [{i}] cam_transformed: {points_cam_corr_transformed[i]}, lidar: {pts_lidar_corr[i]}, dist: {dists[i]:.4f}m")
        metadata['correspondences_count'] = n_corr
        metadata['inlier_ratio'] = float(inlier_ratio)
        metadata['teaser_runtime'] = teaser_time
        metadata['feature_extraction_time'] = feature_time
        if hasattr(ransac_result, 'fitness'):
            metadata['ransac_fitness'] = float(ransac_result.fitness)
            metadata['ransac_inlier_rmse'] = float(ransac_result.inlier_rmse)
        
        if verbose:
            print(f"  [{idx}] TEASER++ runtime: {teaser_time:.3f}s")
            n_inliers = np.sum(dists < inlier_threshold)
            print(f"  [{idx}] Inlier ratio: {inlier_ratio:.4f} ({n_inliers}/{n_corr}) [threshold: {inlier_threshold:.2f}m]")
    else:
        metadata['correspondences_count'] = n_corr
        metadata['teaser_runtime'] = teaser_time
        metadata['feature_extraction_time'] = feature_time
    
    # Visualization: post-registration
    if should_viz and viz_after:
        try:
            cam_aligned_viz = copy.deepcopy(cam_aligned)
            pcd_lidar_viz = copy.deepcopy(pcd_lidar)
            cam_aligned_viz.paint_uniform_color([1.0, 0.0, 0.0])  # Red for aligned camera
            pcd_lidar_viz.paint_uniform_color([0.0, 1.0, 0.0])  # Green for LiDAR
            print(f"  [{idx}] Visualizing post-registration (aligned camera=red, LiDAR=green)...")
            o3d.visualization.draw_geometries([cam_aligned_viz, pcd_lidar_viz],
                                             window_name=f"Post-registration [{idx}]")
        except Exception as e:
            print(f"  [{idx}] WARNING: Post-registration visualization failed: {e}")
    
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


def process_depth_method(depth_method, depth_dir, limit=None, test_mode=False, estimate_scaling=True, depth_mode_dict=None,
                         crop_z_min=2.0, crop_z_max=50.0, crop_fov_scale=0.8, viz=False, viz_after=False, viz_frame_idx=0, k_scale=2.0,
                         rel_crop_min_pct=2, rel_crop_max_pct=95, rel_crop_invert=False):
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
        # Check if this is the frame to visualize
        # In test mode: visualize first frame (i==0). Otherwise: visualize frame at viz_frame_idx
        should_viz_frame = viz and ((test_mode and is_first) or (not test_mode and i == viz_frame_idx))
        result = process_pair(
            idx, depth_dir, depth_method, meta_path, 
            verbose=test_mode, 
            estimate_scaling=estimate_scaling,
            depth_mode=depth_mode,
            is_first_frame=is_first,
            crop_z_min=crop_z_min,
            crop_z_max=crop_z_max,
            crop_fov_scale=crop_fov_scale,
            viz=should_viz_frame,
            viz_after=viz_after if should_viz_frame else False,
            viz_frame_idx=viz_frame_idx,
            k_scale=k_scale,
            rel_crop_min_pct=rel_crop_min_pct,
            rel_crop_max_pct=rel_crop_max_pct,
            rel_crop_invert=rel_crop_invert
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
    parser.add_argument(
        "--crop-z-min",
        type=float,
        default=2.0,
        help="Minimum Z (depth) for camera crop in meters (default: 2.0)"
    )
    parser.add_argument(
        "--crop-z-max",
        type=float,
        default=50.0,
        help="Maximum Z (depth) for camera crop in meters (default: 50.0)"
    )
    parser.add_argument(
        "--crop-fov-scale",
        type=float,
        default=0.8,
        help="FOV crop scale factor: keep central fraction of image FOV (0.8 = 80%%, 1.0 = no FOV crop, default: 0.8)"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Visualize pre-registration clouds (camera + LiDAR) on selected frame"
    )
    parser.add_argument(
        "--viz-after",
        action="store_true",
        help="Also visualize post-registration (aligned camera + LiDAR) if --viz is enabled"
    )
    parser.add_argument(
        "--viz-frame-idx",
        type=int,
        default=0,
        help="Frame index to visualize (0 = first processed frame, default: 0)"
    )
    parser.add_argument(
        "--k-scale",
        type=float,
        default=2.0,
        help="Scale factor for camera intrinsics (fx, fy, cx, cy). Default: 2.0 (for 2x resolution depth maps)"
    )
    parser.add_argument(
        "--rel-crop-min-pct",
        type=float,
        default=2,
        help="Minimum Z percentile for relative depth methods (MiDaS, Marigold, etc.). Default: 2 (2nd percentile)"
    )
    parser.add_argument(
        "--rel-crop-max-pct",
        type=float,
        default=95,
        help="Maximum Z percentile for relative depth methods (MiDaS, Marigold, etc.). Default: 95 (95th percentile)"
    )
    parser.add_argument(
        "--rel-crop-invert",
        action="store_true",
        help="Invert percentile interpretation for relative depth methods (for 'larger value = closer' cases)"
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
    
    # Print crop settings if provided
    if args.crop_z_min != 2.0 or args.crop_z_max != 50.0 or args.crop_fov_scale != 0.8:
        print(f"CROP SETTINGS: Z[{args.crop_z_min:.1f}, {args.crop_z_max:.1f}]m, FOV scale={args.crop_fov_scale:.2f}")
    if args.viz:
        print(f"VISUALIZATION: Enabled (frame_idx={args.viz_frame_idx}, after={args.viz_after})")
    if args.k_scale != 2.0:
        print(f"INTRINSICS SCALING: k_scale={args.k_scale:.2f} (default: 2.0)")
    else:
        print(f"INTRINSICS SCALING: k_scale={args.k_scale:.2f} (default)")
    
    total_processed = 0
    for method in methods_to_process:
        depth_dir = DEPTH_DIRS[method]
        processed = process_depth_method(
            method, 
            depth_dir, 
            limit=args.limit,
            test_mode=args.test,
            estimate_scaling=estimate_scaling,
            depth_mode_dict=depth_mode_dict,
            crop_z_min=args.crop_z_min,
            crop_z_max=args.crop_z_max,
            crop_fov_scale=args.crop_fov_scale,
            viz=args.viz,
            viz_after=args.viz_after,
            viz_frame_idx=args.viz_frame_idx,
            k_scale=args.k_scale,
            rel_crop_min_pct=args.rel_crop_min_pct,
            rel_crop_max_pct=args.rel_crop_max_pct,
            rel_crop_invert=args.rel_crop_invert
        )
        total_processed += processed
    
    print(f"\n{'='*60}")
    print(f"Batch complete! Total files processed: {total_processed}")
    print(f"Metadata files stored in: {OUTPUT_DIR}/metadata_*.jsonl")
    print(f"Aligned point clouds stored in: {OUTPUT_DIR}/cam_aligned_*.ply")
