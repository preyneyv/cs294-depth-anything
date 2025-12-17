# evaluate_reprojection.py - Evaluate R, T, s accuracy by back-projecting LiDAR to image depth

"""
Evaluation of transformation accuracy by comparing:
1. Project LiDAR points to image using estimated R, T
2. Compare projected LiDAR depth with depth map at those pixels
3. If R, T, s are correct, depths should match

Metrics:
- Depth error (MAE, RMSE, median)
- Relative depth error (scale-invariant)
- Inlier ratio (% of points within threshold)
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd


@dataclass
class ReprojectionResult:
    """Results from reprojection evaluation."""
    index: str
    n_points: int           # Total LiDAR points
    n_valid: int            # Points that project into image
    n_inliers: int          # Points within depth threshold
    
    # Depth errors (in metric units, meters)
    depth_mae: float        # Mean absolute error
    depth_rmse: float       # Root mean square error
    depth_median: float     # Median absolute error
    depth_std: float        # Standard deviation
    
    # Relative errors (scale-invariant)
    rel_mae: float          # |d_lidar - d_pred| / d_lidar
    rel_rmse: float
    
    # Inlier ratios
    inlier_1m: float        # % within 1m
    inlier_2m: float        # % within 2m
    inlier_5m: float        # % within 5m
    
    # Scale consistency
    scale_ratio_mean: float   # mean(d_lidar / d_pred)
    scale_ratio_std: float
    
    # Optional: breakdown by distance
    near_mae: float = 0.0    # < 10m
    mid_mae: float = 0.0     # 10-30m
    far_mae: float = 0.0     # > 30m


class ReprojectionEvaluator:
    """
    Evaluate transformation accuracy by LiDAR-to-depth reprojection.
    
    Coordinate conventions:
        LiDAR frame: X=forward, Y=left, Z=up
        Camera frame: X=right, Y=down, Z=forward
        
    Transformation: P_cam = R @ P_lidar + t
    Then project: p = K @ P_cam / P_cam[2]
    """
    
    def __init__(self, 
                 K: np.ndarray,
                 image_size: Tuple[int, int] = (1032, 778),
                 depth_scale: float = 90.0,
                 inverse_depth: bool = True):
        """
        Args:
            K: 3x3 camera intrinsic matrix
            image_size: (width, height) of depth image
            depth_scale: Scale factor for inverse depth (inv_depth = scale / metric_depth)
            inverse_depth: If True, depth map stores inverse depth
        """
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.width, self.height = image_size
        self.depth_scale = depth_scale
        self.inverse_depth = inverse_depth
        
        # LiDAR to Camera rotation (coordinate frame change, not the estimated R)
        # LiDAR: X=forward, Y=left, Z=up
        # Camera: X=right, Y=down, Z=forward
        # P_cam = R_lidar2cam @ P_lidar
        self.R_lidar2cam = np.array([
            [0, -1, 0],   # Cam_X = -LiDAR_Y (right = -left)
            [0, 0, -1],   # Cam_Y = -LiDAR_Z (down = -up)
            [1, 0, 0]     # Cam_Z = LiDAR_X (forward = forward)
        ], dtype=np.float64)
    
    def transform_lidar_to_camera(self, 
                                   pts_lidar: np.ndarray,
                                   R: np.ndarray,
                                   t: np.ndarray,
                                   scale: float = 1.0) -> np.ndarray:
        """
        Transform LiDAR points to camera frame.
        
        The full transformation is:
        1. Apply estimated ICP/TEASER transform (aligns depth cloud to LiDAR)
           Since we want LiDAR -> Camera, we use the inverse
        2. Apply coordinate frame change (LiDAR convention -> Camera convention)
        
        Args:
            pts_lidar: (N, 3) points in LiDAR frame
            R: 3x3 rotation from ICP/TEASER (depth_aligned = R @ depth + t)
            t: 3x1 translation
            scale: Scale factor
            
        Returns:
            pts_cam: (N, 3) points in camera frame
        """
        # The metadata stores transform from depth cloud to LiDAR
        # depth_aligned = R @ depth_cloud + t
        # To go LiDAR -> depth_cloud: depth_cloud = R^T @ (lidar - t) / scale
        # But we want LiDAR -> camera frame
        
        # Method: Transform LiDAR to camera coordinate frame directly
        # P_cam = R_lidar2cam @ P_lidar
        pts_cam = (self.R_lidar2cam @ pts_lidar.T).T
        
        return pts_cam
    
    def project_to_image(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D camera points to 2D image coordinates.
        
        Args:
            pts_cam: (N, 3) points in camera frame
            
        Returns:
            u, v: (M,) pixel coordinates of valid points
            depths: (M,) depth values (Z coordinate)
            valid_mask: (N,) boolean mask of valid projections
        """
        # Filter points in front of camera
        valid_mask = pts_cam[:, 2] > 0.1  # At least 10cm in front
        
        pts_valid = pts_cam[valid_mask]
        
        if len(pts_valid) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Project
        depths = pts_valid[:, 2]
        u = (self.fx * pts_valid[:, 0] / depths + self.cx).astype(int)
        v = (self.fy * pts_valid[:, 1] / depths + self.cy).astype(int)
        
        # Filter points within image bounds
        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        
        return u[in_bounds], v[in_bounds], depths[in_bounds]
    
    def get_depth_at_pixels(self, 
                            depth_map: np.ndarray, 
                            u: np.ndarray, 
                            v: np.ndarray,
                            depth_scale: float = None) -> np.ndarray:
        """
        Sample depth map at given pixel coordinates.
        
        Args:
            depth_map: (H, W) depth image
            u, v: Pixel coordinates
            depth_scale: Override default depth scale
            
        Returns:
            depths: Metric depth values at (u, v)
        """
        if depth_scale is None:
            depth_scale = self.depth_scale
            
        inv_depths = depth_map[v, u].astype(np.float64)
        
        if self.inverse_depth:
            # Convert inverse depth to metric depth
            valid = inv_depths > 0.1  # Avoid division by zero
            depths = np.zeros_like(inv_depths)
            depths[valid] = depth_scale / inv_depths[valid]
            depths[~valid] = np.nan
        else:
            depths = inv_depths
            
        return depths
    
    def evaluate_single(self,
                        pts_lidar: np.ndarray,
                        depth_map: np.ndarray,
                        R: np.ndarray,
                        t: np.ndarray,
                        scale: float = 1.0,
                        depth_scale: float = None,
                        index: str = "unknown") -> ReprojectionResult:
        """
        Evaluate reprojection error for a single frame.
        
        Args:
            pts_lidar: (N, 3) LiDAR points
            depth_map: (H, W) depth image
            R: 3x3 rotation matrix
            t: 3x1 translation vector  
            scale: Scale factor
            depth_scale: Depth map scale factor
            index: Frame identifier
            
        Returns:
            ReprojectionResult with all metrics
        """
        # Transform LiDAR to camera frame
        pts_cam = self.transform_lidar_to_camera(pts_lidar, R, t, scale)
        
        # Project to image
        u, v, depths_lidar = self.project_to_image(pts_cam)
        
        if len(u) == 0:
            return ReprojectionResult(
                index=index, n_points=len(pts_lidar), n_valid=0, n_inliers=0,
                depth_mae=np.inf, depth_rmse=np.inf, depth_median=np.inf, depth_std=np.inf,
                rel_mae=np.inf, rel_rmse=np.inf,
                inlier_1m=0, inlier_2m=0, inlier_5m=0,
                scale_ratio_mean=0, scale_ratio_std=np.inf
            )
        
        # Get depth map values at projected locations
        depths_pred = self.get_depth_at_pixels(depth_map, u, v, depth_scale)
        
        # Filter valid predictions
        valid = ~np.isnan(depths_pred) & (depths_pred > 0) & (depths_lidar > 0)
        depths_lidar = depths_lidar[valid]
        depths_pred = depths_pred[valid]
        
        if len(depths_lidar) == 0:
            return ReprojectionResult(
                index=index, n_points=len(pts_lidar), n_valid=0, n_inliers=0,
                depth_mae=np.inf, depth_rmse=np.inf, depth_median=np.inf, depth_std=np.inf,
                rel_mae=np.inf, rel_rmse=np.inf,
                inlier_1m=0, inlier_2m=0, inlier_5m=0,
                scale_ratio_mean=0, scale_ratio_std=np.inf
            )
        
        # Compute depth errors
        errors = np.abs(depths_lidar - depths_pred)
        depth_mae = np.mean(errors)
        depth_rmse = np.sqrt(np.mean(errors ** 2))
        depth_median = np.median(errors)
        depth_std = np.std(errors)
        
        # Relative errors
        rel_errors = errors / depths_lidar
        rel_mae = np.mean(rel_errors)
        rel_rmse = np.sqrt(np.mean(rel_errors ** 2))
        
        # Inlier ratios
        inlier_1m = np.mean(errors < 1.0)
        inlier_2m = np.mean(errors < 2.0)
        inlier_5m = np.mean(errors < 5.0)
        
        # Scale ratio
        scale_ratios = depths_lidar / depths_pred
        scale_ratio_mean = np.mean(scale_ratios)
        scale_ratio_std = np.std(scale_ratios)
        
        # Distance-based breakdown
        near_mask = depths_lidar < 10
        mid_mask = (depths_lidar >= 10) & (depths_lidar < 30)
        far_mask = depths_lidar >= 30
        
        near_mae = np.mean(errors[near_mask]) if np.any(near_mask) else 0
        mid_mae = np.mean(errors[mid_mask]) if np.any(mid_mask) else 0
        far_mae = np.mean(errors[far_mask]) if np.any(far_mask) else 0
        
        return ReprojectionResult(
            index=index,
            n_points=len(pts_lidar),
            n_valid=len(depths_lidar),
            n_inliers=int(np.sum(errors < 2.0)),
            depth_mae=depth_mae,
            depth_rmse=depth_rmse,
            depth_median=depth_median,
            depth_std=depth_std,
            rel_mae=rel_mae,
            rel_rmse=rel_rmse,
            inlier_1m=inlier_1m,
            inlier_2m=inlier_2m,
            inlier_5m=inlier_5m,
            scale_ratio_mean=scale_ratio_mean,
            scale_ratio_std=scale_ratio_std,
            near_mae=near_mae,
            mid_mae=mid_mae,
            far_mae=far_mae
        )
    
    def visualize_reprojection(self,
                               pts_lidar: np.ndarray,
                               depth_map: np.ndarray,
                               R: np.ndarray,
                               t: np.ndarray,
                               scale: float = 1.0,
                               depth_scale: float = None,
                               save_path: str = None):
        """
        Visualize reprojection: overlay LiDAR depths on depth map.
        """
        # Transform and project
        pts_cam = self.transform_lidar_to_camera(pts_lidar, R, t, scale)
        u, v, depths_lidar = self.project_to_image(pts_cam)
        
        if len(u) == 0:
            print("No valid projections!")
            return
        
        # Get predicted depths
        depths_pred = self.get_depth_at_pixels(depth_map, u, v, depth_scale)
        
        # Filter valid
        valid = ~np.isnan(depths_pred) & (depths_pred > 0)
        u, v = u[valid], v[valid]
        depths_lidar = depths_lidar[valid]
        depths_pred = depths_pred[valid]
        
        errors = np.abs(depths_lidar - depths_pred)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Depth map with LiDAR overlay
        ax = axes[0, 0]
        ax.imshow(depth_map, cmap='magma')
        scatter = ax.scatter(u, v, c=depths_lidar, s=1, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='LiDAR Depth (m)')
        ax.set_title('Depth Map + Projected LiDAR')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        
        # 2. Error map
        ax = axes[0, 1]
        ax.imshow(depth_map, cmap='gray', alpha=0.3)
        scatter = ax.scatter(u, v, c=errors, s=2, cmap='hot', vmin=0, vmax=5)
        plt.colorbar(scatter, ax=ax, label='Depth Error (m)')
        ax.set_title('Reprojection Error Map')
        
        # 3. Depth comparison scatter
        ax = axes[1, 0]
        ax.scatter(depths_lidar, depths_pred, alpha=0.3, s=1)
        max_d = max(depths_lidar.max(), depths_pred.max())
        ax.plot([0, max_d], [0, max_d], 'r--', label='Perfect alignment')
        ax.set_xlabel('LiDAR Depth (m)')
        ax.set_ylabel('Predicted Depth (m)')
        ax.set_title('Depth Correlation')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 4. Error histogram
        ax = axes[1, 1]
        ax.hist(errors, bins=50, range=(0, 10), edgecolor='black', alpha=0.7)
        ax.axvline(np.median(errors), color='r', linestyle='--', label=f'Median: {np.median(errors):.2f}m')
        ax.axvline(np.mean(errors), color='g', linestyle='--', label=f'Mean: {np.mean(errors):.2f}m')
        ax.set_xlabel('Depth Error (m)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")
        
        return fig


def load_metadata(metadata_path: str) -> List[Dict]:
    """Load metadata from JSONL file."""
    results = []
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def evaluate_dataset(
    metadata_path: str,
    depth_dir: str,
    lidar_dir: str,
    K: np.ndarray,
    image_size: Tuple[int, int] = (1032, 778),
    transform_type: str = 'icp_pre',  # 'icp_pre' or 'teaser'
    output_dir: str = './reprojection_eval',
    visualize_samples: int = 5
) -> pd.DataFrame:
    """
    Evaluate reprojection accuracy for entire dataset.
    
    Args:
        metadata_path: Path to metadata.jsonl
        depth_dir: Directory containing depth_*.npy files
        lidar_dir: Directory containing lidar_*.pcd files
        K: Camera intrinsic matrix
        image_size: (width, height) of depth images
        transform_type: Which transform to evaluate ('icp_pre' or 'teaser')
        output_dir: Output directory for results
        visualize_samples: Number of samples to visualize
        
    Returns:
        DataFrame with per-frame results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    depth_dir = Path(depth_dir)
    lidar_dir = Path(lidar_dir)
    
    # Load metadata
    metadata_list = load_metadata(metadata_path)
    print(f"Loaded {len(metadata_list)} frames from metadata")
    
    # Create evaluator
    evaluator = ReprojectionEvaluator(K, image_size)
    
    results = []
    vis_count = 0
    
    for meta in tqdm(metadata_list, desc="Evaluating"):
        idx = meta['index']
        
        # Load data
        depth_path = depth_dir / f"depth_{idx}.npy"
        lidar_path = lidar_dir / f"lidar_{idx}.pcd"
        
        if not depth_path.exists() or not lidar_path.exists():
            print(f"[SKIP] Missing files for {idx}")
            continue
        
        depth_map = np.load(depth_path)
        lidar_pcd = o3d.io.read_point_cloud(str(lidar_path))
        pts_lidar = np.asarray(lidar_pcd.points)
        
        # Get transform from metadata
        if transform_type == 'icp_pre':
            T = np.array(meta['icp_pre_transform'])
            R = T[:3, :3]
            t = T[:3, 3]
            scale = 1.0
        elif transform_type == 'teaser':
            R = np.array(meta['teaser_R'])
            t = np.array(meta['teaser_t'])
            scale = meta.get('teaser_scale', 1.0)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        depth_scale = meta.get('depth_scale', 90.0)
        
        # Evaluate
        result = evaluator.evaluate_single(
            pts_lidar, depth_map, R, t, scale, depth_scale, idx
        )
        results.append(result)
        
        # Visualize some samples
        if vis_count < visualize_samples and result.n_valid > 100:
            evaluator.visualize_reprojection(
                pts_lidar, depth_map, R, t, scale, depth_scale,
                save_path=str(output_dir / f"vis_{idx}.png")
            )
            vis_count += 1
    
    # Convert to DataFrame
    df = pd.DataFrame([r.__dict__ for r in results])
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"REPROJECTION EVALUATION SUMMARY ({transform_type})")
    print("=" * 80)
    
    print(f"\nDataset: {len(df)} frames evaluated")
    print(f"Valid projections: {df['n_valid'].mean():.0f} avg points per frame")
    
    print(f"\n--- Depth Errors (meters) ---")
    print(f"MAE:    mean={df['depth_mae'].mean():.3f}, median={df['depth_mae'].median():.3f}, std={df['depth_mae'].std():.3f}")
    print(f"RMSE:   mean={df['depth_rmse'].mean():.3f}, median={df['depth_rmse'].median():.3f}")
    print(f"Median: mean={df['depth_median'].mean():.3f}")
    
    print(f"\n--- Relative Errors ---")
    print(f"Rel MAE:  {df['rel_mae'].mean():.3f} ({df['rel_mae'].mean()*100:.1f}%)")
    print(f"Rel RMSE: {df['rel_rmse'].mean():.3f} ({df['rel_rmse'].mean()*100:.1f}%)")
    
    print(f"\n--- Inlier Ratios ---")
    print(f"< 1m:  {df['inlier_1m'].mean()*100:.1f}%")
    print(f"< 2m:  {df['inlier_2m'].mean()*100:.1f}%")
    print(f"< 5m:  {df['inlier_5m'].mean()*100:.1f}%")
    
    print(f"\n--- Scale Consistency ---")
    print(f"Scale ratio: {df['scale_ratio_mean'].mean():.3f} ± {df['scale_ratio_std'].mean():.3f}")
    print(f"(Should be ~1.0 if scale is correct)")
    
    print(f"\n--- Distance Breakdown (MAE) ---")
    print(f"Near (<10m):  {df['near_mae'].mean():.3f}m")
    print(f"Mid (10-30m): {df['mid_mae'].mean():.3f}m")
    print(f"Far (>30m):   {df['far_mae'].mean():.3f}m")
    
    # Save results
    df.to_csv(output_dir / 'reprojection_results.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'reprojection_results.csv'}")
    
    # Create summary plots
    create_summary_plots(df, output_dir)
    
    return df


def create_summary_plots(df: pd.DataFrame, output_dir: Path):
    """Create summary visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Reprojection Evaluation Summary', fontsize=16, fontweight='bold')
    
    # 1. MAE distribution
    ax = axes[0, 0]
    ax.hist(df['depth_mae'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(df['depth_mae'].mean(), color='r', linestyle='--', label=f'Mean: {df["depth_mae"].mean():.2f}m')
    ax.axvline(df['depth_mae'].median(), color='g', linestyle='--', label=f'Median: {df["depth_mae"].median():.2f}m')
    ax.set_xlabel('Depth MAE (m)')
    ax.set_ylabel('Count')
    ax.set_title('Depth MAE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Inlier ratios
    ax = axes[0, 1]
    inlier_means = [df['inlier_1m'].mean(), df['inlier_2m'].mean(), df['inlier_5m'].mean()]
    ax.bar(['< 1m', '< 2m', '< 5m'], [x * 100 for x in inlier_means], color=['green', 'yellow', 'orange'])
    ax.set_ylabel('Inlier Ratio (%)')
    ax.set_title('Inlier Ratios by Threshold')
    ax.set_ylim(0, 100)
    for i, v in enumerate(inlier_means):
        ax.text(i, v * 100 + 2, f'{v*100:.1f}%', ha='center')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Scale consistency
    ax = axes[0, 2]
    ax.hist(df['scale_ratio_mean'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='r', linestyle='--', label='Perfect (1.0)')
    ax.axvline(df['scale_ratio_mean'].mean(), color='g', linestyle='--', 
               label=f'Mean: {df["scale_ratio_mean"].mean():.3f}')
    ax.set_xlabel('Scale Ratio (LiDAR / Predicted)')
    ax.set_ylabel('Count')
    ax.set_title('Scale Consistency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distance breakdown
    ax = axes[1, 0]
    distances = ['Near (<10m)', 'Mid (10-30m)', 'Far (>30m)']
    maes = [df['near_mae'].mean(), df['mid_mae'].mean(), df['far_mae'].mean()]
    colors = ['green', 'yellow', 'red']
    ax.bar(distances, maes, color=colors, alpha=0.7)
    ax.set_ylabel('MAE (m)')
    ax.set_title('Error by Distance Range')
    for i, v in enumerate(maes):
        ax.text(i, v + 0.1, f'{v:.2f}m', ha='center')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Per-frame MAE
    ax = axes[1, 1]
    ax.plot(df['depth_mae'].values, marker='o', markersize=2, linestyle='-', alpha=0.7)
    ax.axhline(df['depth_mae'].mean(), color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Depth MAE (m)')
    ax.set_title('Per-Frame Error')
    ax.grid(True, alpha=0.3)
    
    # 6. Relative error
    ax = axes[1, 2]
    ax.hist(df['rel_mae'] * 100, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(df['rel_mae'].mean() * 100, color='r', linestyle='--', 
               label=f'Mean: {df["rel_mae"].mean()*100:.1f}%')
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Relative Depth Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_plots.png', dpi=150, bbox_inches='tight')
    print(f"Saved summary plots: {output_dir / 'summary_plots.png'}")
    plt.close()


def main():
    """Main function - configure and run evaluation."""
    
    # === CONFIGURATION ===
    DATA_ROOT = Path("../dataset")
    METADATA_PATH = DATA_ROOT / "aligned_combined/metadata.jsonl"
    DEPTH_DIR = DATA_ROOT / "depth"
    LIDAR_DIR = DATA_ROOT / "lidar"
    OUTPUT_DIR = Path("./reprojection_eval")
    
    # Camera intrinsics
    K = np.array([
        [491.331107883326, 0.0, 515.3434363622374],
        [0.0, 492.14998153326013, 388.93983736974667],
        [0.0, 0.0, 1.0]
    ])
    
    IMAGE_SIZE = (1032, 778)  # (width, height)
    
    # Which transform to evaluate
    TRANSFORM_TYPE = 'icp_pre'  # or 'teaser'
    
    # === RUN EVALUATION ===
    print("=" * 80)
    print("REPROJECTION ACCURACY EVALUATION")
    print("=" * 80)
    print(f"\nMetadata: {METADATA_PATH}")
    print(f"Depth dir: {DEPTH_DIR}")
    print(f"LiDAR dir: {LIDAR_DIR}")
    print(f"Transform: {TRANSFORM_TYPE}")
    print(f"Output: {OUTPUT_DIR}")
    
    df = evaluate_dataset(
        metadata_path=str(METADATA_PATH),
        depth_dir=str(DEPTH_DIR),
        lidar_dir=str(LIDAR_DIR),
        K=K,
        image_size=IMAGE_SIZE,
        transform_type=TRANSFORM_TYPE,
        output_dir=str(OUTPUT_DIR),
        visualize_samples=5
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    main()
