# evaluate_methods.py - Evaluate depth methods by back-projecting LiDAR to compare with depth maps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
import cv2
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class ReprojectionResult:
    """Results from reprojection evaluation for a single frame."""
    index: str
    n_lidar_pts: int
    n_projected: int
    n_valid: int
    
    # Depth errors
    depth_mae: float
    depth_rmse: float
    depth_median: float
    
    # Relative errors
    rel_mae: float
    rel_rmse: float
    
    # Inlier ratios
    inlier_1m: float
    inlier_2m: float
    inlier_5m: float
    
    # Scale
    scale_ratio: float
    scale_std: float
    
    # From metadata
    icp_fitness: float
    icp_rmse: float
    chamfer: float


class DepthMethodEvaluator:
    """
    Evaluate depth estimation methods by back-projecting LiDAR to image
    and comparing with predicted depth maps.
    
    The pipeline:
    1. Load LiDAR points (in LiDAR frame)
    2. Transform to camera frame using coordinate change
    3. Project to image using scaled camera intrinsics
    4. Compare projected depth with depth map values
    
    IMPORTANT: The depth maps are 2064x980 (2x upscaled, cropped from 1556 to 980).
    We pad them back to 2064x1556 and use 2x scaled intrinsics for correct projection.
    
    Usage:
        evaluator = DepthMethodEvaluator(K, lidar_dir, original_image_size=(1032, 778))
        evaluator.evaluate_method("depth_anything_v3", depth_dir, metadata_path)
        evaluator.compare_methods()
    """
    
    def __init__(self, 
                 K: np.ndarray,
                 lidar_dir: str,
                 original_image_size: Tuple[int, int] = (1032, 778),
                 depth_scale: float = 100.0,
                 inverse_depth: bool = False):
        """
        Args:
            K: 3x3 camera intrinsic matrix (for original image size)
            lidar_dir: Directory containing lidar_*.pcd files
            original_image_size: (width, height) of original image that K corresponds to
            depth_scale: Scale factor for depth conversion
            inverse_depth: If True, depth maps store inverse depth
        """
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.lidar_dir = Path(lidar_dir)
        self.original_image_size = original_image_size  # (width, height)
        self.depth_scale = depth_scale
        self.inverse_depth = inverse_depth
        
        # Compute target padded size (2x original)
        self.target_width = original_image_size[0] * 2   # 2064
        self.target_height = original_image_size[1] * 2  # 1556
        
        # Scaled intrinsics (2x)
        self.scale = 2.0
        self.fx_scaled = self.fx * self.scale
        self.fy_scaled = self.fy * self.scale
        self.cx_scaled = self.cx * self.scale
        self.cy_scaled = self.cy * self.scale
        
        # Results storage
        self.methods = {}  # method_name -> list of ReprojectionResult
    
    def load_lidar(self, idx: str) -> np.ndarray:
        """Load LiDAR point cloud."""
        lidar_path = self.lidar_dir / f"lidar_{idx}.pcd"
        if not lidar_path.exists():
            return None
        pcd = o3d.io.read_point_cloud(str(lidar_path))
        return np.asarray(pcd.points)
    
    def load_depth(self, depth_path: Path) -> np.ndarray:
        """Load depth map and pad to correct size."""
        depth_map = np.load(depth_path)
        return self.pad_depth_map(depth_map)
    
    def pad_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Pad depth map to correct aspect ratio.
        
        Original image: 1032x778, upscaled 2x: 2064x1556
        Current depth: 2064x980 (cropped from bottom)
        Need to pad bottom to get 2064x1556
        """
        h, w = depth_map.shape
        
        if h < self.target_height:
            pad_bottom = self.target_height - h
            depth_map = np.pad(depth_map, ((0, pad_bottom), (0, 0)), 
                              mode='constant', constant_values=0)
        
        return depth_map
    
    def transform_lidar_to_camera(self, pts_lidar: np.ndarray) -> np.ndarray:
        """
        Transform LiDAR points to camera frame.
        
        LiDAR frame: X=forward, Y=left, Z=up
        Camera frame: X=right, Y=down, Z=forward
        
        Args:
            pts_lidar: (N, 3) points in LiDAR frame
            
        Returns:
            pts_cam: (N, 3) points in camera frame
        """
        R_lidar2cam = np.array([
            [0, -1, 0],   # Cam_X = -LiDAR_Y (right = -left)
            [0, 0, -1],   # Cam_Y = -LiDAR_Z (down = -up)
            [1, 0, 0]     # Cam_Z = LiDAR_X (forward = forward)
        ], dtype=np.float64)
        
        pts_cam = (R_lidar2cam @ pts_lidar.T).T
        return pts_cam
    
    def project_to_image(self, pts_cam: np.ndarray, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project camera points to image coordinates using scaled intrinsics.
        
        Args:
            pts_cam: (N, 3) points in camera frame (X=right, Y=down, Z=forward)
            depth_map: Padded depth map (2064x1556)
            
        Returns:
            u, v: Pixel coordinates
            depths: Depth values (Z in camera frame)
        """
        depth_h, depth_w = depth_map.shape
        
        # Filter points in front of camera (Z > 0)
        valid = pts_cam[:, 2] > 0.1
        pts_valid = pts_cam[valid]
        
        if len(pts_valid) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Project using scaled intrinsics: u = fx * X/Z + cx, v = fy * Y/Z + cy
        depths = pts_valid[:, 2]
        u = (self.fx_scaled * pts_valid[:, 0] / depths + self.cx_scaled).astype(int)
        v = (self.fy_scaled * pts_valid[:, 1] / depths + self.cy_scaled).astype(int)
        
        # Filter within image bounds
        in_bounds = (u >= 0) & (u < depth_w) & (v >= 0) & (v < depth_h)
        
        return u[in_bounds], v[in_bounds], depths[in_bounds]
    
    def get_depth_at_pixels(self, depth_map: np.ndarray, u: np.ndarray, v: np.ndarray, 
                            depth_scale: float = None) -> np.ndarray:
        """Sample depth map at pixel locations and convert to metric depth."""
        if depth_scale is None:
            depth_scale = self.depth_scale
        
        values = depth_map[v, u].astype(np.float64)
        
        if self.inverse_depth:
            # Inverse depth stored: metric = scale / value
            valid = values > 1e-6
            depths = np.full_like(values, np.nan)
            depths[valid] = depth_scale / values[valid]
        else:
            # Direct depth stored: metric = value * scale
            valid = values > 1e-6
            depths = np.full_like(values, np.nan)
            depths[valid] = values[valid] * depth_scale
        
        return depths
    
    def evaluate_single_frame(self, 
                               pts_lidar: np.ndarray,
                               depth_map: np.ndarray,
                               meta: dict,
                               depth_scale: float = None) -> ReprojectionResult:
        """Evaluate reprojection error for a single frame."""
        idx = meta.get('index', 'unknown')
        
        if depth_scale is None:
            depth_scale = self.depth_scale
        
        # Transform LiDAR to camera frame
        pts_cam = self.transform_lidar_to_camera(pts_lidar)
        
        # Project to image
        u, v, depths_lidar = self.project_to_image(pts_cam, depth_map)
        
        if len(u) == 0:
            return ReprojectionResult(
                index=idx, n_lidar_pts=len(pts_lidar), n_projected=0, n_valid=0,
                depth_mae=np.inf, depth_rmse=np.inf, depth_median=np.inf,
                rel_mae=np.inf, rel_rmse=np.inf,
                inlier_1m=0, inlier_2m=0, inlier_5m=0,
                scale_ratio=0, scale_std=np.inf,
                icp_fitness=meta.get('fitness', 0),
                icp_rmse=meta.get('rmse', 0),
                chamfer=meta.get('chamfer', np.inf)
            )
        
        # Get predicted depths from depth map
        depths_pred = self.get_depth_at_pixels(depth_map, u, v, depth_scale)
        
        # Filter valid comparisons (exclude padded regions with 0 depth)
        valid = ~np.isnan(depths_pred) & (depths_pred > 0) & (depths_lidar > 0)
        depths_lidar_valid = depths_lidar[valid]
        depths_pred_valid = depths_pred[valid]
        
        if len(depths_lidar_valid) == 0:
            return ReprojectionResult(
                index=idx, n_lidar_pts=len(pts_lidar), n_projected=len(u), n_valid=0,
                depth_mae=np.inf, depth_rmse=np.inf, depth_median=np.inf,
                rel_mae=np.inf, rel_rmse=np.inf,
                inlier_1m=0, inlier_2m=0, inlier_5m=0,
                scale_ratio=0, scale_std=np.inf,
                icp_fitness=meta.get('fitness', 0),
                icp_rmse=meta.get('rmse', 0),
                chamfer=meta.get('chamfer', np.inf)
            )
        
        # Compute errors
        errors = np.abs(depths_lidar_valid - depths_pred_valid)
        rel_errors = errors / depths_lidar_valid
        scale_ratios = depths_lidar_valid / depths_pred_valid
        
        return ReprojectionResult(
            index=idx,
            n_lidar_pts=len(pts_lidar),
            n_projected=len(u),
            n_valid=len(depths_lidar_valid),
            depth_mae=np.mean(errors),
            depth_rmse=np.sqrt(np.mean(errors ** 2)),
            depth_median=np.median(errors),
            rel_mae=np.mean(rel_errors),
            rel_rmse=np.sqrt(np.mean(rel_errors ** 2)),
            inlier_1m=np.mean(errors < 1.0),
            inlier_2m=np.mean(errors < 2.0),
            inlier_5m=np.mean(errors < 5.0),
            scale_ratio=np.median(scale_ratios),
            scale_std=np.std(scale_ratios),
            icp_fitness=meta.get('fitness', 0),
            icp_rmse=meta.get('rmse', 0),
            chamfer=meta.get('chamfer', np.inf)
        )
    
    def evaluate_method(self, 
                        method_name: str,
                        depth_dir: str,
                        metadata_path: str,
                        depth_scale: float = None,
                        verbose: bool = True) -> List[ReprojectionResult]:
        """
        Evaluate a depth estimation method.
        
        Args:
            method_name: Name for this method (e.g., "depth_anything_v3")
            depth_dir: Directory containing depth_*.npy files
            metadata_path: Path to metadata.jsonl file
            depth_scale: Override depth scale (uses default if None)
            verbose: Print progress
        """
        depth_dir = Path(depth_dir)
        
        if depth_scale is None:
            depth_scale = self.depth_scale
        
        # Load metadata
        metadata_list = []
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    metadata_list.append(json.loads(line))
        
        if verbose:
            print(f"\nEvaluating {method_name}: {len(metadata_list)} frames")
            print(f"  Scale factor: {self.scale}x")
            print(f"  Scaled intrinsics: fx={self.fx_scaled:.2f}, fy={self.fy_scaled:.2f}, "
                  f"cx={self.cx_scaled:.2f}, cy={self.cy_scaled:.2f}")
            print(f"  Target padded size: {self.target_width}x{self.target_height}")
        
        # Check first depth map
        first_idx = metadata_list[0]['index'] if metadata_list else None
        if first_idx:
            first_depth_path = depth_dir / f"depth_{first_idx}.npy"
            if first_depth_path.exists():
                first_depth_raw = np.load(first_depth_path)
                if verbose:
                    print(f"  Raw depth map: {first_depth_raw.shape[1]}x{first_depth_raw.shape[0]}")
                    print(f"  After padding: {self.target_width}x{self.target_height}")
        
        results = []
        iterator = tqdm(metadata_list, desc=method_name) if verbose else metadata_list
        
        for meta in iterator:
            idx = meta['index']
            
            # Load LiDAR
            pts_lidar = self.load_lidar(idx)
            if pts_lidar is None:
                continue
            
            # Load and pad depth
            depth_path = depth_dir / f"depth_{idx}.npy"
            if not depth_path.exists():
                continue
            depth_map = self.load_depth(depth_path)
            
            # Evaluate
            result = self.evaluate_single_frame(pts_lidar, depth_map, meta, depth_scale)
            results.append(result)
        
        self.methods[method_name] = results
        
        if verbose:
            valid_results = [r for r in results if r.n_valid > 0]
            print(f"  Evaluated: {len(valid_results)}/{len(results)} valid frames")
            if valid_results:
                mae = np.mean([r.depth_mae for r in valid_results])
                inlier_2m = np.mean([r.inlier_2m for r in valid_results])
                scale = np.mean([r.scale_ratio for r in valid_results])
                print(f"  MAE: {mae:.3f}m, Inlier<2m: {inlier_2m*100:.1f}%, Scale: {scale:.3f}")
        
        return results
    
    def auto_calibrate_scale(self, 
                              depth_dir: str, 
                              metadata_path: str, 
                              num_samples: int = 10) -> float:
        """
        Automatically estimate optimal depth scale from samples.
        """
        depth_dir = Path(depth_dir)
        
        with open(metadata_path, 'r') as f:
            metas = [json.loads(line) for line in f if line.strip()][:num_samples]
        
        all_ratios = []
        
        for meta in metas:
            idx = meta['index']
            
            pts_lidar = self.load_lidar(idx)
            if pts_lidar is None:
                continue
            
            depth_path = depth_dir / f"depth_{idx}.npy"
            if not depth_path.exists():
                continue
            
            depth_map = self.load_depth(depth_path)  # This pads the depth map
            
            pts_cam = self.transform_lidar_to_camera(pts_lidar)
            u, v, depths_lidar = self.project_to_image(pts_cam, depth_map)
            
            if len(u) == 0:
                continue
            
            raw_values = depth_map[v, u].astype(np.float64)
            valid = (raw_values > 1e-6) & (depths_lidar > 0)
            
            if valid.sum() > 0:
                # For relative depth: scale = lidar_depth / raw_value
                ratios = depths_lidar[valid] / raw_values[valid]
                all_ratios.extend(ratios)
        
        if not all_ratios:
            print("Warning: Could not calibrate scale, using default")
            return self.depth_scale
        
        optimal_scale = np.median(all_ratios)
        print(f"Auto-calibrated depth_scale: {optimal_scale:.2f}")
        return optimal_scale
    
    def get_statistics(self) -> pd.DataFrame:
        """Compute summary statistics for all methods."""
        stats_list = []
        
        for method, results in self.methods.items():
            valid = [r for r in results if r.n_valid > 0 and r.depth_mae < 100]
            
            if not valid:
                continue
            
            mae = [r.depth_mae for r in valid]
            rmse = [r.depth_rmse for r in valid]
            median = [r.depth_median for r in valid]
            rel_mae = [r.rel_mae for r in valid]
            inlier_1m = [r.inlier_1m for r in valid]
            inlier_2m = [r.inlier_2m for r in valid]
            inlier_5m = [r.inlier_5m for r in valid]
            scale = [r.scale_ratio for r in valid]
            
            stats_list.append({
                'method': method,
                'n_frames': len(valid),
                'mae_mean': np.mean(mae),
                'mae_std': np.std(mae),
                'mae_median': np.median(mae),
                'rmse_mean': np.mean(rmse),
                'median_mean': np.mean(median),
                'rel_mae_mean': np.mean(rel_mae),
                'inlier_1m': np.mean(inlier_1m) * 100,
                'inlier_2m': np.mean(inlier_2m) * 100,
                'inlier_5m': np.mean(inlier_5m) * 100,
                'scale_ratio': np.mean(scale),
                'scale_std': np.std(scale),
            })
        
        return pd.DataFrame(stats_list)
    
    def compare_methods(self, metric: str = 'depth_mae') -> pd.DataFrame:
        """Statistical comparison between methods."""
        method_names = list(self.methods.keys())
        comparisons = []
        
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                v1 = [r.__dict__[metric] for r in self.methods[m1] if r.n_valid > 0 and r.__dict__[metric] < 100]
                v2 = [r.__dict__[metric] for r in self.methods[m2] if r.n_valid > 0 and r.__dict__[metric] < 100]
                
                if not v1 or not v2:
                    continue
                
                stat, pval = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                mean_diff = np.mean(v1) - np.mean(v2)
                pooled_std = np.sqrt((np.std(v1)**2 + np.std(v2)**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                comparisons.append({
                    'method1': m1,
                    'method2': m2,
                    f'{metric}_1': np.mean(v1),
                    f'{metric}_2': np.mean(v2),
                    'difference': mean_diff,
                    'p_value': pval,
                    'cohens_d': cohens_d,
                    'significant': pval < 0.05,
                    'better': m1 if mean_diff < 0 else m2
                })
        
        return pd.DataFrame(comparisons)
    
    def plot_comparison(self, save_path: str = None):
        """Create comparison plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Depth Method Comparison (Reprojection Metrics)', fontsize=16, fontweight='bold')
        
        metrics = [
            ('depth_mae', 'MAE (m)'),
            ('depth_median', 'Median Error (m)'),
            ('rel_mae', 'Relative MAE'),
            ('inlier_2m', 'Inlier <2m (%)'),
            ('scale_ratio', 'Scale Ratio'),
            ('inlier_5m', 'Inlier <5m (%)')
        ]
        
        for ax, (metric, label) in zip(axes.flatten(), metrics):
            data = []
            for method, results in self.methods.items():
                valid = [r for r in results if r.n_valid > 0 and r.depth_mae < 100]
                for r in valid:
                    val = r.__dict__[metric]
                    if metric in ['inlier_2m', 'inlier_5m']:
                        val *= 100
                    data.append({'method': method, 'value': val})
            
            if data:
                df = pd.DataFrame(data)
                sns.boxplot(data=df, x='method', y='value', ax=ax)
                ax.set_ylabel(label)
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_per_scene(self, save_path: str = None):
        """Plot per-scene comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = [('depth_mae', 'MAE (m)'), ('inlier_2m', 'Inlier <2m'), ('scale_ratio', 'Scale Ratio')]
        
        for ax, (metric, label) in zip(axes, metrics):
            for method, results in self.methods.items():
                sorted_r = sorted([r for r in results if r.n_valid > 0], key=lambda x: x.index)
                indices = [r.index for r in sorted_r]
                values = [r.__dict__[metric] for r in sorted_r]
                if metric == 'inlier_2m':
                    values = [v * 100 for v in values]
                ax.plot(range(len(values)), values, marker='.', label=method, alpha=0.7)
            
            ax.set_xlabel('Frame')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, output_path: str = 'evaluation_report.txt'):
        """Generate text report."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEPTH METHOD EVALUATION REPORT (Reprojection-based)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Original image size: {self.original_image_size[0]} x {self.original_image_size[1]}\n")
            f.write(f"Scale factor: {self.scale}x\n")
            f.write(f"Target padded size: {self.target_width} x {self.target_height}\n")
            f.write(f"Scaled intrinsics: fx={self.fx_scaled:.2f}, fy={self.fy_scaled:.2f}, "
                   f"cx={self.cx_scaled:.2f}, cy={self.cy_scaled:.2f}\n")
            f.write(f"Default depth scale: {self.depth_scale}\n")
            f.write(f"Inverse depth: {self.inverse_depth}\n\n")
            
            stats_df = self.get_statistics()
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(stats_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("PAIRWISE COMPARISON (MAE)\n")
            f.write("-" * 80 + "\n")
            comp_df = self.compare_methods('depth_mae')
            if not comp_df.empty:
                f.write(comp_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"Report saved: {output_path}")

    def evaluate_method_debug(self, 
                              method_name: str,
                              depth_dir: str,
                              metadata_path: str,
                              depth_scale: float = None):
        """Debug version with visualization."""
        import matplotlib.pyplot as plt
        
        depth_dir = Path(depth_dir)
        
        if depth_scale is None:
            depth_scale = self.depth_scale
        
        with open(metadata_path, 'r') as f:
            meta = json.loads(f.readline())
        
        idx = meta['index']
        print(f"\n{'='*60}")
        print(f"DEBUG Frame {idx}")
        print(f"{'='*60}")
        
        # Load LiDAR
        pts_lidar = self.load_lidar(idx)
        if pts_lidar is None:
            print(f"ERROR: Cannot load LiDAR for {idx}")
            return
        print(f"\n[LiDAR]")
        print(f"  Points: {len(pts_lidar)}")
        print(f"  X (forward): [{pts_lidar[:,0].min():.2f}, {pts_lidar[:,0].max():.2f}]")
        print(f"  Y (left):    [{pts_lidar[:,1].min():.2f}, {pts_lidar[:,1].max():.2f}]")
        print(f"  Z (up):      [{pts_lidar[:,2].min():.2f}, {pts_lidar[:,2].max():.2f}]")
        
        # Load depth (raw, before padding)
        depth_path = depth_dir / f"depth_{idx}.npy"
        if not depth_path.exists():
            print(f"ERROR: Cannot load depth from {depth_path}")
            return
        depth_raw = np.load(depth_path)
        print(f"\n[Depth Map (raw)]")
        print(f"  Shape: {depth_raw.shape[1]} x {depth_raw.shape[0]}")
        print(f"  Range: [{depth_raw.min():.4f}, {depth_raw.max():.4f}]")
        print(f"  Mean: {depth_raw.mean():.4f}")
        
        # Pad depth map
        depth_map = self.pad_depth_map(depth_raw)
        depth_h, depth_w = depth_map.shape
        print(f"\n[Depth Map (padded)]")
        print(f"  Shape: {depth_w} x {depth_h}")
        print(f"  Padding added: {depth_h - depth_raw.shape[0]} rows at bottom")
        
        # Intrinsics
        print(f"\n[Scaled Intrinsics (2x)]")
        print(f"  fx={self.fx_scaled:.2f}, fy={self.fy_scaled:.2f}")
        print(f"  cx={self.cx_scaled:.2f}, cy={self.cy_scaled:.2f}")
        
        # Coordinate frame change
        print(f"\n[Coordinate Frame Change]")
        pts_cam = self.transform_lidar_to_camera(pts_lidar)
        print(f"  X (right): [{pts_cam[:,0].min():.2f}, {pts_cam[:,0].max():.2f}]")
        print(f"  Y (down):  [{pts_cam[:,1].min():.2f}, {pts_cam[:,1].max():.2f}]")
        print(f"  Z (fwd):   [{pts_cam[:,2].min():.2f}, {pts_cam[:,2].max():.2f}]")
        
        # Filter points in front of camera
        in_front = pts_cam[:, 2] > 0.1
        pts_front = pts_cam[in_front]
        print(f"  Points in front (Z > 0.1): {in_front.sum()}")
        
        if in_front.sum() == 0:
            print("ERROR: No points in front of camera!")
            return
        
        # Project to image
        depths = pts_front[:, 2]
        u_all = self.fx_scaled * pts_front[:, 0] / depths + self.cx_scaled
        v_all = self.fy_scaled * pts_front[:, 1] / depths + self.cy_scaled
        
        print(f"\n[Projection]")
        print(f"  u range: [{u_all.min():.1f}, {u_all.max():.1f}] (width: 0-{depth_w})")
        print(f"  v range: [{v_all.min():.1f}, {v_all.max():.1f}] (height: 0-{depth_h})")
        
        # Filter within image bounds
        in_bounds = (u_all >= 0) & (u_all < depth_w) & (v_all >= 0) & (v_all < depth_h)
        print(f"  Points in image bounds: {in_bounds.sum()}")
        
        if in_bounds.sum() == 0:
            print("\n[ISSUE] No points project into image!")
            return
        
        u = u_all[in_bounds].astype(int)
        v = v_all[in_bounds].astype(int)
        depths_lidar = depths[in_bounds]
        
        print(f"  Projected u: [{u.min()}, {u.max()}]")
        print(f"  Projected v: [{v.min()}, {v.max()}]")
        print(f"  LiDAR depths: [{depths_lidar.min():.2f}, {depths_lidar.max():.2f}]")
        
        # Check how many points are in padded region vs original
        in_original = v < depth_raw.shape[0]
        print(f"  Points in original region (v < {depth_raw.shape[0]}): {in_original.sum()}")
        print(f"  Points in padded region: {(~in_original).sum()}")
        
        # Get depth map values at projected locations
        raw_values = depth_map[v, u]
        valid_mask = raw_values > 1e-6
        
        print(f"\n[Depth Map Sampling]")
        print(f"  Raw values range: [{raw_values[valid_mask].min():.4f}, {raw_values[valid_mask].max():.4f}]")
        print(f"  Valid pixels (>1e-6): {valid_mask.sum()}")
        print(f"  Invalid (in padded region): {(~valid_mask).sum()}")
        
        if valid_mask.sum() == 0:
            print("ERROR: No valid depth values at projected locations!")
            return
        
        # Auto-calibrate scale
        auto_scale = np.median(depths_lidar[valid_mask] / raw_values[valid_mask])
        print(f"  Auto-calibrated scale: {auto_scale:.2f}")
        print(f"  (Provided depth_scale: {depth_scale})")
        
        # Compute depths with auto scale
        depths_pred_auto = raw_values * auto_scale
        
        # Compute depths with provided scale
        depths_pred_provided = raw_values * depth_scale
        
        # Compute errors with auto scale
        errors_auto = np.abs(depths_lidar[valid_mask] - depths_pred_auto[valid_mask])
        
        # Compute errors with provided scale
        errors_provided = np.abs(depths_lidar[valid_mask] - depths_pred_provided[valid_mask])
        
        print(f"\n[Results with AUTO scale ({auto_scale:.2f})]")
        print(f"  Predicted depths: [{depths_pred_auto[valid_mask].min():.2f}, {depths_pred_auto[valid_mask].max():.2f}]")
        print(f"  MAE: {errors_auto.mean():.2f}m")
        print(f"  Median: {np.median(errors_auto):.2f}m")
        print(f"  RMSE: {np.sqrt((errors_auto**2).mean()):.2f}m")
        print(f"  <1m: {(errors_auto < 1).mean()*100:.1f}%")
        print(f"  <2m: {(errors_auto < 2).mean()*100:.1f}%")
        print(f"  <5m: {(errors_auto < 5).mean()*100:.1f}%")
        
        print(f"\n[Results with PROVIDED scale ({depth_scale})]")
        print(f"  Predicted depths: [{depths_pred_provided[valid_mask].min():.2f}, {depths_pred_provided[valid_mask].max():.2f}]")
        print(f"  MAE: {errors_provided.mean():.2f}m")
        print(f"  Median: {np.median(errors_provided):.2f}m")
        print(f"  RMSE: {np.sqrt((errors_provided**2).mean()):.2f}m")
        print(f"  <1m: {(errors_provided < 1).mean()*100:.1f}%")
        print(f"  <2m: {(errors_provided < 2).mean()*100:.1f}%")
        print(f"  <5m: {(errors_provided < 5).mean()*100:.1f}%")
        
        # ============ VISUALIZATION ============
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Debug Frame {idx} - {method_name}\n'
                     f'Scale: auto={auto_scale:.1f}, provided={depth_scale}', 
                     fontsize=14, fontweight='bold')
        
        vmin, vmax = 0, min(50, depths_lidar.max() + 5)
        
        # 1. Depth map with LiDAR overlay
        ax = axes[0, 0]
        depth_vis = depth_map * auto_scale
        im = ax.imshow(depth_vis, cmap='turbo', vmin=vmin, vmax=vmax)
        ax.scatter(u[valid_mask], v[valid_mask], c=depths_lidar[valid_mask], 
                   cmap='turbo', s=2, vmin=vmin, vmax=vmax, 
                   edgecolors='white', linewidths=0.1)
        # Draw line showing original vs padded region
        ax.axhline(y=depth_raw.shape[0], color='red', linestyle='--', linewidth=1, label='Padding boundary')
        ax.set_title('Depth Map + LiDAR overlay')
        ax.set_xlim(0, depth_w)
        ax.set_ylim(depth_h, 0)
        ax.legend(loc='lower right')
        plt.colorbar(im, ax=ax, label='Depth (m)')
        
        # 2. LiDAR projected depth only
        ax = axes[0, 1]
        ax.imshow(np.zeros((depth_h, depth_w, 3), dtype=np.uint8) + 40)
        sc = ax.scatter(u[valid_mask], v[valid_mask], c=depths_lidar[valid_mask], 
                        cmap='turbo', s=3, vmin=vmin, vmax=vmax)
        ax.axhline(y=depth_raw.shape[0], color='red', linestyle='--', linewidth=1)
        ax.set_title('LiDAR Projected Depth')
        ax.set_xlim(0, depth_w)
        ax.set_ylim(depth_h, 0)
        plt.colorbar(sc, ax=ax, label='Depth (m)')
        
        # 3. Error map
        ax = axes[0, 2]
        ax.imshow(np.zeros((depth_h, depth_w, 3), dtype=np.uint8) + 40)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        sc = ax.scatter(u_valid, v_valid, c=errors_auto, cmap='hot', s=3, vmin=0, vmax=15)
        ax.axhline(y=depth_raw.shape[0], color='cyan', linestyle='--', linewidth=1)
        ax.set_title(f'Error Map (MAE={errors_auto.mean():.2f}m)')
        ax.set_xlim(0, depth_w)
        ax.set_ylim(depth_h, 0)
        plt.colorbar(sc, ax=ax, label='Error (m)')
        
        # 4. Scatter: LiDAR vs Predicted depth
        ax = axes[1, 0]
        ax.scatter(depths_lidar[valid_mask], depths_pred_auto[valid_mask], alpha=0.3, s=1, label='Auto scale')
        max_d = max(depths_lidar[valid_mask].max(), depths_pred_auto[valid_mask].max())
        ax.plot([0, max_d], [0, max_d], 'r--', linewidth=2, label='Perfect alignment')
        ax.set_xlabel('LiDAR Depth (m)')
        ax.set_ylabel('Predicted Depth (m)')
        ax.set_title('Depth Correlation')
        ax.legend()
        ax.set_xlim(0, max_d)
        ax.set_ylim(0, max_d)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 5. Error vs Distance
        ax = axes[1, 1]
        ax.scatter(depths_lidar[valid_mask], errors_auto, alpha=0.3, s=1)
        ax.axhline(y=errors_auto.mean(), color='r', linestyle='--', label=f'MAE={errors_auto.mean():.2f}m')
        ax.set_xlabel('LiDAR Depth (m)')
        ax.set_ylabel('Error (m)')
        ax.set_title('Error vs Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, min(30, errors_auto.max() + 2))
        
        # 6. Error histogram
        ax = axes[1, 2]
        ax.hist(errors_auto, bins=50, range=(0, 30), edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(errors_auto.mean(), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {errors_auto.mean():.2f}m')
        ax.axvline(np.median(errors_auto), color='b', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(errors_auto):.2f}m')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add stats text
        stats_text = (
            f"Points: {valid_mask.sum()}\n"
            f"MAE: {errors_auto.mean():.2f}m\n"
            f"Median: {np.median(errors_auto):.2f}m\n"
            f"RMSE: {np.sqrt((errors_auto**2).mean()):.2f}m\n"
            f"<2m: {(errors_auto < 2).mean()*100:.1f}%\n"
            f"<5m: {(errors_auto < 5).mean()*100:.1f}%"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        save_path = f'debug_{idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
        plt.show()
        
        return {
            'idx': idx,
            'u': u, 'v': v,
            'valid_mask': valid_mask,
            'depths_lidar': depths_lidar,
            'depths_pred_auto': depths_pred_auto,
            'errors_auto': errors_auto,
            'auto_scale': auto_scale,
        }


def quick_evaluate(
    methods_config: Dict[str, Tuple[str, str]],
    lidar_dir: str,
    K: np.ndarray,
    output_dir: str = './evaluation',
    original_image_size: Tuple[int, int] = (1032, 778),
    depth_scale: float = 100.0,
    inverse_depth=False,
    auto_calibrate: bool = True
):
    """
    Quick evaluation of multiple methods.
    
    Args:
        methods_config: Dict mapping method_name -> (depth_dir, metadata_path)
        lidar_dir: Directory with LiDAR files
        K: Camera intrinsics (for original image size)
        output_dir: Output directory
        original_image_size: (width, height) of original image
        depth_scale: Default depth scale
        auto_calibrate: Auto-calibrate scale for each method
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    evaluator = DepthMethodEvaluator(
        K, 
        lidar_dir, 
        original_image_size=original_image_size,
        depth_scale=depth_scale,
        inverse_depth=inverse_depth,
    )
    
    for method_name, (depth_dir, metadata_path) in methods_config.items():
        if auto_calibrate:
            scale = evaluator.auto_calibrate_scale(str(depth_dir), str(metadata_path))
        else:
            scale = depth_scale
        evaluator.evaluate_method(method_name, str(depth_dir), str(metadata_path), depth_scale=scale)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    stats_df = evaluator.get_statistics()
    print("\nSummary:")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(output_dir / 'summary_stats.csv', index=False)
    
    comp_df = evaluator.compare_methods('depth_mae')
    if not comp_df.empty:
        print("\nComparison:")
        print(comp_df.to_string(index=False))
        comp_df.to_csv(output_dir / 'comparison.csv', index=False)
    
    evaluator.plot_comparison(output_dir / 'comparison.png')
    evaluator.plot_per_scene(output_dir / 'per_scene.png')
    evaluator.generate_report(output_dir / 'report.txt')
    
    print(f"\nOutputs saved to: {output_dir}")
    
    return evaluator


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    K = np.array([
        [491.331107883326, 0.0, 515.3434363622374],
        [0.0, 492.14998153326013, 388.93983736974667],
        [0.0, 0.0, 1.0]
    ])
    
    DATA_ROOT = Path("../dataset")
    LIDAR_DIR = DATA_ROOT / "lidar"
    METADATA_PATH = DATA_ROOT / "aligned/metadata.jsonl"
    
    methods = {
        'depth_anything_v2': (DATA_ROOT / "depth_depth_anything_v2", METADATA_PATH),
        'depth_anything_v3': (DATA_ROOT / "depth_depth_anything_v3", METADATA_PATH),
        'midas': (DATA_ROOT / "depth_midas", METADATA_PATH),
        'marigold': (DATA_ROOT / "depth_marigold", METADATA_PATH),
        'depth_pro': (DATA_ROOT / "depth_depth_pro", METADATA_PATH),
    }
    
    methods = {k: (str(v[0]), str(v[1])) for k, v in methods.items() if Path(v[0]).exists()}
    
    if not methods:
        print("No depth directories found!")
    else:
        quick_evaluate(
            methods_config=methods,
            lidar_dir=str(LIDAR_DIR),
            K=K,
            output_dir='./evaluation_results',
            original_image_size=(1032, 778),
            depth_scale=100.0,
            auto_calibrate=True
        )
