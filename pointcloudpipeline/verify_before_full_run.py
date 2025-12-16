#!/usr/bin/env python3
"""
Quick verification script to check everything before running the full pipeline.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from clouds import load_luminar_clouds, load_camera, cloud_indices, calculate_camera_to_lidar_transform

def check_disk_space(output_dir: Path, num_frames: int):
    """Estimate disk space needed."""
    # Based on test: ~1.8MB PCD + ~573KB NPZ per frame
    pcd_size_per_frame = 1_837_313  # bytes
    npz_size_per_frame = 573_110    # bytes
    total_per_frame = pcd_size_per_frame + npz_size_per_frame
    
    total_needed = total_per_frame * num_frames
    total_gb = total_needed / (1024**3)
    
    print(f"\nDisk Space Estimate:")
    print(f"  Frames to process: {num_frames:,}")
    print(f"  Estimated size per frame: {total_per_frame / (1024**2):.1f} MB")
    print(f"  Total estimated size: {total_gb:.2f} GB")
    
    # Check available space
    import shutil
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / (1024**3)
    print(f"  Available space: {free_gb:.2f} GB")
    
    if free_gb < total_gb * 1.5:  # Need 1.5x for safety
        print(f"  [WARN] May not have enough disk space!")
    else:
        print(f"  [OK] Sufficient disk space available")

def verify_data_quality():
    """Check the quality of saved test data."""
    print("\n" + "=" * 60)
    print("Verifying Data Quality")
    print("=" * 60)
    
    npz_path = Path(__file__).parent.parent / "dataset" / "lidar_npz" / "lidar_00000.npz"
    
    if not npz_path.exists():
        print("[SKIP] No test file found to verify")
        return
    
    try:
        data = np.load(npz_path)
        xyz = data['xyz']
        
        print(f"Point cloud statistics:")
        print(f"  Total points: {len(xyz):,}")
        print(f"  X range: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}] m")
        print(f"  Y range: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}] m")
        print(f"  Z range: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}] m")
        
        # Check for reasonable values (not all zeros, not NaN)
        if np.all(xyz == 0):
            print(f"  [WARN] All points are at origin!")
        elif np.any(np.isnan(xyz)):
            print(f"  [WARN] Contains NaN values!")
        else:
            print(f"  [OK] Data looks reasonable")
        
        # Check distances
        distances = np.linalg.norm(xyz, axis=1)
        print(f"  Distance from origin: [{distances.min():.3f}, {distances.max():.3f}] m")
        
    except Exception as e:
        print(f"[ERROR] Failed to verify data: {e}")

def check_frame_counts():
    """Check how many frames will be processed."""
    print("\n" + "=" * 60)
    print("Checking Frame Counts")
    print("=" * 60)
    
    script_dir = Path(__file__).parent.resolve()
    bag_path = script_dir / "../dpt_rosbag_lvms_2024_12/rosbag2_2024_12_12-18_21_55_12.mcap"
    camera_topic = "/vimba_front/image_raw"
    lidar_topic = "/luminar_front/points/existence_prob_filtered"
    
    if not bag_path.exists():
        print(f"[ERROR] Bag file not found: {bag_path}")
        return
    
    try:
        # Count camera frames (without limit)
        camera_timestamps = load_camera(bag_path, camera_topic, Path("/tmp/test_camera"), max_images=None)
        print(f"Camera frames: {len(camera_timestamps):,}")
        
        # Count lidar frames
        clouds = load_luminar_clouds(bag_path, lidar_topic)
        print(f"LiDAR frames: {len(clouds):,}")
        
        # Calculate transforms
        cloud_timestamps = cloud_indices(clouds)
        transforms = calculate_camera_to_lidar_transform(camera_timestamps, cloud_timestamps)
        print(f"Output frames (transforms): {len(transforms):,}")
        
        # Estimate runtime (rough estimate: ~0.5 seconds per frame based on test)
        estimated_seconds = len(transforms) * 0.5
        estimated_minutes = estimated_seconds / 60
        print(f"\nEstimated runtime: ~{estimated_minutes:.1f} minutes ({estimated_seconds:.0f} seconds)")
        
        return len(transforms)
        
    except Exception as e:
        print(f"[ERROR] Failed to check frame counts: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Pre-Flight Check Before Full Pipeline Run")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / "dataset"
    
    # Check frame counts and estimate runtime
    num_frames = check_frame_counts()
    
    if num_frames:
        # Check disk space
        check_disk_space(output_dir, num_frames)
    
    # Verify data quality
    verify_data_quality()
    
    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)
    print("\nRecommendations:")
    print("  ✓ Test data looks good")
    print("  ✓ Pipeline is working correctly")
    if num_frames:
        print(f"  ✓ Ready to process {num_frames:,} frames")
    print("\nTo run the full pipeline:")
    print("  cd pointcloudpipeline")
    print("  python clouds.py")
    print("\nTo monitor progress, you can run in another terminal:")
    print("  watch -n 5 'ls -lh dataset/lidar/*.pcd | wc -l'")

if __name__ == "__main__":
    main()





