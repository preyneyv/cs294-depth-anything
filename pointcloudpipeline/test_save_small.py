#!/usr/bin/env python3
"""
Quick test to save a few frames to verify the pipeline works.
This saves only 2 camera frames to test the full pipeline.
"""

import sys
from pathlib import Path

# Add the parent directory to path so we can import clouds
sys.path.insert(0, str(Path(__file__).parent))

from clouds import (
    load_luminar_clouds,
    load_camera,
    cloud_indices,
    calculate_camera_to_lidar_transform,
    interpolate_luminar_frames,
    write_pcd_ascii,
    write_lidar_npz,
)

def main():
    """Run a small test that saves 2 frames."""
    script_dir = Path(__file__).parent.resolve()
    bag_path = script_dir / "../dpt_rosbag_lvms_2024_12/rosbag2_2024_12_12-18_21_55_12.mcap"
    out_dir = (script_dir / "../dataset").resolve()
    camera_topic = "/vimba_front/image"  # Use uncompressed image topic
    lidar_topic = "/luminar_front/points/existence_prob_filtered"
    
    print("=" * 60)
    print("Small Pipeline Test - Saving 2 frames")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()
    
    # Load camera frames (limit to 2)
    print("Loading camera frames...")
    camera_timestamps = load_camera(bag_path, camera_topic, out_dir / "camera", max_images=2)
    if not camera_timestamps:
        print("[ERROR] No camera frames found")
        return
    print(f"[OK] Loaded {len(camera_timestamps)} camera frames")
    
    # Load lidar clouds
    print("Loading LiDAR point clouds...")
    clouds = load_luminar_clouds(bag_path, lidar_topic)
    if not clouds:
        print("[ERROR] No lidar frames found")
        return
    print(f"[OK] Loaded {len(clouds)} lidar frames")
    
    # Calculate transforms
    print("Calculating camera-to-LiDAR transforms...")
    cloud_timestamps = cloud_indices(clouds)
    camera_to_lidar_transforms = calculate_camera_to_lidar_transform(camera_timestamps, cloud_timestamps)
    print(f"[OK] Generated {len(camera_to_lidar_transforms)} transforms")
    
    # Create output directories
    lidar_pcd_dir = out_dir / "lidar"
    lidar_npz_dir = out_dir / "lidar_npz"
    
    # Process and save each frame
    print("\nProcessing and saving frames...")
    for camera_timestamp_ns, image_index, (cloud_index0, cloud_index1), scalar_value in camera_to_lidar_transforms:
        timestamp0, cloud0 = clouds[cloud_index0]
        timestamp1, cloud1 = clouds[cloud_index1]
        
        # Interpolate
        xyz_mid, refl_mid, ts_mid, idx0, idx1 = interpolate_luminar_frames(
            cloud0, cloud1, t=scalar_value, azimuth_bins=36000
        )
        
        if xyz_mid.shape[0] == 0:
            print(f"[WARN] No points for image {image_index}; skipping")
            continue
        
        # Save files
        pcd_path = lidar_pcd_dir / f"lidar_{image_index:05d}.pcd"
        npz_path = lidar_npz_dir / f"lidar_{image_index:05d}.npz"
        
        write_pcd_ascii(pcd_path, xyz_mid)
        write_lidar_npz(npz_path, xyz_mid, refl_mid, ts_mid, idx0, idx1)
        
        print(f"[OK] Saved frame {image_index}: {pcd_path.name} ({xyz_mid.shape[0]:,} points)")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Files saved to: {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

