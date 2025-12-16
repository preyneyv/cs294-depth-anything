#!/usr/bin/env python3
"""
Test script for clouds.py

This script helps test and debug the pointcloud pipeline functions.
"""

import sys
from pathlib import Path

# Add the parent directory to path so we can import clouds
sys.path.insert(0, str(Path(__file__).parent))

from clouds import (
    load_luminar_clouds,
    load_camera,
    cloud_indices,
    find_surrounding_cloud_indices,
    calculate_camera_to_lidar_transform,
    interpolate_luminar_frames,
    write_pcd_ascii,
    write_lidar_npz,
)

def test_load_luminar_clouds():
    """Test loading LiDAR point clouds from bag file."""
    print("=" * 60)
    print("Testing load_luminar_clouds()")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    bag_path = script_dir / "../dpt_rosbag_lvms_2024_12/rosbag2_2024_12_12-18_21_55_12.mcap"
    
    if not bag_path.exists():
        print(f"[SKIP] Bag file not found: {bag_path}")
        return None
    
    try:
        clouds = load_luminar_clouds(bag_path)
        print(f"[OK] Loaded {len(clouds)} LiDAR point cloud frames")
        if len(clouds) > 0:
            ts, arr = clouds[0]
            print(f"  First frame: timestamp={ts}, points={len(arr)}")
        return clouds
    except Exception as e:
        print(f"[ERROR] Failed to load clouds: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_load_camera():
    """Test loading camera images from bag file."""
    print("\n" + "=" * 60)
    print("Testing load_camera()")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    bag_path = script_dir / "../dpt_rosbag_lvms_2024_12/rosbag2_2024_12_12-18_21_55_12.mcap"
    out_dir = script_dir / "../dataset/test_camera"
    camera_topic = "/vimba_front/image_raw"
    
    if not bag_path.exists():
        print(f"[SKIP] Bag file not found: {bag_path}")
        return None
    
    try:
        # Test with max_images=5 to limit output
        camera_timestamps = load_camera(bag_path, camera_topic, out_dir, max_images=5)
        print(f"[OK] Loaded {len(camera_timestamps)} camera frames (limited to 5 for testing)")
        if len(camera_timestamps) > 0:
            ts, idx = camera_timestamps[0]
            print(f"  First frame: timestamp={ts}, index={idx}")
        return camera_timestamps
    except Exception as e:
        print(f"[ERROR] Failed to load camera: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_find_surrounding_cloud_indices():
    """Test finding surrounding cloud indices."""
    print("\n" + "=" * 60)
    print("Testing find_surrounding_cloud_indices()")
    print("=" * 60)
    
    # Create test data
    cloud_timestamps = [100, 200, 300, 400, 500]
    test_timestamps = [50, 150, 250, 350, 450, 550]
    
    for ts in test_timestamps:
        i0, i1 = find_surrounding_cloud_indices(ts, cloud_timestamps)
        print(f"  timestamp={ts} -> indices=({i0}, {i1}), times=({cloud_timestamps[i0]}, {cloud_timestamps[i1]})")
    
    print("[OK] find_surrounding_cloud_indices() test passed")

def test_calculate_camera_to_lidar_transform():
    """Test camera to LiDAR transform calculation."""
    print("\n" + "=" * 60)
    print("Testing calculate_camera_to_lidar_transform()")
    print("=" * 60)
    
    # Create test data
    camera_timestamps = [(150, 0), (250, 1), (350, 2)]
    cloud_timestamps = [100, 200, 300, 400, 500]
    
    try:
        transforms = calculate_camera_to_lidar_transform(camera_timestamps, cloud_timestamps)
        print(f"[OK] Generated {len(transforms)} transforms")
        for cam_ts, img_idx, (c0, c1), scalar in transforms:
            print(f"  Camera frame {img_idx} (ts={cam_ts}): cloud_indices=({c0}, {c1}), scalar={scalar:.3f}")
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()

def test_interpolate_luminar_frames():
    """Test LiDAR frame interpolation."""
    print("\n" + "=" * 60)
    print("Testing interpolate_luminar_frames()")
    print("=" * 60)
    
    import numpy as np
    
    # Create minimal test structured arrays
    n_points = 100
    dtype = np.dtype({
        'names': ['timestamp', 'x', 'y', 'z', 'reflectance', 'return_index', 
                  'last_return_index', 'sensor_id', 'azimuth', 'elevation', 
                  'depth', 'line_index', 'frame_index', 'detector_site_id',
                  'scan_checkpoint', 'existence_prob', 'data_qualifier', 'blockage_level'],
        'formats': ['<u8', '<f4', '<f4', '<f4', '<f4', 'u1', 'u1', 'u1',
                   '<f4', '<f4', '<f4', '<u2', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1'],
        'offsets': [0, 8, 12, 16, 20, 24, 25, 26, 32, 36, 40, 44, 46, 47, 48, 49, 50, 51],
        'itemsize': 52
    })
    
    arr0 = np.zeros(10, dtype=dtype)
    arr1 = np.zeros(10, dtype=dtype)
    
    # Fill with test data
    for i in range(10):
        arr0[i] = (1000 + i, 1.0, 2.0, 3.0, 0.5, 1, 1, 0, 0.1 * i, 0.0, 1.0, i, 0, 0, 0, 100, 0, 0)
        arr1[i] = (2000 + i, 2.0, 3.0, 4.0, 0.6, 1, 1, 0, 0.1 * i, 0.0, 1.0, i, 0, 0, 0, 100, 0, 0)
    
    try:
        xyz, refl, ts, idx0, idx1 = interpolate_luminar_frames(arr0, arr1, t=0.5)
        print(f"[OK] Interpolated {len(xyz)} points")
        if len(xyz) > 0:
            print(f"  First point: xyz={xyz[0]}, reflectance={refl[0]}")
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()

def test_file_saving(clouds, camera_timestamps):
    """Test that files are actually saved correctly."""
    print("\n" + "=" * 60)
    print("Testing File Saving")
    print("=" * 60)
    
    import numpy as np
    import tempfile
    import shutil
    
    if not clouds or not camera_timestamps:
        print("[SKIP] Need clouds and camera_timestamps to test file saving")
        return
    
    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="test_clouds_"))
    print(f"  Using test directory: {test_dir}")
    
    try:
        cloud_timestamps = cloud_indices(clouds)
        transforms = calculate_camera_to_lidar_transform(camera_timestamps, cloud_timestamps)
        
        if len(transforms) == 0:
            print("[SKIP] No transforms to test")
            return
        
        # Test saving just the first frame
        camera_timestamp_ns, image_index, (cloud_index0, cloud_index1), scalar_value = transforms[0]
        timestamp0, cloud0 = clouds[cloud_index0]
        timestamp1, cloud1 = clouds[cloud_index1]
        
        # Interpolate
        xyz_mid, refl_mid, ts_mid, idx0, idx1 = interpolate_luminar_frames(
            cloud0, cloud1, t=scalar_value, azimuth_bins=36000
        )
        
        if xyz_mid.shape[0] == 0:
            print("[SKIP] No points in interpolated cloud")
            return
        
        # Test PCD saving
        pcd_dir = test_dir / "lidar"
        pcd_path = pcd_dir / f"lidar_{image_index:05d}.pcd"
        write_pcd_ascii(pcd_path, xyz_mid)
        
        if pcd_path.exists():
            file_size = pcd_path.stat().st_size
            print(f"[OK] Saved PCD file: {pcd_path}")
            print(f"  File size: {file_size:,} bytes")
            print(f"  Points: {xyz_mid.shape[0]:,}")
            
            # Verify PCD header
            with open(pcd_path, 'r') as f:
                header = f.read(200)  # Read first 200 chars
                if "POINTS" in header and "FIELDS x y z" in header:
                    print(f"  [OK] PCD header looks valid")
                else:
                    print(f"  [WARN] PCD header may be invalid")
        else:
            print(f"[ERROR] PCD file was not created: {pcd_path}")
        
        # Test NPZ saving
        npz_dir = test_dir / "lidar_npz"
        npz_path = npz_dir / f"lidar_{image_index:05d}.npz"
        write_lidar_npz(npz_path, xyz_mid, refl_mid, ts_mid, idx0, idx1)
        
        if npz_path.exists():
            file_size = npz_path.stat().st_size
            print(f"[OK] Saved NPZ file: {npz_path}")
            print(f"  File size: {file_size:,} bytes")
            
            # Verify NPZ contents
            loaded = np.load(npz_path)
            expected_keys = {'xyz', 'reflectance', 'timestamp_mid', 'src_idx0', 'src_idx1'}
            if set(loaded.keys()) == expected_keys:
                print(f"  [OK] NPZ contains all expected keys: {expected_keys}")
                print(f"  xyz shape: {loaded['xyz'].shape}")
                print(f"  reflectance shape: {loaded['reflectance'].shape}")
            else:
                print(f"  [WARN] NPZ keys mismatch. Expected: {expected_keys}, Got: {set(loaded.keys())}")
        else:
            print(f"[ERROR] NPZ file was not created: {npz_path}")
        
        print(f"\n  Test files saved to: {test_dir}")
        print(f"  (Files will be cleaned up automatically)")
        
    except Exception as e:
        print(f"[ERROR] File saving test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test directory
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"  [OK] Cleaned up test directory")


def main():
    """Run all tests."""
    print("Pointcloud Pipeline Test Suite")
    print("=" * 60)
    
    # Test individual functions
    test_find_surrounding_cloud_indices()
    test_calculate_camera_to_lidar_transform()
    test_interpolate_luminar_frames()
    
    # Test loading from bag file (requires actual bag file)
    clouds = test_load_luminar_clouds()
    camera_timestamps = test_load_camera()
    
    # If we have both, test the full pipeline
    if clouds and camera_timestamps:
        print("\n" + "=" * 60)
        print("Testing Full Pipeline Integration")
        print("=" * 60)
        
        cloud_timestamps = cloud_indices(clouds)
        transforms = calculate_camera_to_lidar_transform(camera_timestamps, cloud_timestamps)
        
        print(f"[OK] Generated {len(transforms)} camera-to-LiDAR transforms")
        if len(transforms) > 0:
            cam_ts, img_idx, (c0, c1), scalar = transforms[0]
            print(f"  Example: Camera frame {img_idx} -> LiDAR clouds ({c0}, {c1}) with scalar={scalar:.3f}")
        
        # Test file saving
        test_file_saving(clouds, camera_timestamps)
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()

