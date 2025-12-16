#!/usr/bin/env python3
"""
Inspect and verify saved point cloud results.

This script helps you verify that the pipeline saved files correctly
and inspect their contents.
"""

import sys
from pathlib import Path
import numpy as np

def inspect_pcd_file(pcd_path: Path):
    """Inspect a PCD file and print statistics."""
    print(f"\nInspecting PCD file: {pcd_path}")
    print("=" * 60)
    
    if not pcd_path.exists():
        print(f"[ERROR] File does not exist: {pcd_path}")
        return
    
    # Read header
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header_lines = []
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            header_lines.append(line.strip())
            data_start = i + 1
            break
        header_lines.append(line.strip())
    
    print("Header:")
    for line in header_lines:
        print(f"  {line}")
    
    # Count points
    num_points = 0
    if data_start:
        num_points = len(lines) - data_start
    
    print(f"\nStatistics:")
    print(f"  Total lines: {len(lines)}")
    print(f"  Data lines: {num_points}")
    
    # Read first few points
    if data_start and num_points > 0:
        print(f"\nFirst 3 points:")
        for i in range(min(3, num_points)):
            print(f"  {lines[data_start + i].strip()}")


def inspect_npz_file(npz_path: Path):
    """Inspect an NPZ file and print statistics."""
    print(f"\nInspecting NPZ file: {npz_path}")
    print("=" * 60)
    
    if not npz_path.exists():
        print(f"[ERROR] File does not exist: {npz_path}")
        return
    
    try:
        data = np.load(npz_path)
        
        print("Contents:")
        for key in data.keys():
            arr = data[key]
            print(f"  {key}:")
            print(f"    Shape: {arr.shape}")
            print(f"    Dtype: {arr.dtype}")
            if arr.size > 0:
                print(f"    Min: {arr.min()}")
                print(f"    Max: {arr.max()}")
                if arr.size <= 10:
                    print(f"    Values: {arr}")
                else:
                    print(f"    First 3: {arr[:3]}")
                    print(f"    Last 3: {arr[-3:]}")
        
        # Verify xyz data
        if 'xyz' in data:
            xyz = data['xyz']
            print(f"\nXYZ Statistics:")
            print(f"  Total points: {len(xyz):,}")
            print(f"  X range: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
            print(f"  Y range: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
            print(f"  Z range: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
            
            # Distance from origin
            distances = np.linalg.norm(xyz, axis=1)
            print(f"  Distance from origin:")
            print(f"    Min: {distances.min():.3f} m")
            print(f"    Max: {distances.max():.3f} m")
            print(f"    Mean: {distances.mean():.3f} m")
    
    except Exception as e:
        print(f"[ERROR] Failed to load NPZ file: {e}")
        import traceback
        traceback.print_exc()


def list_output_directory(output_dir: Path):
    """List all files in the output directory."""
    print(f"\nListing output directory: {output_dir}")
    print("=" * 60)
    
    if not output_dir.exists():
        print(f"[ERROR] Directory does not exist: {output_dir}")
        return
    
    # Count files
    pcd_files = sorted(output_dir.glob("lidar/*.pcd"))
    npz_files = sorted(output_dir.glob("lidar_npz/*.npz"))
    camera_files = sorted(output_dir.glob("camera/*.png"))
    
    print(f"Files found:")
    print(f"  PCD files: {len(pcd_files)}")
    print(f"  NPZ files: {len(npz_files)}")
    print(f"  Camera images: {len(camera_files)}")
    
    if pcd_files:
        print(f"\nFirst 5 PCD files:")
        for pcd in pcd_files[:5]:
            size = pcd.stat().st_size
            print(f"  {pcd.name}: {size:,} bytes")
    
    if npz_files:
        print(f"\nFirst 5 NPZ files:")
        for npz in npz_files[:5]:
            size = npz.stat().st_size
            print(f"  {npz.name}: {size:,} bytes")


def main():
    """Main inspection function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect point cloud pipeline results")
    parser.add_argument("--output-dir", type=Path, 
                       default=(Path(__file__).parent.resolve() / "../dataset").resolve(),
                       help="Output directory containing saved files")
    parser.add_argument("--pcd", type=Path, help="Inspect a specific PCD file")
    parser.add_argument("--npz", type=Path, help="Inspect a specific NPZ file")
    parser.add_argument("--list", action="store_true", 
                       help="List all files in output directory")
    
    args = parser.parse_args()
    
    if args.pcd:
        inspect_pcd_file(args.pcd)
    elif args.npz:
        inspect_npz_file(args.npz)
    elif args.list:
        list_output_directory(args.output_dir)
    else:
        # Default: list directory and inspect first file if available
        list_output_directory(args.output_dir)
        
        # Try to inspect first NPZ file if available
        npz_dir = args.output_dir / "lidar_npz"
        if npz_dir.exists():
            npz_files = sorted(npz_dir.glob("*.npz"))
            if npz_files:
                print("\n" + "=" * 60)
                inspect_npz_file(npz_files[0])


if __name__ == "__main__":
    main()

