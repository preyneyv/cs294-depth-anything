#!/usr/bin/env python3
"""
Check Image Dimensions vs Camera Calibration

Compares actual image dimensions with what the K matrix expects.
Helps identify resolution mismatches that cause undistortion failures.

Usage:
    python check_image_dimensions.py --image image.png --camera-info camera_info.yaml
    python check_image_dimensions.py --image image.png --k "[[fx,0,cx],[0,fy,cy],[0,0,1]]" --resolution 2064x1544
    python check_image_dimensions.py --rosbag bag.mcap --topic /vimba_front/image --camera-info-topic /vimba_front/camera_info
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from rosbags.image import message_to_cvimage
    ROSBAGS_AVAILABLE = True
except ImportError:
    ROSBAGS_AVAILABLE = False


def load_camera_info(yaml_path):
    """Load camera calibration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError(
            "yaml module not available. Install with: pip install pyyaml\n"
            "Or use --rosbag option which doesn't require yaml."
        )
    
    with open(yaml_path) as f:
        calib = yaml.safe_load(f)
    
    K = np.array(calib['camera_matrix']['data']).reshape(3, 3)
    D = np.array(calib['distortion_coefficients']['data'])
    distortion_model = calib['distortion_coefficients']['model']
    width = calib['width']
    height = calib['height']
    
    return K, D, distortion_model, width, height


def analyze_image_dimensions(img_or_path, K, expected_width, expected_height):
    """
    Analyze image dimensions vs K matrix expectations.
    
    Args:
        img_or_path: Either a file path (str/Path) or numpy array (already loaded image)
        K: Camera matrix (3x3)
        expected_width: Expected image width from calibration
        expected_height: Expected image height from calibration
    
    Returns:
        dict with analysis results
    """
    # Handle both file path and already-loaded image
    if isinstance(img_or_path, (str, Path)) or img_or_path is None:
        # It's a file path (or None for rosbag case)
        if img_or_path is None:
            raise ValueError("img_or_path cannot be None. Pass the image array directly when reading from rosbag.")
        img = cv2.imread(str(img_or_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_or_path}")
        image_source = str(img_or_path)
    else:
        # It's already a numpy array
        img = img_or_path
        image_source = "rosbag"
    
    actual_height, actual_width = img.shape[:2]
    
    # K matrix center (cx, cy) should be approximately at image center
    cx = K[0, 2]
    cy = K[1, 2]
    expected_cx = expected_width / 2.0
    expected_cy = expected_height / 2.0
    
    # Calculate expected resolution from K (approximate)
    # For a properly calibrated camera, cx should be ~width/2, cy should be ~height/2
    k_expected_width = cx * 2 if cx > 0 else None
    k_expected_height = cy * 2 if cy > 0 else None
    
    # Check for mismatches
    width_match = actual_width == expected_width
    height_match = actual_height == expected_height
    cx_match = abs(cx - expected_cx) < 10  # Allow 10 pixel tolerance
    cy_match = abs(cy - expected_cy) < 10
    
    # Calculate scale factors if mismatch
    width_scale = actual_width / expected_width if expected_width > 0 else None
    height_scale = actual_height / expected_height if expected_height > 0 else None
    
    return {
        'image_path': image_source,
        'actual_width': actual_width,
        'actual_height': actual_height,
        'expected_width': expected_width,
        'expected_height': expected_height,
        'k_cx': cx,
        'k_cy': cy,
        'expected_cx': expected_cx,
        'expected_cy': expected_cy,
        'k_expected_width': k_expected_width,
        'k_expected_height': k_expected_height,
        'width_match': width_match,
        'height_match': height_match,
        'cx_match': cx_match,
        'cy_match': cy_match,
        'width_scale': width_scale,
        'height_scale': height_scale,
        'has_mismatch': not (width_match and height_match),
        'k_matrix_mismatch': not (cx_match and cy_match),
    }


def check_rosbag_images(bag_path, image_topic, camera_info_topic, num_samples=3):
    """Check dimensions of images from rosbag."""
    if not ROSBAGS_AVAILABLE:
        raise RuntimeError("rosbags not available. Install with: pip install rosbags")
    
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.LATEST)
    
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    
    results = []
    
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        # Get camera info first
        info_conns = [c for c in reader.connections if c.topic == camera_info_topic]
        if not info_conns:
            raise ValueError(f"Camera info topic '{camera_info_topic}' not found")
        
        # Get first camera_info message
        for conn, t, raw in reader.messages(info_conns):
            msg = reader.deserialize(raw, conn.msgtype)
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            D = np.array(msg.d, dtype=float)
            distortion_model = msg.distortion_model
            expected_width = msg.width
            expected_height = msg.height
            break
        
        # Get image messages
        img_conns = [c for c in reader.connections if c.topic == image_topic]
        if not img_conns:
            raise ValueError(f"Image topic '{image_topic}' not found")
        
        conn = img_conns[0]
        is_compressed = 'CompressedImage' in conn.msgtype
        
        count = 0
        for conn, timestamp, raw_msg in reader.messages(img_conns):
            msg = reader.deserialize(raw_msg, conn.msgtype)
            
            # Decode image
            if is_compressed:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                img = message_to_cvimage(msg)
            
            if img is None:
                print(f"[WARN] Could not decode image at timestamp {timestamp}")
                continue
            
            actual_height, actual_width = img.shape[:2]
            
            # Analyze (pass the image array directly, not a file path)
            result = analyze_image_dimensions(
                img,  # Pass the decoded image array
                K,
                expected_width,
                expected_height
            )
            result['timestamp'] = timestamp
            result['message_width'] = msg.width if hasattr(msg, 'width') else None
            result['message_height'] = msg.height if hasattr(msg, 'height') else None
            
            results.append(result)
            
            count += 1
            if count >= num_samples:
                break
    
    return results, K, D, distortion_model, expected_width, expected_height


def print_analysis(results, K, D, distortion_model, expected_width, expected_height):
    """Print formatted analysis results."""
    print(f"\n{'='*80}")
    print("IMAGE DIMENSION ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Camera Calibration:")
    print(f"  Expected resolution: {expected_width}x{expected_height} (WxH)")
    print(f"  Distortion model: {distortion_model}")
    print(f"  K matrix:")
    print(f"    fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"    cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    print(f"  D coefficients: {D}")
    print(f"  Number of D coefficients: {len(D)}")
    
    print(f"\n{'='*80}")
    print("IMAGE COMPARISONS")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- Image {i} ---")
        if 'image_path' in result and result['image_path']:
            print(f"Path: {result['image_path']}")
        if 'timestamp' in result:
            print(f"Timestamp: {result['timestamp']}")
        
        print(f"Actual dimensions: {result['actual_width']}x{result['actual_height']} (WxH)")
        print(f"Expected dimensions: {result['expected_width']}x{result['expected_height']} (WxH)")
        
        if result['message_width']:
            print(f"Message dimensions: {result['message_width']}x{result['message_height']} (WxH)")
        
        # Check matches
        if result['width_match'] and result['height_match']:
            print(f"✅ Resolution MATCHES calibration")
        else:
            print(f"❌ Resolution MISMATCH!")
            if result['width_scale']:
                print(f"   Width scale factor: {result['width_scale']:.4f}")
            if result['height_scale']:
                print(f"   Height scale factor: {result['height_scale']:.4f}")
            print(f"   ⚠️  K matrix needs to be scaled if images were resized!")
        
        # Check K matrix center
        print(f"\nK matrix center check:")
        print(f"  K center (cx, cy): ({result['k_cx']:.2f}, {result['k_cy']:.2f})")
        print(f"  Expected center: ({result['expected_cx']:.2f}, {result['expected_cy']:.2f})")
        
        if result['k_matrix_mismatch']:
            print(f"  ❌ K matrix center doesn't match image center!")
            print(f"     This suggests K was calibrated for different resolution")
        else:
            print(f"  ✅ K matrix center matches image center")
        
        if result['k_expected_width']:
            print(f"  K suggests resolution: ~{result['k_expected_width']:.0f}x{result['k_expected_height']:.0f} (WxH)")
        
        print()
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    mismatches = [r for r in results if r['has_mismatch']]
    if mismatches:
        print(f"❌ FOUND {len(mismatches)} IMAGE(S) WITH RESOLUTION MISMATCH")
        print(f"\nThis will cause undistortion to fail!")
        print(f"\nSolutions:")
        print(f"1. Scale K matrix to match actual image resolution:")
        for r in mismatches:
            if r['width_scale'] and r['height_scale']:
                print(f"   K_scaled = K * diag([{r['width_scale']:.4f}, {r['height_scale']:.4f}, 1])")
        print(f"2. Resize images to match calibration resolution: {expected_width}x{expected_height}")
        print(f"3. Recalibrate camera with actual image resolution")
    else:
        print(f"✅ All images match calibration resolution")
    
    k_mismatches = [r for r in results if r['k_matrix_mismatch']]
    if k_mismatches:
        print(f"\n⚠️  K matrix center doesn't match image center for {len(k_mismatches)} image(s)")
        print(f"   This suggests calibration was done at different resolution")


def main():
    parser = argparse.ArgumentParser(
        description='Check image dimensions vs camera calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single image file
  python check_image_dimensions.py -i image.png -c camera_info.yaml
  
  # Check rosbag images
  python check_image_dimensions.py --rosbag bag.mcap \\
      --image-topic /vimba_front/image \\
      --camera-info-topic /vimba_front/camera_info
  
  # Manual parameters
  python check_image_dimensions.py -i image.png \\
      --k "[[491.33,0,515.34],[0,492.15,388.94],[0,0,1]]" \\
      --resolution 2064x1544
        """
    )
    
    parser.add_argument('--image', '-i', help='Input image file path')
    parser.add_argument('--camera-info', '-c', help='Path to camera_info.yaml')
    parser.add_argument('--k', help='Camera matrix as string')
    parser.add_argument('--resolution', help='Expected resolution as WxH (e.g., 2064x1544)')
    parser.add_argument('--rosbag', '-b', help='Path to rosbag file')
    parser.add_argument('--image-topic', '-t', help='Image topic name (for rosbag)')
    parser.add_argument('--camera-info-topic', help='Camera info topic name (for rosbag)')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples from rosbag')
    
    args = parser.parse_args()
    
    # Determine input source
    if args.rosbag:
        if not args.image_topic or not args.camera_info_topic:
            parser.error("--rosbag requires --image-topic and --camera-info-topic")
        
        print(f"Analyzing rosbag: {args.rosbag}")
        print(f"Image topic: {args.image_topic}")
        print(f"Camera info topic: {args.camera_info_topic}")
        
        results, K, D, distortion_model, expected_width, expected_height = check_rosbag_images(
            args.rosbag,
            args.image_topic,
            args.camera_info_topic,
            num_samples=args.samples
        )
        
    elif args.image:
        if not args.camera_info and not (args.k and args.resolution):
            parser.error("Must provide either --camera-info or both --k and --resolution")
        
        if args.camera_info:
            if not YAML_AVAILABLE:
                parser.error(
                    "yaml module required for --camera-info. Install with: pip install pyyaml\n"
                    "Or use --k and --resolution instead."
                )
            K, D, distortion_model, expected_width, expected_height = load_camera_info(args.camera_info)
        else:
            # Manual parameters
            K = np.array(eval(args.k))
            width_str, height_str = args.resolution.split('x')
            expected_width = int(width_str)
            expected_height = int(height_str)
            distortion_model = 'unknown'
            D = np.array([])
        
        result = analyze_image_dimensions(args.image, K, expected_width, expected_height)
        result['image_path'] = args.image
        results = [result]
        
    else:
        parser.error("Must provide either --image or --rosbag")
    
    # Print analysis
    print_analysis(results, K, D, distortion_model, expected_width, expected_height)


if __name__ == "__main__":
    main()

