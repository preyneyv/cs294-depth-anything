#!/usr/bin/env python3
"""
Inspect ROS2 MCAP bag contents - list all topics, message types, and sample data.

Usage:
    python inspect_rosbag.py [--bag path/to/bag.mcap] [--topic TOPIC_NAME] [--sample]

This script helps identify:
- Available image topics (/image, /image_raw, /image/compressed, etc.)
- Camera info topics
- Message types for each topic
- Sample message data (with --sample flag)
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from rosbags.image import message_to_cvimage
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Missing required packages: {e}")
    print("\nTo install dependencies:")
    print("  conda activate your_env")
    print("  pip install rosbags opencv-python numpy")
    sys.exit(1)

typestore = get_typestore(Stores.LATEST)


def list_all_topics(bag_path, verbose=False):
    """List all topics in the bag with their message types."""
    print(f"\n{'='*80}")
    print(f"INSPECTING ROSBAG: {bag_path}")
    print(f"{'='*80}\n")
    
    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"[ERROR] Bag file not found: {bag_path}")
        return None
    
    topics_info = {}
    
    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = reader.connections
            
            print(f"Total topics found: {len(connections)}\n")
            
            # Group by topic category
            image_topics = []
            camera_info_topics = []
            lidar_topics = []
            other_topics = []
            
            for conn in connections:
                topic = conn.topic
                msgtype = conn.msgtype
                is_compressed = 'CompressedImage' in msgtype
                
                info = {
                    'topic': topic,
                    'msgtype': msgtype,
                    'is_compressed': is_compressed,
                    'connection': conn
                }
                
                topics_info[topic] = info
                
                # Categorize topics
                if '/image' in topic.lower():
                    image_topics.append(info)
                elif 'camera_info' in topic.lower():
                    camera_info_topics.append(info)
                elif 'point' in topic.lower() or 'lidar' in topic.lower() or 'cloud' in topic.lower():
                    lidar_topics.append(info)
                else:
                    other_topics.append(info)
            
            # Print categorized topics
            if image_topics:
                print(f"{'─'*80}")
                print("IMAGE TOPICS:")
                print(f"{'─'*80}")
                for info in sorted(image_topics, key=lambda x: x['topic']):
                    raw_marker = " [RAW]" if 'raw' in info['topic'].lower() else ""
                    rect_marker = " [POSSIBLY RECTIFIED]" if '/image' in info['topic'] and 'raw' not in info['topic'].lower() else ""
                    comp_marker = " [COMPRESSED]" if info['is_compressed'] else ""
                    print(f"  {info['topic']}")
                    print(f"    Type: {info['msgtype']}{raw_marker}{rect_marker}{comp_marker}")
                
                print()
            
            if camera_info_topics:
                print(f"{'─'*80}")
                print("CAMERA INFO TOPICS:")
                print(f"{'─'*80}")
                for info in sorted(camera_info_topics, key=lambda x: x['topic']):
                    print(f"  {info['topic']}")
                    print(f"    Type: {info['msgtype']}")
                print()
            
            if lidar_topics:
                print(f"{'─'*80}")
                print("LIDAR/POINT CLOUD TOPICS:")
                print(f"{'─'*80}")
                for info in sorted(lidar_topics, key=lambda x: x['topic']):
                    print(f"  {info['topic']}")
                    print(f"    Type: {info['msgtype']}")
                print()
            
            if other_topics and verbose:
                print(f"{'─'*80}")
                print("OTHER TOPICS:")
                print(f"{'─'*80}")
                for info in sorted(other_topics, key=lambda x: x['topic']):
                    print(f"  {info['topic']}")
                    print(f"    Type: {info['msgtype']}")
                print()
            
            return topics_info, reader
            
    except Exception as e:
        print(f"[ERROR] Failed to read bag: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def check_specific_topic(bag_path, topic_name):
    """Check if a specific topic exists and show details."""
    print(f"\n{'='*80}")
    print(f"CHECKING TOPIC: {topic_name}")
    print(f"{'='*80}\n")
    
    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"[ERROR] Bag file not found: {bag_path}")
        return False
    
    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = [c for c in reader.connections if c.topic == topic_name]
            
            if not connections:
                print(f"[NOT FOUND] Topic '{topic_name}' does not exist in bag.")
                print(f"\nSimilar topics found:")
                similar = [c.topic for c in reader.connections if topic_name.split('/')[-1] in c.topic.lower()]
                for t in similar[:10]:
                    print(f"  - {t}")
                return False
            
            conn = connections[0]
            print(f"[FOUND] Topic exists!")
            print(f"  Message type: {conn.msgtype}")
            print(f"  Is compressed: {'CompressedImage' in conn.msgtype}")
            
            # Count messages
            count = 0
            for _ in reader.messages(connections=connections):
                count += 1
                if count >= 1000:  # Limit to avoid long wait
                    break
            
            print(f"  Message count: {count}{'+' if count >= 1000 else ''}")
            
            # Get first message for analysis
            for conn, timestamp, raw_msg in reader.messages(connections=connections):
                msg = reader.deserialize(raw_msg, conn.msgtype)
                
                print(f"\n  First message timestamp: {timestamp}")
                
                # If it's an image, get dimensions
                if hasattr(msg, 'width') and hasattr(msg, 'height'):
                    print(f"  Image dimensions: {msg.width}x{msg.height}")
                if hasattr(msg, 'encoding'):
                    print(f"  Encoding: {msg.encoding}")
                
                # If it's camera_info, show key properties
                if 'CameraInfo' in conn.msgtype:
                    print(f"  Resolution: {msg.width}x{msg.height}")
                    print(f"  Distortion model: {msg.distortion_model}")
                    K = np.array(msg.k, dtype=float).reshape(3, 3)
                    print(f"  K matrix center (cx, cy): ({K[0,2]:.1f}, {K[1,2]:.1f})")
                
                break
            
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to check topic: {e}")
        import traceback
        traceback.print_exc()
        return False


def sample_topic_messages(bag_path, topic_name, num_samples=3):
    """Sample a few messages from a topic and show details."""
    print(f"\n{'='*80}")
    print(f"SAMPLING MESSAGES FROM: {topic_name}")
    print(f"{'='*80}\n")
    
    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"[ERROR] Bag file not found: {bag_path}")
        return
    
    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = [c for c in reader.connections if c.topic == topic_name]
            
            if not connections:
                print(f"[ERROR] Topic '{topic_name}' not found")
                return
            
            conn = connections[0]
            is_compressed = 'CompressedImage' in conn.msgtype
            is_image = 'Image' in conn.msgtype or 'CompressedImage' in conn.msgtype
            
            print(f"Message type: {conn.msgtype}")
            print(f"Sampling {num_samples} messages...\n")
            
            count = 0
            for conn, timestamp, raw_msg in reader.messages(connections=connections):
                msg = reader.deserialize(raw_msg, conn.msgtype)
                
                print(f"--- Message {count + 1} ---")
                print(f"Timestamp: {timestamp}")
                
                if is_image:
                    if is_compressed:
                        arr = np.frombuffer(msg.data, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    else:
                        img = message_to_cvimage(msg)
                    
                    if img is not None:
                        print(f"Image shape: {img.shape} (H, W, C)")
                        print(f"Image dtype: {img.dtype}")
                        print(f"Pixel range: [{img.min()}, {img.max()}]")
                    
                    if hasattr(msg, 'width') and hasattr(msg, 'height'):
                        print(f"Message width/height: {msg.width}x{msg.height}")
                    if hasattr(msg, 'encoding'):
                        print(f"Encoding: {msg.encoding}")
                
                elif 'CameraInfo' in conn.msgtype:
                    print(f"Resolution: {msg.width}x{msg.height}")
                    print(f"Distortion model: {msg.distortion_model}")
                    K = np.array(msg.k, dtype=float).reshape(3, 3)
                    D = np.array(msg.d, dtype=float)
                    print(f"K matrix:\n{K}")
                    print(f"D coefficients: {D}")
                    print(f"Number of D coefficients: {len(D)}")
                
                count += 1
                if count >= num_samples:
                    break
                    
    except Exception as e:
        print(f"[ERROR] Failed to sample messages: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect ROS2 MCAP bag contents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all topics
  python inspect_rosbag.py
  
  # Check if specific topic exists
  python inspect_rosbag.py --topic /vimba_front/image_raw
  
  # Sample messages from a topic
  python inspect_rosbag.py --topic /vimba_front/image --sample
  
  # Verbose output (show all topics)
  python inspect_rosbag.py --verbose
        """
    )
    
    script_dir = Path(__file__).parent.resolve()
    default_bag = script_dir / "dpt_rosbag_lvms_2024_12/rosbag2_2024_12_12-18_21_55_12.mcap"
    
    parser.add_argument(
        '--bag', '-b',
        type=str,
        default=str(default_bag),
        help=f'Path to MCAP bag file (default: {default_bag})'
    )
    parser.add_argument(
        '--topic', '-t',
        type=str,
        default=None,
        help='Check specific topic (e.g., /vimba_front/image_raw)'
    )
    parser.add_argument(
        '--sample', '-s',
        action='store_true',
        help='Sample messages from the specified topic'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all topics, not just image/camera/lidar'
    )
    
    args = parser.parse_args()
    
    # Check if bag exists
    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"[ERROR] Bag file not found: {bag_path}")
        print(f"\nPlease provide a valid bag path:")
        print(f"  python inspect_rosbag.py --bag /path/to/your/bag.mcap")
        sys.exit(1)
    
    # If specific topic requested
    if args.topic:
        if args.sample:
            sample_topic_messages(bag_path, args.topic)
        else:
            check_specific_topic(bag_path, args.topic)
    else:
        # List all topics
        topics_info, reader = list_all_topics(bag_path, verbose=args.verbose)
        
        if topics_info:
            # Summary
            print(f"{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            
            image_topics = [t for t in topics_info.keys() if '/image' in t.lower()]
            raw_topics = [t for t in image_topics if 'raw' in t.lower()]
            rectified_topics = [t for t in image_topics if 'raw' not in t.lower() and '/image' in t]
            
            print(f"\nImage topics found: {len(image_topics)}")
            if raw_topics:
                print(f"  Raw topics (use these for undistortion):")
                for t in sorted(raw_topics):
                    print(f"    - {t}")
            if rectified_topics:
                print(f"  Possibly rectified topics (may already be undistorted):")
                for t in sorted(rectified_topics):
                    print(f"    - {t}")
            
            camera_info_topics = [t for t in topics_info.keys() if 'camera_info' in t.lower()]
            if camera_info_topics:
                print(f"\nCamera info topics: {len(camera_info_topics)}")
                for t in sorted(camera_info_topics):
                    print(f"    - {t}")
            
            print(f"\n{'='*80}")
            print("RECOMMENDATIONS:")
            print(f"{'='*80}")
            if raw_topics:
                print(f"✅ Use one of these raw topics for undistortion:")
                for t in sorted(raw_topics)[:3]:
                    print(f"   {t}")
            elif rectified_topics:
                print(f"⚠️  No /image_raw topics found!")
                print(f"   Topics like {rectified_topics[0]} may already be rectified.")
                print(f"   Check if undistortion is needed before applying it.")
            else:
                print(f"⚠️  No image topics found matching expected patterns.")


if __name__ == "__main__":
    main()

