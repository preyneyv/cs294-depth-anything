#!/usr/bin/env python3
# Extract images from a ROS2 MCAP without installing ROS.
# Usage examples:
#   python extract_images_from_mcap.py --src . --camera vimba_front
#   python extract_images_from_mcap.py --src . --camera vimba_rear --max-frames 100
#   python extract_images_from_mcap.py --src rosbag_dir --camera vimba_front

from pathlib import Path
import argparse
import os
import sys
import cv2
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

def decode_compressed_image(msg) -> np.ndarray:
    """Decode sensor_msgs/msg/CompressedImage to BGR numpy image."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode CompressedImage payload.")
    return img

def pick_image_topic(conns, base_topic: str):
    """
    Prefer raw image if present (/<base>/image). Otherwise use compressed (/<base>/image/compressed).
    Returns (topic_name, connection, is_compressed).
    """
    raw = f'/{base_topic}/image'
    cmp = f'/{base_topic}/image/compressed'

    if raw in conns:
        return raw, conns[raw], False
    if cmp in conns:
        return cmp, conns[cmp], True
    return None, None, None

def main():
    

    
    ap = argparse.ArgumentParser(description="Extract images from ROS2 MCAP (.mcap) without ROS.")
    ap.add_argument("--src", default=".", help="Path to directory containing .mcap + metadata.yaml OR to the .mcap file")
    ap.add_argument("--camera", choices=["vimba_front", "vimba_rear"], default="vimba_front",
                    help="Which camera stream to extract")
    ap.add_argument("--outdir", default="export", help="Output root directory")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0 = all)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[ERROR] Source path not found: {src}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) / args.camera
    imgdir = outdir / "images"
    imgdir.mkdir(parents=True, exist_ok=True)

    # Open MCAP (folder or file)
    with AnyReader([src]) as reader:
        conns = {c.topic: c for c in reader.connections}

        # --- Camera info extraction (save YAML) ---
        topic_info = f'/{args.camera}/camera_info'
        if topic_info in conns:
            from yaml import safe_dump
            for conn, t, raw in reader.messages(conns[topic_info]):
                msg = reader.deserialize(raw, conn.msgtype)
                K = np.array(msg.k, dtype=float).reshape(3, 3)
                D = np.array(msg.d, dtype=float).ravel().tolist()
                R = np.array(msg.r, dtype=float).reshape(3, 3)
                P = np.array(msg.p, dtype=float).reshape(3, 4)
                calib = {
                    'width': msg.width,
                    'height': msg.height,
                    'camera_matrix': {'data': K.flatten().tolist(), 'rows': 3, 'cols': 3},
                    'distortion_coefficients': {'data': D, 'model': msg.distortion_model},
                    'rectification_matrix': {'data': R.flatten().tolist(), 'rows': 3, 'cols': 3},
                    'projection_matrix': {'data': P.flatten().tolist(), 'rows': 3, 'cols': 4},
                }
                with open(outdir / 'camera_info.yaml', 'w') as f:
                    safe_dump(calib, f)
                print(f"[INFO] Wrote camera_info.yaml to {outdir}")
                break
        else:
            print(f"[WARN] No camera_info topic found for {args.camera}")

        topic, conn, is_compressed = pick_image_topic(conns, args.camera)
        if topic is None:
            # Help the user by listing close matches
            candidates = [t for t in conns.keys() if args.camera in t and ('/image' in t)]
            print(f"[ERROR] No image topic found for '{args.camera}'. Looked for '/{args.camera}/image' and '/{args.camera}/image/compressed'.",
                  file=sys.stderr)
            if candidates:
                print("        Did you mean one of:", file=sys.stderr)
                for t in candidates[:10]:
                    print(f"          {t}", file=sys.stderr)
            else:
                print("        No similar image topics found. Run a topic dump to inspect available topics.", file=sys.stderr)
            sys.exit(2)

        print(f"[INFO] Using topic: {topic}  ({'CompressedImage' if is_compressed else 'Image'})")
        count = 0

        # Iterate messages on the chosen topic
        for _, t, raw in reader.messages(conn):
            msg = reader.deserialize(raw, conn.msgtype)
            if is_compressed:
                img = decode_compressed_image(msg)
            else:
                img = message_to_cvimage(msg)  # returns numpy image in BGR

            ts_ns = int(t)  # nanoseconds timestamp from bag
            outpath = imgdir / f"{ts_ns}.png"

            ok = cv2.imwrite(str(outpath), img)
            if not ok:
                print(f"[WARN] Failed to write image: {outpath}", file=sys.stderr)

            count += 1
            if args.max_frames and count >= args.max_frames:
                break
            if count % 100 == 0:
                print(f"[INFO] Wrote {count} frames...")

        print(f"[DONE] Wrote {count} frames to: {imgdir}")

if __name__ == "__main__":
    main()
