import os
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# =========================
# Depth Batch Generation
# =========================
# Expected dataset structure:
# dataset/
#   camera/
#     image_001.png
#     image_002.png
#   lidar/
#     lidar_001.pcd
# Output after running:
# dataset/depth/
#     depth_001.npy
#     depth_002.npy
# =========================

# ---- Paths ----
DATA_ROOT = Path("../dataset")
CAMERA_DIR = DATA_ROOT / "camera"
# CAMERA_DIR = DATA_ROOT / "camera_Timothy"
DEPTH_DIR = DATA_ROOT / "depth"
# DEPTH_DIR = DATA_ROOT / "depth_Timothy"
DEPTH_IMG_DIR = DATA_ROOT / "depthImg"
# DEPTH_IMG_DIR = DATA_ROOT / "depthImg_Timothy"
DEPTH_DIR.mkdir(exist_ok=True)
DEPTH_IMG_DIR.mkdir(exist_ok=True)

# ---- Device Selection ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ---- Model Configs ----
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl'  # change if required

# ---- Load Model ----
print("Loading DepthAnythingV2 model...")
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
print("Model loaded.")

# ---- Camera Intrinsics + Distortion ----
# Need to change K D
# Load intrinsic parameters
K = np.array([491.331107883326, 0.0, 515.3434363622374,
              0.0, 492.14998153326013, 388.93983736974667,
              0.0, 0.0, 1.0]).reshape((3, 3))

# Load distortion coefficients
D = np.array([-0.17351231738659792, 0.041198014750453794, 0.0001161732754962265, 5.6722871890938046e-05, 0.0])

# ---- Core Processing Function ----
def process_all_depths():
    image_files = sorted(CAMERA_DIR.glob("*.png"))

    if not image_files:
        print("No images found in dataset/camera. Exiting.")
        return

    for img_path in tqdm(image_files, desc="Predicting depth", ncols=100):
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            print(f"[WARNING] Failed to read {img_path}")
            continue

        # [PROBLEMS]: The undistort works bad
        # undist = cv2.undistort(raw_img, K, D)
        undist = raw_img
        cropped = undist[:980, :, :]

        depth = model.infer_image(cropped)

        idx = img_path.stem.split("_")[-1]
        depth_out_path = DEPTH_DIR / f"depth_{idx}.npy"
        np.save(depth_out_path, depth)

        depthImg_out_path = str(DEPTH_IMG_DIR / f"depthImg_{idx}.png")
        cv2.imwrite(depthImg_out_path, depth)

        print(f"[SAVED] depth_{idx}.npy")

    print("\nAll depth maps saved to dataset/depth/")


# ---- Main ----
if __name__ == "__main__":
    process_all_depths()
