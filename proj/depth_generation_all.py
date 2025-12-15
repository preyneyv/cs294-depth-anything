"""
=========================
Multi-Model Depth Batch Generation
=========================
Supports: depth_anything_v2, midas, marigold, depth_pro, sgbm

Expected dataset structure:
dataset/
  camera/
    image_001.png
    image_002.png
  camera_right/          # Only needed for SGBM stereo
    image_001.png
    image_002.png

Output after running:
dataset/depth_{MODEL}/
    depth_001.npy
dataset/depthImg_{MODEL}/
    depthImg_001.png
=========================

INSTALLATION INSTRUCTIONS:
--------------------------

# Base dependencies (all models)
pip install numpy opencv-python torch torchvision tqdm pillow matplotlib

# 1. Depth Anything V2
#    Clone repo and download weights:
#    git clone https://github.com/DepthAnything/Depth-Anything-V2
#    Download checkpoints from: https://huggingface.co/depth-anything/Depth-Anything-V2-Large
#    Place in: checkpoints/depth_anything_v2_vitl.pth

# 2. MiDaS (Intel) - Uses torch.hub, no additional install needed
pip install timm

# 3. Marigold (Diffusion-based)
pip install diffusers transformers accelerate

# 4. Depth Pro (Apple)
#    git clone https://github.com/apple/ml-depth-pro
#    cd ml-depth-pro && pip install -e .
#    bash get_pretrained_models.sh  # Downloads to ./checkpoints/depth_pro.pt

# 5. SGBM (OpenCV stereo) - No additional install, uses cv2.StereoSGBM
#    Requires stereo image pairs (left/right cameras)

"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# =========================
# Configuration
# =========================
class Config:
    DATA_ROOT = Path("../dataset")
    CAMERA_DIR = DATA_ROOT / "camera"
    CAMERA_RIGHT_DIR = DATA_ROOT / "camera_right"  # For stereo SGBM
    
    # Camera Intrinsics (modify for your camera)
    K = np.array([491.331107883326, 0.0, 515.3434363622374,
                  0.0, 492.14998153326013, 388.93983736974667,
                  0.0, 0.0, 1.0]).reshape((3, 3))
    D = np.array([-0.17351231738659792, 0.041198014750453794, 
                  0.0001161732754962265, 5.6722871890938046e-05, 0.0])
    
    # Stereo baseline (meters) - for SGBM metric depth
    BASELINE = 0.12  # Adjust for your stereo setup
    FOCAL_LENGTH_PX = 491.33  # fx from K matrix


# =========================
# Device Selection
# =========================
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# =========================
# Model Loaders
# =========================

def load_depth_anything_v2(encoder='vitl', device='cuda'):
    """Load Depth Anything V2 model"""
    from depth_anything_v2.dpt import DepthAnythingV2
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    return model, None


def load_midas(model_type='DPT_Large', device='cuda'):
    """
    Load MiDaS model via torch.hub
    
    model_type options:
    - 'DPT_Large': MiDaS v3 Large (highest accuracy, slowest)
    - 'DPT_Hybrid': MiDaS v3 Hybrid (medium accuracy/speed)
    - 'MiDaS_small': MiDaS v2.1 Small (fastest, lowest accuracy)
    """
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model = model.to(device).eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    return model, transform


def load_marigold(device='cuda'):
    """
    Load Marigold model via diffusers
    Uses LCM variant for faster inference (1-4 steps vs 50)
    """
    from diffusers import MarigoldDepthPipeline
    
    # Use LCM variant for speed, or 'prs-eth/marigold-depth-v1-1' for quality
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0",
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        variant="fp16" if device == 'cuda' else None
    )
    pipe = pipe.to(device)
    return pipe, None


def load_depth_pro(device='cuda'):
    """Load Apple Depth Pro model"""
    import depth_pro
    from depth_pro.depth_pro import DepthProConfig
    DEFAULT_MONODEPTH_CONFIG_DICT = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri="ml-depth-pro/checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    
    model, transform = depth_pro.create_model_and_transforms(config=DEFAULT_MONODEPTH_CONFIG_DICT)
    model = model.to(device).eval()
    return model, transform


def load_sgbm():
    """
    Configure OpenCV StereoSGBM for stereo depth estimation
    Returns configured stereo matcher
    """
    # SGBM Parameters - tune these for your setup
    min_disparity = 0
    num_disparities = 128  # Must be divisible by 16
    block_size = 5  # Odd number, typically 3-11
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # WLS filter for post-processing (optional but recommended)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    stereo_right = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)
    
    return stereo, (stereo_right, wls_filter)


# =========================
# Inference Functions
# =========================

def infer_depth_anything_v2(model, img_bgr, device='cuda'):
    """Inference with Depth Anything V2"""
    depth = model.infer_image(img_bgr)
    return depth


def infer_midas(model, transform, img_bgr, device='cuda'):
    """Inference with MiDaS"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    # MiDaS outputs inverse depth, convert if needed
    # depth = 1.0 / (depth + 1e-6)  # Uncomment for regular depth
    return depth


def infer_marigold(pipe, img_bgr, device='cuda'):
    """Inference with Marigold"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Inference
    output = pipe(
        img_pil,
        num_inference_steps=4,      # LCM: 1-4 steps, Original: 10-50 steps
        ensemble_size=1,            # Increase for better quality (slower)
        processing_resolution=768,
        match_input_resolution=True,
    )
    
    depth = output.prediction.squeeze()
    return depth


def infer_depth_pro(model, transform, img_path, device='cuda'):
    """Inference with Apple Depth Pro"""
    import depth_pro
    
    # Load image using depth_pro's loader
    image, _, f_px = depth_pro.load_rgb(img_path)
    image = image[:980]
    image = transform(image).to(device)
    
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
    
    depth = prediction["depth"].cpu().numpy()  # Metric depth in meters
    focal_px = prediction["focallength_px"]
    
    return depth


def infer_sgbm(stereo, extras, img_left, img_right):
    """
    Inference with OpenCV SGBM stereo matching
    Returns disparity map (convert to depth using: depth = baseline * focal / disparity)
    """
    stereo_right, wls_filter = extras
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity
    disp_left = stereo.compute(gray_left, gray_right)
    disp_right = stereo_right.compute(gray_right, gray_left)
    
    # Apply WLS filter
    disparity = wls_filter.filter(disp_left, gray_left, None, disp_right)
    
    # Convert to float and scale (SGBM returns fixed-point)
    disparity = disparity.astype(np.float32) / 16.0
    
    # Convert disparity to metric depth
    # depth = baseline * focal_length / disparity
    valid_mask = disparity > 0
    depth = np.zeros_like(disparity)
    depth[valid_mask] = (Config.BASELINE * Config.FOCAL_LENGTH_PX) / disparity[valid_mask]
    
    return depth, disparity


# =========================
# Main Processing Function
# =========================

def process_all_depths(model_name='depth_anything_v2', encoder='vitl'):
    """Process all images with selected model"""
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    
    # Setup output directories
    depth_dir = Config.DATA_ROOT / f"depth_{model_name}"
    depth_img_dir = Config.DATA_ROOT / f"depthImg_{model_name}"
    depth_dir.mkdir(exist_ok=True)
    depth_img_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading {model_name} model...")
    
    if model_name == 'depth_anything_v2':
        model, transform = load_depth_anything_v2(encoder, device)
    elif model_name == 'midas':
        model, transform = load_midas('DPT_Large', device)
    elif model_name == 'midas_small':
        model, transform = load_midas('MiDaS_small', device)
    elif model_name == 'marigold':
        model, transform = load_marigold(device)
    elif model_name == 'depth_pro':
        model, transform = load_depth_pro(device)
    elif model_name == 'sgbm':
        model, transform = load_sgbm()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print("Model loaded.")
    
    # Get image files
    image_files = sorted(Config.CAMERA_DIR.glob("*.png"))
    if not image_files:
        image_files = sorted(Config.CAMERA_DIR.glob("*.jpg"))
    
    if not image_files:
        print(f"No images found in {Config.CAMERA_DIR}. Exiting.")
        return
    
    # Process images
    for img_path in tqdm(image_files, desc=f"Predicting depth ({model_name})", ncols=100):
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            print(f"[WARNING] Failed to read {img_path}")
            continue
        
        # Optional: undistort
        # undist = cv2.undistort(raw_img, Config.K, Config.D)
        undist = raw_img
        undist = undist[:980, :, :]
        
        # Run inference based on model
        if model_name == 'depth_anything_v2':
            depth = infer_depth_anything_v2(model, undist, device)
        
        elif model_name in ['midas', 'midas_small']:
            depth = infer_midas(model, transform, undist, device)
        
        elif model_name == 'marigold':
            depth = infer_marigold(model, undist, device)
        
        elif model_name == 'depth_pro':
            depth = infer_depth_pro(model, transform, str(img_path), device)
        
        elif model_name == 'sgbm':
            # Load right image for stereo
            right_path = Config.CAMERA_RIGHT_DIR / img_path.name
            if not right_path.exists():
                print(f"[WARNING] Right image not found: {right_path}")
                continue
            right_img = cv2.imread(str(right_path))
            depth, disparity = infer_sgbm(model, transform, undist, right_img)
        
        # Save outputs
        idx = img_path.stem.split("_")[-1]
        
        # Save raw depth as numpy
        depth_out_path = depth_dir / f"depth_{idx}.npy"
        np.save(depth_out_path, depth)
        
        # Normalize for visualization
        depth_vis = depth.copy()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        depth_img_path = depth_img_dir / f"depthImg_{idx}.png"
        cv2.imwrite(str(depth_img_path), depth_colored)
    
    print(f"\nAll depth maps saved to {depth_dir}")
    print(f"Visualizations saved to {depth_img_dir}")


# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model depth estimation")
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='depth_anything_v2',
        choices=['depth_anything_v2', 'midas', 'midas_small', 'marigold', 'depth_pro', 'sgbm'],
        help='Depth estimation model to use'
    )
    parser.add_argument(
        '--encoder', '-e',
        type=str,
        default='vitl',
        choices=['vits', 'vitb', 'vitl', 'vitg'],
        help='Encoder for Depth Anything V2 (vits/vitb/vitl/vitg)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='../dataset',
        help='Root directory of dataset'
    )
    
    args = parser.parse_args()
    
    # Update config
    Config.DATA_ROOT = Path(args.data_root)
    Config.CAMERA_DIR = Config.DATA_ROOT / "camera"
    Config.CAMERA_RIGHT_DIR = Config.DATA_ROOT / "camera_right"
    
    process_all_depths(model_name=args.model, encoder=args.encoder)
