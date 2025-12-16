"""
Depth Anything V2 Metric Depth Model
Converts input images to metric depth maps and saves as greyscale images.

Usage:
    python depth_anything_metric.py --input path/to/image.png
    python depth_anything_metric.py --input path/to/image.png --output depth_output.png --show

Requirements:
    - PyTorch
    - OpenCV (cv2)
    - NumPy
    - Matplotlib (optional, for --show flag)
    - Depth Anything V2 repository cloned locally OR available via torch.hub
    
Setup:
    1. Clone Depth Anything V2 repository:
       git clone https://github.com/depth-anything/Depth-Anything-V2.git
       
    2. Download metric depth checkpoints to Depth-Anything-V2/checkpoints/
       - depth_anything_v2_metric_hypersim_vitb.pth (indoor)
       - depth_anything_v2_metric_hypersim_vitl.pth (indoor, large)
       - depth_anything_v2_metric_vkitti2_vitb.pth (outdoor)
       - depth_anything_v2_metric_vkitti2_vitl.pth (outdoor, large)
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_depth_anything_metric_model(model_type='vitb', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load Depth Anything V2 metric depth model.
    
    Supports multiple loading methods:
    1. From local Depth-Anything-V2 repository
    2. From torch.hub (if available)
    
    Args:
        model_type: 'vitb' (base) or 'vitl' (large)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model in eval mode, device
    """
    print(f"Loading Depth Anything V2 metric model ({model_type}) on {device}...")
    
    # Method 1: Try loading from local Depth-Anything-V2 repository
    try:
        import sys
        from pathlib import Path
        
        # Check for local installation (parent directory or current directory)
        possible_paths = [
            Path(__file__).parent.parent / 'Depth-Anything-V2',
            Path(__file__).parent / 'Depth-Anything-V2',
            Path('../Depth-Anything-V2'),
            Path('./Depth-Anything-V2'),
        ]
        
        depth_anything_path = None
        for path in possible_paths:
            if path.exists() and (path / 'depth_anything_v2').exists():
                depth_anything_path = path
                break
        
        if depth_anything_path:
            sys.path.insert(0, str(depth_anything_path))
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # Model configurations
            model_configs = {
                'vitb': {
                    'encoder': 'vits',
                    'features': 64,
                    'out_channels': [48, 96, 192, 384]
                },
                'vitl': {
                    'encoder': 'vitl',
                    'features': 128,
                    'out_channels': [96, 192, 384, 768]
                }
            }
            
            model = DepthAnythingV2(**model_configs[model_type])
            
            # Load metric checkpoint
            checkpoint_paths = [
                depth_anything_path / 'checkpoints' / f'depth_anything_v2_metric_hypersim_{model_type}.pth',
                depth_anything_path / 'checkpoints' / f'depth_anything_v2_metric_vkitti2_{model_type}.pth',
            ]
            
            checkpoint_path = None
            for cp in checkpoint_paths:
                if cp.exists():
                    checkpoint_path = cp
                    break
            
            if checkpoint_path:
                print(f"Loading checkpoint from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
                model.to(device)
                model.eval()
                print("Model loaded successfully from local repository!")
                return model, device
            else:
                print(f"Warning: Checkpoint not found. Expected one of: {checkpoint_paths}")
    
    except Exception as e:
        print(f"Failed to load from local repository: {e}")
    
    # Method 2: Try loading from torch.hub
    try:
        print("Attempting to load from torch.hub...")
        model_name = f'depth_anything_v2_metric_hypersim_{model_type}'
        model = torch.hub.load('depth-anything/Depth-Anything-V2', model_name, pretrained=True, trust_repo=True)
        model.to(device)
        model.eval()
        print("Model loaded successfully from torch.hub!")
        return model, device
    except Exception as e:
        print(f"Failed to load from torch.hub: {e}")
    
    # If all methods fail
    raise RuntimeError(
        "Could not load Depth Anything V2 model.\n"
        "Please ensure:\n"
        "1. Depth-Anything-V2 repository is cloned and available, OR\n"
        "2. Model is available via torch.hub\n"
        "3. Checkpoints are downloaded to the checkpoints/ directory"
    )


def preprocess_image(image, target_size=(518, 518)):
    """
    Preprocess image for Depth Anything V2 model.
    
    Args:
        image: Input image (BGR format from cv2)
        target_size: Target size for model input
    
    Returns:
        Preprocessed tensor and original image info
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape[:2]  # (H, W)
    
    # Resize image
    image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] and then to [-1, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - 0.5) / 0.5
    
    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_shape


def predict_depth(model, image_tensor, device, original_shape):
    """
    Predict metric depth from preprocessed image tensor.
    
    Args:
        model: Depth Anything V2 model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        original_shape: Original image shape (H, W)
    
    Returns:
        Depth map as numpy array (in meters)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        depth = model(image_tensor)
    
    # Convert to numpy and resize to original image size
    depth_np = depth.squeeze().cpu().numpy()
    depth_resized = cv2.resize(depth_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return depth_resized


def depth_to_greyscale(depth_map, normalize_method='minmax'):
    """
    Convert metric depth map to greyscale image (0-255).
    
    Args:
        depth_map: Depth map in meters (numpy array)
        normalize_method: 'minmax' or 'percentile' for normalization
    
    Returns:
        Greyscale image (uint8, 0-255)
    """
    # Remove invalid depths (zeros or negative)
    valid_mask = depth_map > 0
    if not np.any(valid_mask):
        print("Warning: No valid depth values found!")
        return np.zeros_like(depth_map, dtype=np.uint8)
    
    if normalize_method == 'minmax':
        # Min-max normalization
        depth_min = depth_map[valid_mask].min()
        depth_max = depth_map[valid_mask].max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    elif normalize_method == 'percentile':
        # Percentile-based normalization (more robust to outliers)
        depth_min = np.percentile(depth_map[valid_mask], 2)
        depth_max = np.percentile(depth_map[valid_mask], 98)
        depth_normalized = np.clip((depth_map - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")
    
    # Convert to 0-255 range
    # Invert so closer objects are brighter (common convention)
    greyscale = (1.0 - depth_normalized) * 255
    greyscale = greyscale.astype(np.uint8)
    
    # Set invalid pixels to black
    greyscale[~valid_mask] = 0
    
    return greyscale


def main():
    parser = argparse.ArgumentParser(description='Run Depth Anything V2 metric depth model and save greyscale image')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save greyscale depth image (default: input_name_depth.png)')
    parser.add_argument('--model', type=str, default='vitb', choices=['vitb', 'vitl'],
                        help='Model type: vitb (base) or vitl (large)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cuda or cpu (default: auto-detect)')
    parser.add_argument('--normalize', type=str, default='minmax', choices=['minmax', 'percentile'],
                        help='Normalization method for greyscale conversion')
    parser.add_argument('--show', action='store_true',
                        help='Display the depth image using matplotlib')
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model, device = load_depth_anything_metric_model(model_type=args.model, device=device)
    
    # Load input image
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    print(f"Loading image: {input_path}")
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    # Preprocess
    print("Preprocessing image...")
    image_tensor, original_shape = preprocess_image(image)
    
    # Predict depth
    print("Running depth estimation...")
    depth_map = predict_depth(model, image_tensor, device, original_shape)
    
    print(f"Depth range: {depth_map[depth_map > 0].min():.3f}m - {depth_map[depth_map > 0].max():.3f}m")
    
    # Convert to greyscale
    print("Converting to greyscale...")
    greyscale = depth_to_greyscale(depth_map, normalize_method=args.normalize)
    
    # Determine output path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_metric_depth.png"
    else:
        output_path = Path(args.output)
    
    # Save greyscale image
    print(f"Saving greyscale depth image to: {output_path}")
    cv2.imwrite(str(output_path), greyscale)
    
    # Optionally display
    if args.show:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(greyscale, cmap='gray')
        plt.title('Metric Depth (Greyscale)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("Done!")


if __name__ == '__main__':
    main()

