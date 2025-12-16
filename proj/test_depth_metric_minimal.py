"""
Minimal script to test Depth Anything V2 metric depth model.
Outputs raw depth values with minimal postprocessing for bug testing.
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys


def load_model(model_type='vitb', device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_path=None):
    """Load Depth Anything V2 metric model."""
    print(f"Loading model ({model_type}) on {device}...")
    
    # Try local repository first
    possible_paths = [
        Path(__file__).parent.parent / 'Depth-Anything-V2',
        Path(__file__).parent / 'Depth-Anything-V2',
        Path('../Depth-Anything-V2'),
    ]
    
    # Try to find and import the metric depth model
    DepthAnythingV2 = None
    metric_depth_path = None
    
    for path in possible_paths:
        # Check for metric_depth version first (has max_depth parameter)
        if path.exists() and (path / 'metric_depth' / 'depth_anything_v2').exists():
            sys.path.insert(0, str(path / 'metric_depth'))
            from depth_anything_v2.dpt import DepthAnythingV2
            metric_depth_path = path
            break
        elif path.exists() and (path / 'depth_anything_v2').exists():
            sys.path.insert(0, str(path))
            from depth_anything_v2.dpt import DepthAnythingV2
            metric_depth_path = path
            break
    
    if DepthAnythingV2 is None:
        raise RuntimeError("Could not find Depth-Anything-V2 repository")
    
    # Model configs from official example
    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    # Map model_type to encoder type (will be overridden if checkpoint detected)
    encoder_type = model_type
    if encoder_type not in configs:
        encoder_type = 'vits'
    
    # Determine dataset and max_depth from checkpoint filename
    dataset = 'vkitti'  # default to outdoor
    max_depth = 80  # default to outdoor (80m)
    
    # Check checkpoint paths to determine dataset
    checkpoint_paths_to_try = []
    
    # If user provided checkpoint path, use it
    if checkpoint_path:
        checkpoint_paths_to_try.append(Path(checkpoint_path))
    
    # Also check standard locations
    if metric_depth_path:
        for cp_name in [f'depth_anything_v2_metric_hypersim_{model_type}.pth',
                       f'depth_anything_v2_metric_vkitti2_{model_type}.pth',
                       f'depth_anything_v2_metric_vkitti_{model_type}.pth',
                       f'depth_anything_v2_metric_vkitti_vits.pth']:
            checkpoint_paths_to_try.append(metric_depth_path / 'checkpoints' / cp_name)
    
    # Also check root directory for checkpoint
    root_dir = Path(__file__).parent.parent
    for cp_name in ['depth_anything_v2_metric_vkitti_vits.pth',
                   f'depth_anything_v2_metric_vkitti2_{model_type}.pth',
                   f'depth_anything_v2_metric_vkitti_{model_type}.pth']:
        checkpoint_paths_to_try.append(root_dir / cp_name)
    
    # Find checkpoint and determine dataset and model type
    found_checkpoint = None
    detected_model_type = encoder_type  # default to requested type
    for cp_path in checkpoint_paths_to_try:
        if cp_path.exists():
            found_checkpoint = cp_path
            cp_str = str(cp_path)
            # Determine dataset from filename
            if 'hypersim' in cp_str:
                dataset = 'hypersim'
                max_depth = 20  # indoor model
            elif 'vkitti' in cp_str:
                dataset = 'vkitti'
                max_depth = 80  # outdoor model
            
            # Auto-detect model type from checkpoint filename
            if '_vits.pth' in cp_str or '_vits.' in cp_str:
                detected_model_type = 'vits'
            elif '_vitb.pth' in cp_str or '_vitb.' in cp_str:
                detected_model_type = 'vitb'
            elif '_vitl.pth' in cp_str or '_vitl.' in cp_str:
                detected_model_type = 'vitl'
            break
    
    if not found_checkpoint:
        raise RuntimeError("Could not find checkpoint file. Please ensure checkpoint is downloaded.")
    
    # Use detected model type from checkpoint if available
    if detected_model_type != encoder_type:
        print(f"Detected model type '{detected_model_type}' from checkpoint (requested: '{encoder_type}'), using detected type")
        encoder_type = detected_model_type
        if encoder_type not in configs:
            raise RuntimeError(f"Unknown model type detected: {encoder_type}")
    
    # Create model with max_depth (only metric_depth version supports this)
    try:
        model = DepthAnythingV2(**{**configs[encoder_type], 'max_depth': max_depth})
        print(f"Using max_depth={max_depth} for {dataset} dataset with {encoder_type} encoder")
    except TypeError:
        # Fallback if max_depth not supported (shouldn't happen with metric_depth version)
        model = DepthAnythingV2(**configs[encoder_type])
        print(f"Warning: max_depth parameter not supported, using default")
    
    # Load checkpoint
    print(f"Loading checkpoint: {found_checkpoint}")
    checkpoint = torch.load(found_checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device
    
    # Try torch.hub
    try:
        model = torch.hub.load('depth-anything/Depth-Anything-V2', 
                              f'depth_anything_v2_metric_hypersim_{model_type}',
                              pretrained=True, trust_repo=True)
        model.to(device)
        model.eval()
        return model, device
    except:
        raise RuntimeError("Could not load model. Check Depth-Anything-V2 installation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default=None, help='Output depth image path')
    parser.add_argument('--model', default='vitb', choices=['vitb', 'vitl', 'vits'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--checkpoint', '-c', default=None, help='Path to .pth checkpoint file')
    parser.add_argument('--csv', action='store_true', help='Export depth values to CSV')
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, device = load_model(args.model, device, checkpoint_path=args.checkpoint)
    
    # Load image
    img_path = Path(args.input)
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    original_shape = image.shape[:2]  # (H, W)
    
    # Minimal preprocessing: resize and normalize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (518, 518), interpolation=cv2.INTER_LINEAR)
    image_tensor = (image_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        depth = model(image_tensor)
    
    # Get depth as numpy array
    depth_np = depth.squeeze().cpu().numpy()
    depth_resized = cv2.resize(depth_np, (original_shape[1], original_shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
    
    # Print statistics
    valid_mask = depth_resized > 0
    if np.any(valid_mask):
        print(f"\nDepth Statistics:")
        print(f"  Min: {depth_resized[valid_mask].min():.3f}m")
        print(f"  Max: {depth_resized[valid_mask].max():.3f}m")
        print(f"  Mean: {depth_resized[valid_mask].mean():.3f}m")
        print(f"  Median: {np.median(depth_resized[valid_mask]):.3f}m")
        print(f"  Shape: {depth_resized.shape}")
    else:
        print("Warning: No valid depth values!")
    
    # Minimal normalization for visualization (no inversion, just scale to 0-255)
    depth_min = depth_resized[valid_mask].min() if np.any(valid_mask) else 0
    depth_max = depth_resized[valid_mask].max() if np.any(valid_mask) else 1
    depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)
    depth_greyscale = (depth_normalized * 255).astype(np.uint8)
    depth_greyscale[~valid_mask] = 0
    
    # Save output
    output_path = args.output or img_path.parent / f"{img_path.stem}_metric_depth_raw.png"
    cv2.imwrite(str(output_path), depth_greyscale)
    print(f"\nSaved greyscale depth image to: {output_path}")
    print("Note: Dark = close, Bright = far (no inversion applied)")
    
    # Also save raw depth values as numpy array for inspection
    raw_output = output_path.parent / f"{img_path.stem}_metric_depth_raw.npy"
    np.save(str(raw_output), depth_resized)
    print(f"Saved raw depth values (meters) to: {raw_output}")
    
    # Optionally save as CSV with pixel coordinates
    if args.csv:
        csv_output = output_path.parent / f"{img_path.stem}_metric_depth_raw.csv"
        h, w = depth_resized.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Create CSV with columns: row, col, depth_meters
        csv_data = np.column_stack([
            y_coords.flatten(),
            x_coords.flatten(),
            depth_resized.flatten()
        ])
        
        # Save as CSV
        np.savetxt(str(csv_output), csv_data, delimiter=',', 
                  header='row,col,depth_meters', comments='', fmt='%d,%d,%.6f')
        print(f"Saved depth values to CSV: {csv_output}")


if __name__ == '__main__':
    main()

