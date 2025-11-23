import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add sam-3d-objects paths
SAM3D_ROOT = os.path.join(os.getcwd(), "sam-3d-objects")
sys.path.append(SAM3D_ROOT)
sys.path.append(os.path.join(SAM3D_ROOT, "notebook"))

# Import SAM 3D specific modules
# We wrap these in try-except to allow the script to parse even if dependencies aren't installed yet (for checking code structure)
try:
    from inference import Inference, load_image
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    print(f"Warning: Dependencies missing ({e}). Ensure requirements are installed.")
    # Mocks for syntax checking
    Inference = object
    load_image = lambda x: np.zeros((100, 100, 3), dtype=np.uint8)

def setup_sam2(model_cfg="sam2_hiera_l.yaml", checkpoint="sam2_hiera_large.pt"):
    """
    Initialize SAM 2 Automatic Mask Generator.
    This requires the SAM 2 config and checkpoint to be present.
    """
    print("Initializing SAM 2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Note: In a real deployment, ensure these paths are correct relative to the runtime root
    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    
    # Configure the mask generator to be sensitive enough to pick up the figures
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Filter out tiny noise
    )
    return mask_generator

def setup_sam3d(config_path=None):
    """
    Initialize SAM 3D Objects Inference Pipeline.
    """
    print("Initializing SAM 3D...")
    if config_path is None:
        # Default path structure in sam-3d-objects repo
        config_path = os.path.join(SAM3D_ROOT, "checkpoints/hf/pipeline.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SAM 3D Config not found at {config_path}. Download weights to checkpoints/hf/")
        
    inference = Inference(config_path, compile=False)
    return inference

def filter_masks(masks, image_shape):
    """
    Filter the generated masks to find likely candidates for 'men'.
    Heuristics:
    1. Area size (not too small, not too big)
    2. Aspect ratio (men are usually taller than wide)
    """
    valid_masks = []
    h, w = image_shape[:2]
    total_area = h * w
    
    for i, mask_data in enumerate(masks):
        segmentation = mask_data['segmentation']
        area = mask_data['area']
        bbox = mask_data['bbox'] # [x, y, w, h]
        
        # Heuristic 1: Area should be significant but not the whole page
        # e.g., between 1% and 20% of the page
        if not (0.01 * total_area < area < 0.2 * total_area):
            continue
            
        # Heuristic 2: Aspect Ratio (Height / Width)
        # Men standing are likely > 1.0 aspect ratio (taller than wide)
        # Allow some buffer for wide signals
        box_w, box_h = bbox[2], bbox[3]
        aspect_ratio = box_h / box_w
        if aspect_ratio < 0.8: # Reject very wide, flat objects (like text lines sometimes)
            continue
            
        valid_masks.append(mask_data)
        
    print(f"Filtered {len(masks)} masks down to {len(valid_masks)} potential figures.")
    return valid_masks

def process_railroad_manual(image_path, output_dir="output_3d"):
    """
    Main pipeline:
    1. Load Image
    2. Segment using SAM 2
    3. Reconstruct using SAM 3D
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading image: {image_path}")
    image = load_image(image_path) # Returns numpy array
    
    # 1. Segmentation with SAM 2
    try:
        # Checkpoint paths need to be managed in deployment
        mask_generator = setup_sam2() 
        masks = mask_generator.generate(image)
    except Exception as e:
        print(f"SAM 2 execution failed: {e}")
        print("Falling back to dummy masks for demonstration...")
        # Create dummy data for script validation if model fails/missing
        masks = [{'segmentation': np.zeros(image.shape[:2], dtype=bool), 'area': 1000, 'bbox': [0,0,100,200]}]
    
    # 2. Filter Masks to find the men
    men_masks = filter_masks(masks, image.shape)
    
    if not men_masks:
        print("No suitable figures found in image.")
        return

    # 3. 3D Reconstruction with SAM 3D
    try:
        inference = setup_sam3d()
    except Exception as e:
        print(f"SAM 3D Setup failed: {e}")
        return

    for i, mask_data in enumerate(men_masks):
        print(f"Processing Figure {i+1}/{len(men_masks)}...")
        
        # Extract boolean mask
        mask = mask_data['segmentation']
        
        # Run Inference
        # SAM 3D expects: image (H,W,3), mask (H,W) boolean/uint8
        try:
            output = inference(image, mask, seed=42)
            
            # Save Result
            filename = f"figure_{i+1}.ply"
            save_path = os.path.join(output_dir, filename)
            output["gs"].save_ply(save_path)
            print(f"Saved 3D model to {save_path}")
            
            # Optional: Save a debug image showing what was segmented
            # debug_viz = image.copy()
            # debug_viz[mask] = debug_viz[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            # Image.fromarray(debug_viz).save(os.path.join(output_dir, f"figure_{i+1}_debug.png"))
            
        except Exception as e:
            print(f"Failed to reconstruct figure {i+1}: {e}")

if __name__ == "__main__":
    # Expecting TestImage.jpg in current directory
    process_railroad_manual("TestImage.jpg")

