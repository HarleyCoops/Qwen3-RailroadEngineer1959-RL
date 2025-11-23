import sys
import os
import numpy as np
from PIL import Image
import torch
import google.generativeai as genai

# Setup paths
SAM3D_PATH = os.path.join(os.getcwd(), "sam-3d-objects")
NOTEBOOK_PATH = os.path.join(SAM3D_PATH, "notebook")

sys.path.append(SAM3D_PATH)
sys.path.append(NOTEBOOK_PATH)

# Import SAM 3D
try:
    from inference import Inference, load_image
except ImportError:
    print("Error: Could not import 'inference' from sam-3d-objects.")
    sys.exit(1)

def create_dummy_mask(image_shape):
    """
    Creates a dummy center crop mask for testing purposes.
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=bool)
    h, w = image_shape[:2]
    # Create a mask for the center 50% of the image
    h_start, h_end = int(h * 0.25), int(h * 0.75)
    w_start, w_end = int(w * 0.25), int(w * 0.75)
    mask[h_start:h_end, w_start:w_end] = True
    return mask

def generate_railroad_rules(image, reconstruction_ply_path):
    """
    Uses Gemini 3 Pro to generate railroad rules based on the visual analysis.
    """
    print("Generating railroad rules using Gemini 3 Pro...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. Skipping Gemini generation.")
        return "Rule Generation Skipped: No API Key"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        prompt = (
            "You are an expert Railroad Engineer from 1959. "
            "Analyze this image and the 3D reconstruction context. "
            "This image depicts a railroad signal or operation. "
            "Based on standard 1959 operating rules, define the 'Indication' and 'Name' of this signal, "
            "and write the corresponding Operating Rule as a conditional statement "
            "(e.g., IF signal is X, THEN proceed at speed Y)."
        )
        
        # Pass image to Gemini
        response = model.generate_content([prompt, image])
        return response.text
        
    except Exception as e:
        return f"Gemini Generation Failed: {e}"

def main():
    image_path = "TestImage.jpg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Loading image: {image_path}")
    sam3d_image = load_image(image_path) # Load for SAM 3D (usually [0,1] float)
    
    # Step 1: Mask Generation (Using Dummy Mask as we only use SAM 3D which needs a mask input)
    # In a real SAM 3D workflow without SAM 2, the mask might come from a different source or user input.
    # We use a dummy mask here to satisfy the SAM 3D input requirement.
    print("Step 1: Generating Mask...")
    mask = create_dummy_mask(sam3d_image.shape)
    
    # Step 2: SAM 3D Reconstruction
    print("Step 2: Running SAM 3D Reconstruction...")
    tag = "hf"
    config_path = os.path.join(SAM3D_PATH, f"checkpoints/{tag}/pipeline.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: SAM 3D Config not found at {config_path}. Skipping 3D step.")
        return

    try:
        inference = Inference(config_path, compile=False)
        output = inference(sam3d_image, mask, seed=42)
        
        output_ply = "railroad_part_reconstruction.ply"
        output["gs"].save_ply(output_ply)
        print(f"Reconstruction saved to {output_ply}")
        
        # Step 3: Gemini 3 Pro Rule Generation
        print("Step 3: Gemini 3 Pro Rule Generation...")
        # Reload image as PIL for Gemini
        pil_image = Image.open(image_path)
        rules = generate_railroad_rules(pil_image, output_ply)
        print("\nGenerated Rules:")
        print(rules)

    except Exception as e:
        print(f"\nPipeline Execution Failed: {e}")

if __name__ == "__main__":
    main()
