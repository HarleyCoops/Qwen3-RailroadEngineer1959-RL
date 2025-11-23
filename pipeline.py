import sys
import os
import numpy as np
from PIL import Image
import google.generativeai as genai

# Setup paths to include sam-3d-objects
SAM3D_PATH = os.path.join(os.getcwd(), "sam-3d-objects")
NOTEBOOK_PATH = os.path.join(SAM3D_PATH, "notebook")

sys.path.append(SAM3D_PATH)
sys.path.append(NOTEBOOK_PATH)

try:
    from inference import Inference, load_image
except ImportError:
    print("Error: Could not import 'inference' from sam-3d-objects. Make sure the submodule is cloned and dependencies are installed.")
    sys.exit(1)

def create_dummy_mask(image_shape):
    """
    Creates a dummy center crop mask for testing purposes.
    In a real scenario, this would be replaced by SAM 2 or another segmentation model.
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=bool)
    h, w = image_shape[:2]
    # Create a mask for the center 50% of the image
    h_start, h_end = int(h * 0.25), int(h * 0.75)
    w_start, w_end = int(w * 0.25), int(w * 0.75)
    mask[h_start:h_end, w_start:w_end] = True
    return mask

def generate_railroad_rules(image, reconstruction_output):
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
        # Use the preview model for best multimodal capabilities
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        prompt = (
            "You are an expert Railroad Engineer from 1959. "
            "Analyze this image and the context of a 3D signal reconstruction. "
            "This image depicts a railroad signal or operation. "
            "Based on standard 1959 operating rules, define the 'Indication' and 'Name' of this signal, "
            "and write the corresponding Operating Rule as a conditional statement "
            "(e.g., IF signal is X, THEN proceed at speed Y)."
        )
        
        # Gemini 3 Pro accepts images directly
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
    try:
        # load_image from sam-3d-objects handles PIL/numpy conversion
        image = load_image(image_path)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    print("Generating mask (Using Dummy Mask for now)...")
    # TODO: Replace with SAM 2 segmentation
    mask = create_dummy_mask(image.shape)

    print("Initializing SAM 3D Inference...")
    # Note: This requires the checkpoints to be downloaded to sam-3d-objects/checkpoints/hf/
    tag = "hf"
    config_path = os.path.join(SAM3D_PATH, f"checkpoints/{tag}/pipeline.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}.")
        print("Please ensure model weights are downloaded to 'sam-3d-objects/checkpoints/hf/'")
        # We continue for now to show the pipeline structure, but inference will fail
    
    try:
        inference = Inference(config_path, compile=False)
        
        print("Running Inference...")
        output = inference(image, mask, seed=42)
        
        output_ply = "railroad_part_reconstruction.ply"
        output["gs"].save_ply(output_ply)
        print(f"Reconstruction saved to {output_ply}")
        
        # Rule Generation Step
        rules = generate_railroad_rules(image, output)
        print("\nGenerated Rules:")
        print(rules)

    except Exception as e:
        print(f"\nPipeline Execution Failed (Expected if weights/dependencies are missing): {e}")
        print("Next Steps: Install dependencies from requirements.txt and download model weights.")

if __name__ == "__main__":
    main()

