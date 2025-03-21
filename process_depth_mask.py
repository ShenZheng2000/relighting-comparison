import os
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor  # Ensure this module is correctly imported

# Load FluxControlNet depth preprocessor
depth_processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")

# Supported image extensions
valid_extensions = (".png", ".jpg", ".jpeg", ".webp")


def process_depth_map(input_path, output_path):
    """
    Process an image to generate its depth map.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the generated depth map.
    """
    control_image = load_image(input_path)
    depth_map = depth_processor(control_image)[0].convert("RGB")
    depth_map.save(output_path)
    print(f"Saved depth map to {output_path}")


def process_body_mask(input_path, white_mask_output_path):
    """
    Convert an RGBA image to both white and black foreground masks.
    The white foreground mask has the foreground in white and background in black.
    The black foreground mask has the foreground in black and background in white.
    
    Args:
        input_path (str): Path to the input RGBA body mask.
        white_mask_output_path (str): Path to save the white foreground mask.
    
    Returns:
        tuple: (white_mask_path, black_mask_path)
    """
    img = Image.open(input_path).convert("RGBA")
    img = np.array(img)

    # Extract the alpha channel (mask)
    alpha_channel = img[:, :, 3]

    # Create the white foreground mask (foreground in white, background in black)
    white_fg_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
    # Create the black foreground mask (foreground in black, background in white)
    black_fg_mask = np.where(alpha_channel > 0, 0, 255).astype(np.uint8)

    # Save the masks
    Image.fromarray(white_fg_mask).save(white_mask_output_path)
    black_mask_output_path = white_mask_output_path.replace("white_fg_mask.png", "black_fg_mask.png")
    Image.fromarray(black_fg_mask).save(black_mask_output_path)
    print(f"Saved masks: {white_mask_output_path}, {black_mask_output_path}")
    return white_mask_output_path, black_mask_output_path



def process_folder(folder_path):
    """
    Process depth and body mask images in a given folder.

    Args:
        folder_path (str): Path to a subfolder containing images and pre_processing data.
    """
    pre_processing_path = os.path.join(folder_path, "pre_processing")
    if not os.path.exists(pre_processing_path):
        print(f"Skipping {folder_path}: 'pre_processing' folder not found.")
        return

    # Step 1: Process Depth Map
    depth_output_path = os.path.join(pre_processing_path, "depth.png")
    original_images = [f for f in os.listdir(folder_path) if f.startswith("bdy_") and f.lower().endswith(valid_extensions)]
    if original_images:
        original_image_path = os.path.join(folder_path, original_images[0])
        # NOTE: Uncomment the following line to process the depth map if needed:
        # process_depth_map(original_image_path, depth_output_path)

    # Step 2: Process Body Mask
    body_mask_path = os.path.join(pre_processing_path, "body_mask")
    white_mask_output_path = None
    black_mask_output_path = None
    if os.path.exists(body_mask_path):
        mask_images = [f for f in os.listdir(body_mask_path) if f.startswith("bdy_") and f.endswith(".png")]
        if mask_images:
            input_mask_path = os.path.join(body_mask_path, mask_images[0])
            white_mask_output_path, black_mask_output_path = process_body_mask(
                input_mask_path,
                os.path.join(pre_processing_path, "white_fg_mask.png")
            )

    # Step 3: Process Depth Map with Black Body Mask (keep depth only where mask indicates foreground)
    if os.path.exists(depth_output_path) and black_mask_output_path and os.path.exists(black_mask_output_path):
        depth_map = Image.open(depth_output_path).convert("RGB")
        black_fg_mask = Image.open(black_mask_output_path).convert("L")
        # Invert the black mask: foreground (black, value 0) becomes 1 and background (white, value 255) becomes 0.
        multiplier = 1 - (np.array(black_fg_mask, dtype=np.uint8) / 255)
        filtered_depth = np.array(depth_map) * multiplier[:, :, None]
        depth_output_path_filtered = os.path.join(pre_processing_path, "depth_filtered.png")
        Image.fromarray(filtered_depth.astype(np.uint8)).save(depth_output_path_filtered)
        print(f"Filtered depth map with body mask: {depth_output_path_filtered}")



def process_all_folders(root_directory):
    """
    Process all subdirectories in the given root directory.

    Args:
        root_directory (str): Path to the parent directory containing subfolders.
    """
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            process_folder(folder_path)


# Example usage
root_directory = "/home/shenzhen/Datasets/dataset_with_garment_debug_100"
process_all_folders(root_directory)
