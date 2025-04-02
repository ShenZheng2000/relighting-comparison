import os
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor  # Ensure this module is correctly imported
import yaml

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
    orig_w, orig_h = img.size
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


def pad_depth_image(input_path, output_path, target_width, target_height):
    """
    Resize and pad a depth image to exactly target_width x target_height.
    If the image is larger in any dimension, it is scaled down (keeping aspect ratio)
    such that it fits within the target dimensions, then padded with zeros (black) to fill.

    Args:
        input_path (str): Path to the input depth image.
        output_path (str): Path to save the padded depth image.
        target_width (int): The target width.
        target_height (int): The target height.
    """
    img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img.size

    # Calculate scaling factor (do not upscale if image is already smaller)
    scale = min(target_width / orig_w, target_height / orig_h, 1.0)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Create a new black image of target dimensions
    new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    # Center the resized image in the new blank image
    pad_left = (target_width - new_w) // 2
    pad_top = (target_height - new_h) // 2

    new_img.paste(resized_img, (pad_left, pad_top))
    new_img.save(output_path)
    print(f"Saved padded depth image to {output_path}")


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
        # Uncomment the next line to generate the depth map if needed:
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

        # Step 4: Pad or resize the filtered depth map to match the original image's resolution
        # orig_img = Image.open(original_image_path)  # Load the original image
        # orig_w, orig_h = orig_img.size              # Get its dimensions
        # NOTE: hardcode as 768x768 for now
        orig_w, orig_h = 768, 768

        depth_output_path_filtered_pad = os.path.join(pre_processing_path, "depth_filtered_pad.png")
        pad_depth_image(depth_output_path_filtered, depth_output_path_filtered_pad, orig_w, orig_h)


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
