import torch
from diffusers import FluxFillPipeline
from PIL import Image, ImageOps
import numpy as np

def crop_to_smallest_rectangle(original_image, mask_image, target_color="black"):
    """
    Crop the original image to the smallest bounding box enclosing either black or white areas in the mask.
    
    Parameters:
    - original_image: PIL.Image, the original RGB image.
    - mask_image: PIL.Image, the mask image (grayscale).
    - target_color: str, "black" to crop around black pixels, "white" to crop around white pixels.

    Returns:
    - result_image: PIL.Image, the cropped image on a black background.
    - new_mask: PIL.Image, a new mask highlighting the cropped area.
    """
    # Ensure the mask image is grayscale
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")
    
    # Convert mask image to numpy array for processing
    mask_array = np.array(mask_image)

    # Define pixel intensity based on target_color
    target_value = 0 if target_color == "black" else 255

    # Find the coordinates of target pixels
    target_pixels = np.where(mask_array == target_value)

    # Check if any target pixels exist
    if len(target_pixels[0]) == 0 or len(target_pixels[1]) == 0:
        raise ValueError(f"No {target_color} pixels found in the mask image.")

    # Get the bounding box of the target pixels
    x_min, x_max = np.min(target_pixels[1]), np.max(target_pixels[1])
    y_min, y_max = np.min(target_pixels[0]), np.max(target_pixels[0])

    # Create a new black image with the same size as the original image
    result_image = Image.new("RGB", original_image.size, (0, 0, 0))

    # Crop the region within the bounding box from the original image
    cropped_region = original_image.crop((x_min, y_min, x_max, y_max))

    # Paste the cropped region onto the black image
    result_image.paste(cropped_region, (x_min, y_min))

    # Create a new mask where the smallest rectangle area is black and the rest is white
    new_mask = Image.new("L", original_image.size, 255)  # Start with a white image
    new_mask.paste(0, (x_min, y_min, x_max, y_max))  # Set the rectangle area to black

    return result_image, new_mask

# File paths
input_f = "/home/shenzhen/Datasets/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/bdy_1.png"
mask_image_path = "/home/shenzhen/Datasets/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/pre_processing/black_fg_mask.png"
outf = "debug_mask.png"
new_mask_outf = "new_black_fg_mask.png"  # Path to save the new mask
num_steps = 30

# right_prompt = "Relit by warm, golden-hour sunlight filtering through the trees, casting long, soft-edged shadows and creating a dreamy, atmospheric glow."
right_prompt = "a person sitting in warm, golden-hour sunlight filtering through the trees, casting long, soft-edged shadows and creating a dreamy, atmospheric glow."

# Load model
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

# Load images using PIL
original_image = Image.open(input_f)
mask_image = Image.open(mask_image_path)

# Process the image and create a new mask
img_pil_in, new_mask = crop_to_smallest_rectangle(original_image, mask_image, target_color="black")
img_pil_in.save("debug_processed_input.png")  # Save for debugging
new_mask.save(new_mask_outf)  # Save the new mask

# Run FluxFillPipeline with processed image and new mask
img_out = pipe(
    prompt=right_prompt,
    image=img_pil_in,
    mask_image=new_mask,
    height=768,
    width=768,
    guidance_scale=30,
    num_inference_steps=num_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]

img_out.save(outf)




# import torch
# from diffusers import FluxFillPipeline
# from PIL import Image
# import numpy as np
# import argparse
# import os
# import glob
# import json

# input_f = "/home/shenzhen/Datasets/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/pre_processing/body_mask/bdy_1.png"
# mask_image = "/home/shenzhen/Datasets/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/pre_processing/black_fg_mask.png"
# outf = "debug_mask.png"
# num_steps = 30

# right_prompt = "Relit by warm, golden-hour sunlight"

# pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

# img_pil_in = Image.open(input_f).convert("RGB")

# mask = Image.open(mask_image)
# # mask = Image.fromarray(255 - np.array(mask.convert("L"))) # invert the mask
# img_out = pipe(prompt=right_prompt, image=img_pil_in, mask_image=mask, height=768, width=768, guidance_scale=30, 
#             num_inference_steps=num_steps, max_sequence_length=512, generator=torch.Generator("cpu").manual_seed(42)).images[0]
# img_out.save(outf)