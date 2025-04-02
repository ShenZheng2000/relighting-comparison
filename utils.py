from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from image_gen_aux import DepthPreprocessor

# depth_processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")

# NOTE: hardcoded prompts
relighting_prompts = {
    "golden_hour": "Relit by warm, golden-hour sunlight filtering through the trees, casting long, soft-edged shadows and creating a dreamy, atmospheric glow.",
    "moonlight": "Relit by soft, bluish moonlight streaming through an open window, casting gentle, diffused shadows and creating a serene, nighttime ambiance.",
    "noon_sunlight": "Relit by bright, overhead noon sunlight, creating strong, well-defined shadows with high contrast and a crisp, sharp atmosphere.",
    "neon_lights": "Relit by vibrant neon signs reflecting off wet pavement, casting colorful, dynamic glows in shades of pink, blue, and purple, creating a futuristic cyberpunk mood.",
    "candlelight": "Relit by flickering candlelight, casting soft, warm, golden hues with gentle, moving shadows, creating an intimate and cozy ambiance.",
    "spotlight": "Relit by a harsh, focused spotlight, creating extreme contrast with bright highlights and deep, sharp-edged shadows.",
    "thunderstorm": "Relit by flashes of lightning in a dark storm, creating dramatic, high-contrast illumination with deep shadows and eerie blue highlights.",
    "meteor_shower": "Relit by streaking meteors across the night sky, casting fleeting, dynamic glows with shifting highlights and deep cosmic shadows.",
    "volcanic_glow": "Relit by the fiery red-orange glow of molten lava, casting intense, flickering shadows with deep contrast and an apocalyptic atmosphere.",
    "foggy_morning": "Relit by soft, diffused morning light filtering through thick fog, muting colors and softening edges to create an ethereal, mysterious ambiance.",
}

# NOTE: I make the prompt longer (more details and more concret objects), and use relit instead of Relit (though it should not matter) => suitable for outpainting
relighting_prompts_2 = {
    "golden_hour": "relit by warm, golden-hour sunlight streaming through tall oak trees in a tranquil park, highlighting patches of wildflowers and casting long, soft-edged shadows across the grassy ground, creating a dreamy, atmospheric glow.",
    "moonlight": "relit by cool, bluish moonlight streaming through an ancient, open window of a secluded manor, softly illuminating weathered stone walls draped in ivy and casting gentle, diffused shadows across a dew-kissed courtyard, evoking a serene, enchanted nocturnal ambiance.",
    "noon_sunlight": "relit by bright, overhead noon sunlight blazing over a lively urban plaza, sharply defining every corner with crisp shadows and vivid highlights on modern glass and concrete structures, creating a dynamic and energetic daytime scene.",
    "neon_lights": "relit by vibrant neon lights reflecting off rain-slicked city streets, where electric hues of pink, blue, and purple burst from storefronts and billboards, bathing the surroundings in a futuristic, cyberpunk glow.",
    "candlelight": "relit by the gentle flicker of candlelight in an intimate setting, where warm amber tones softly dance over rustic wooden surfaces and delicate fabrics, creating a cozy, nostalgic ambiance filled with quiet charm.",
    "spotlight": "relit by a harsh, focused spotlight that isolates its subject on a dark stage, casting stark, dramatic shadows and accentuating fine details, resulting in an intense, theatrical visual impact.",
    "thunderstorm": "relit by sudden bursts of lightning during a raging thunderstorm, illuminating turbulent, swirling clouds and casting deep, shifting shadows over rain-soaked landscapes, evoking a dramatic, high-contrast spectacle.",
    "meteor_shower": "relit by a dazzling meteor shower streaking across a starry night sky over a barren desert, with each fleeting, radiant trail briefly lighting up the horizon and lending an ethereal, cosmic mystique.",
    "volcanic_glow": "relit by the fierce, fiery glow of molten lava cascading down a rugged mountainside, its vivid red and orange hues flickering against dark, ashen terrain, evoking an apocalyptic, otherworldly scene.",
    "foggy_morning": "relit by the soft, diffused light of an early foggy morning in a quiet countryside, where gentle rays pierce through a thick mist over dew-covered fields and ancient trees, creating a serene, dreamlike atmosphere."
}

def concat_images_side_by_side(image1, image2):
    """
    Concatenates two images side-by-side, resizing both to the height of image2,
    
    Args:
        image1 (PIL.Image): The first image (left side).
        image2 (PIL.Image): The second image (right side).
    Returns:
        PIL.Image: The concatenated image
    """
    # Resize both images to the height of image2, preserving aspect ratio
    target_height = image2.height
    
    # Resize image1
    aspect_ratio1 = image1.width / image1.height
    new_width1 = int(target_height * aspect_ratio1)
    image1 = image1.resize((new_width1, target_height))
    
    # Create a new blank image with combined width
    concatenated_image = Image.new('RGB', (image1.width + image2.width, target_height))
    
    # Paste the images side-by-side
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (image1.width, 0))
    
    return concatenated_image


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to the base configuration file")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to the experiment-specific configuration file")
    parser.add_argument("--relight_type", type=str, required=True, help="Specify relighting type")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
    return parser.parse_args()


def load_config(args):
    """Load and merge configuration files and inject CLI arguments."""
    base_cfg = OmegaConf.load(args.base_config)
    exp_cfg = OmegaConf.load(args.exp_config)
    config = OmegaConf.merge(base_cfg, exp_cfg)
    config.relight_type = args.relight_type
    config.gpu = args.gpu
    config.output_dir = os.path.splitext(os.path.basename(args.exp_config))[0]
    print(OmegaConf.to_yaml(config))
    return config


# def load_pipeline(config):
#     """Load the appropriate pre-trained model on the specified GPU."""
#     device = f"cuda:{config.gpu}"
#     pipe = FluxControlPipeline.from_pretrained(
#             "black-forest-labs/FLUX.1-Depth-dev",
#             torch_dtype=torch.bfloat16
#     ).to(device)
#     pipe.set_progress_bar_config(disable=True)
#     return pipe


def load_depth_map(subfolder_path, config, relight_id):
    if config.depth_mode == "filtered":
        depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered.png")
        depth_map = Image.open(depth_path).convert('RGB')
        return Image.fromarray(np.hstack([np.array(depth_map), np.array(depth_map)]))

    elif config.depth_mode == "filtered_pad":
        depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered_pad.png")
        depth_map = Image.open(depth_path).convert('RGB')
        return Image.fromarray(np.hstack([np.array(depth_map), np.array(depth_map)]))

    elif config.depth_mode == "raw":
        depth_path = os.path.join(subfolder_path, "pre_processing/depth.png")
        depth_map = Image.open(depth_path).convert('RGB')
        return Image.fromarray(np.hstack([np.array(depth_map), np.array(depth_map)]))

    elif "outpaint" in config.depth_mode:
        # hardcoded_path = f"outpaint/{config.depth_mode}"
        # base_path = os.path.join(hardcoded_path, relight_id, os.path.basename(subfolder_path), "depth_base.png")
        # relight_path = os.path.join(hardcoded_path, relight_id, os.path.basename(subfolder_path), "depth_relight.png")

        outpaint_folder = os.path.join("outpaint", config.output_dir, relight_id, os.path.basename(subfolder_path))
        base_path = os.path.join(outpaint_folder, "depth_base.png")
        relight_path = os.path.join(outpaint_folder, "depth_relight.png")

        base = Image.open(base_path).convert('RGB')
        relight = Image.open(relight_path).convert('RGB')
        
        assert base.size == relight.size, "Depth maps must have the same size!"
        return Image.fromarray(np.hstack([np.array(base), np.array(relight)]))

    else:
        raise ValueError(f"Unknown depth_mode: {config.depth_mode}")


def extract_background(prompt: str) -> str:
    # Find the index where "background" starts, ignoring case.
    index = prompt.lower().find("background")
    if index != -1:
        return prompt[index:]
    return ""


# def process_depth_map(input_path, output_path):
#     """
#     Process an image to generate its depth map.

#     Args:
#         input_path (str): Path to the input image.
#         output_path (str): Path to save the generated depth map.
#     """
#     control_image = load_image(input_path)
#     depth_map = depth_processor(control_image)[0].convert("RGB")
#     depth_map.save(output_path)
#     print(f"Saved depth map to {output_path}")


def process_depth_map(input_path, output_path, depth_model, use_v2=False):
    """
    Process an image to generate its depth map using the provided depth model.
    
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the generated depth map.
        depth_model: A pre-loaded depth estimation model (pipeline or processor).
        use_v2 (bool): Flag indicating if the new pipeline is used.
    """
    image = Image.open(input_path).convert("RGB")
    
    if use_v2:
        depth_map = depth_model(image)["depth"]
    else:
        control_image = load_image(input_path)
        depth_map = depth_model(control_image)[0].convert("RGB")
    
    depth_map.save(output_path)
    print(f"Saved depth map to {output_path}")


def process_body_mask(input_path, black_mask_output_path):
    """
    Convert an RGBA image to a black foreground mask.
    Foreground will be black (0), background white (255).

    Args:
        input_path (str): Path to the input RGBA body mask.
        black_mask_output_path (str): Path to save the black foreground mask.

    Returns:
        str: Path to the saved black foreground mask.
    """
    img = Image.open(input_path).convert("RGBA")
    alpha_channel = np.array(img)[:, :, 3]

    black_fg_mask = np.where(alpha_channel > 0, 0, 255).astype(np.uint8)
    Image.fromarray(black_fg_mask).save(black_mask_output_path)

    print(f"Saved black foreground mask: {black_mask_output_path}")
    return black_mask_output_path

# def process_body_mask(input_path, white_mask_output_path):
#     """
#     Convert an RGBA image to both white and black foreground masks.
#     The white foreground mask has the foreground in white and background in black.
#     The black foreground mask has the foreground in black and background in white.
    
#     Args:
#         input_path (str): Path to the input RGBA body mask.
#         white_mask_output_path (str): Path to save the white foreground mask.
    
#     Returns:
#         tuple: (white_mask_path, black_mask_path)
#     """
#     img = Image.open(input_path).convert("RGBA")
#     orig_w, orig_h = img.size
#     img = np.array(img)

#     # Extract the alpha channel (mask)
#     alpha_channel = img[:, :, 3]

#     # Create the white foreground mask (foreground in white, background in black)
#     white_fg_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
#     # Create the black foreground mask (foreground in black, background in white)
#     black_fg_mask = np.where(alpha_channel > 0, 0, 255).astype(np.uint8)

#     # Save the masks
#     Image.fromarray(white_fg_mask).save(white_mask_output_path)
#     black_mask_output_path = white_mask_output_path.replace("white_fg_mask.png", "black_fg_mask.png")
#     Image.fromarray(black_fg_mask).save(black_mask_output_path)
#     print(f"Saved masks: {white_mask_output_path}, {black_mask_output_path}")
#     return white_mask_output_path, black_mask_output_path

# def prepare_canvas_and_mask(image, target_width, target_height, apply_fg_mask=False, body_mask=None):
#     """
#     If apply_fg_mask is False: keep entire original image, outpaint only out-of-bound areas.
#     If True: preserve only the body_mask (black) and outpaint background+OOB (white).
#     """
#     orig_w, orig_h = image.size
#     # scale = min(target_width / orig_w, target_height / orig_h, 1.0)
#     scale = min(target_width / orig_w, target_height / orig_h)
#     new_w, new_h = int(orig_w * scale), int(orig_h * scale)

#     # Resize image and mask
#     image = image.resize((new_w, new_h), Image.LANCZOS)
#     if apply_fg_mask and body_mask is not None:
#         body_mask = body_mask.resize((new_w, new_h), Image.NEAREST)

#     # Create a blank canvas for the image
#     canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
#     x_offset = (target_width - new_w) // 2
#     y_offset = (target_height - new_h) // 2
#     canvas.paste(image, (x_offset, y_offset))

#     if apply_fg_mask and body_mask is not None:
#         # body_mask: black=preserved, white=outpaint
#         mask_canvas = Image.new("L", (target_width, target_height), color=255)
#         mask_canvas.paste(body_mask, (x_offset, y_offset))
#         return canvas, mask_canvas
#     else:
#         # Only outpaint OOB: black=preserved image region, white=rest
#         mask_canvas = Image.new("L", (target_width, target_height), color=255)
#         draw = ImageDraw.Draw(mask_canvas)
#         draw.rectangle((x_offset, y_offset, x_offset + new_w, y_offset + new_h), fill=0)
#         return canvas, mask_canvas

def prepare_canvas_and_mask(image, target_width, target_height, apply_fg_mask=False, body_mask=None):
    orig_w, orig_h = image.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # # Debug prints for canvas creation.
    # print("prepare_canvas_and_mask:")
    # print("  Original image size:", image.size)
    # print("  Target size:", (target_width, target_height))
    # print("  Scale:", scale)
    # print("  New size:", (new_w, new_h))
    # print("  Offsets:", (x_offset, y_offset))
    
    canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    canvas.paste(image.resize((new_w, new_h), Image.LANCZOS), (x_offset, y_offset))

    if apply_fg_mask and body_mask is not None:
        body_mask_resized = body_mask.resize((new_w, new_h), Image.NEAREST)
        mask_canvas = Image.new("L", (target_width, target_height), color=255)
        mask_canvas.paste(body_mask_resized, (x_offset, y_offset))
        return canvas, mask_canvas, scale, x_offset, y_offset
    else:
        mask_canvas = Image.new("L", (target_width, target_height), color=255)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask_canvas)
        draw.rectangle((x_offset, y_offset, x_offset + new_w, y_offset + new_h), fill=0)
        return canvas, mask_canvas, scale, x_offset, y_offset

def resize_mask_to_canvas(mask, target_width, target_height):
    # Get original size
    orig_w, orig_h = mask.size
    # Compute scale preserving aspect ratio
    # scale = min(target_width / orig_w, target_height / orig_h, 1.0)
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    # Resize the mask using NEAREST (to preserve binary values)
    resized_mask = mask.resize((new_w, new_h), Image.NEAREST)
    # Create a blank canvas of target size (fill with white, assuming white is background)
    canvas = Image.new("L", (target_width, target_height), color=255)
    # Center the resized mask onto the canvas
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas.paste(resized_mask, (x_offset, y_offset))
    return canvas