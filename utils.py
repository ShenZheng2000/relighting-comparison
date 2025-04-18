from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from image_gen_aux import DepthPreprocessor
from PIL import ImageOps

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
    "noon_sunlight": "relit by bright, overhead noon sunlight blazing over a lively urban plaza, sharply defining every corner with crisp shadows and vivid highlights on modern glass and concrete structures, creating a dynamic and energetic daytime scene.",
    "neon_lights": "relit by vibrant neon lights reflecting off rain-slicked city streets, where electric hues of pink, blue, and purple burst from storefronts and billboards, bathing the surroundings in a futuristic, cyberpunk glow.",
    "candlelight": "relit by the gentle flicker of candlelight in an intimate setting, where warm amber tones softly dance over rustic wooden surfaces and delicate fabrics, creating a cozy, nostalgic ambiance filled with quiet charm.",
    "foggy_morning": "relit by the soft, diffused light of an early foggy morning in a quiet countryside, where gentle rays pierce through a thick mist over dew-covered fields and ancient trees, creating a serene, dreamlike atmosphere.",
    "moonlight": "relit by soft, bluish moonlight filtering through an open window framed by gently swaying curtains, casting pale, silvery light across worn wooden floorboards, scattered books, and the edge of a cozy armchair, creating a serene, nighttime glow filled with quiet stillness.",
}

# NOTE: indoor scenes
relighting_prompts_3 = {
    "golden_hour": "relit by warm, golden-hour sunlight streaming through a living room window, casting long, soft-edged shadows across wooden floors and gently illuminating cozy furniture, creating a calm, atmospheric glow.",
    "noon_sunlight": "relit by bright noon sunlight pouring through large apartment windows, creating sharp, well-defined shadows on white walls and highlighting indoor plants and shelves, adding energy to the quiet space.",
    "neon_lights": "relit by colorful neon lights from signs outside a downtown apartment, casting pink, blue, and purple glows across a modern interior with glass tables and framed artwork, creating a futuristic, urban ambiance.",
    "candlelight": "relit by the gentle flicker of candlelight in a dimly lit room, where warm amber tones dance over bookshelves, soft cushions, and old wooden furniture, creating a cozy and nostalgic atmosphere.",
    "foggy_morning": "relit by soft, diffused morning light seeping through sheer curtains in a quiet bedroom, muting colors and softening edges of the bed, rug, and potted plants, creating a peaceful, dreamlike indoor scene.",
    "moonlight": "relit by soft, bluish moonlight filtering through a bedroom window, casting pale shadows across the bed, nightstand, and curtains, filling the space with a serene and quiet nighttime mood.",
}

# NOTE: extremely simplified 
relighting_prompts_4 = {
    "golden_hour": "relit by golden-hour sunlight.",
    "noon_sunlight": "relit by bright noon sunlight.",
    "neon_lights": "relit by colorful neon lights.",
    "candlelight": "relit by warm candlelight.",
    "foggy_morning": "relit by diffused foggy morning light.",
    "moonlight": "relit by soft moonlight.",
}

# NOTE: explicitly mention the relighting direction => not working
relighting_prompts_5 = {
    "golden_hour_front": "Relit by golden-hour sunlight shining directly on the subject's face, illuminating the front evenly.",
    "golden_hour_side": "Relit by golden-hour sunlight coming from the side, casting soft shadows across the subject's face.",
    "golden_hour_back": "Relit by golden-hour sunlight coming from behind the subject, creating a warm rim light around the hair and shoulders.",
}

relighting_prompts_6 = {
    # all of these are not working!
    # "golden_hour_front": "relit by golden-hour light directly on the face, evenly lit.",
    # "golden_hour_front": "relit by golden-hour light from the front, fully lighting the face and body.",
    # "golden_hour_front": "relit by golden-hour light hitting the subject straight on, lighting the face clearly with no shadow.",
    # "golden_hour_front": "relit by golden-hour light directly in front of the subject, fully illuminating the face and body, no side shadows.",
    # "golden_hour_front": "relit by strong golden-hour light shining straight into the face, no shadows.",
    # "golden_hour_front": "relit with harsh golden-hour light from the front, full facial illumination.",
    # "golden_hour_front": "relit by direct golden-hour sun in front, face and body fully lit, flat lighting.",
    # "golden_hour_front": "relit by intense golden-hour light directly facing the subject, no side light.",
    # "golden_hour_front": "relit with golden-hour light aligned with camera, fully frontal, shadowless.",
    # "golden_hour_front": "relit by golden-hour light directly casting onto the person's face, no shadows.",
    # "golden_hour_front": "relit with golden-hour sunlight shining straight onto the face, fully illuminated.",
    # "golden_hour_front": "relit by warm light directly illuminating the face and body from the front.",
    # "golden_hour_front": "relit with golden-hour sun hitting the person's face directly, flat light.",
    # "golden_hour_front": "relit by sunlight casting frontally on the face, erasing all side shadows.",
    # "golden_hour_front": "golden-hour sunlight placed directly behind the camera, casting full light onto the subject's face and body.",
    # "golden_hour_front": "subject lit by golden-hour sun positioned at camera level, light hitting face straight on with no angle.",
    # "golden_hour_front": "golden-hour light blasting directly onto the face from the camera's direction, removing all shadows.",
    # "golden_hour_front": "strong golden-hour spotlight aligned with the camera, fully illuminating the subject's face and body.",
    # "golden_hour_front": "golden-hour sun directly in front of the person, between camera and subject, flooding the face with light.",
    # "golden_hour_side": "relit by golden-hour light from the side, casting shadows.", # not working
    # "golden_hour_left": "relit by golden-hour light from left, leaving the right side in deep shadow.", # not working
    # "golden_hour_right": "relit by golden-hour light from right, leaving the left side in deep shadow.", # not working

    # NOTE: these prompts looks good!!!
    "golden_hour_back": "relit with golden-hour sunlight from behind, face in shadow, rim glow",
    "golden_hour_side": "relit with golden-hour sunlight from the side, one side lit, one side shadowed",
    "golden_hour_front": "relit with golden-hour sunlight from the front, face fully lit, no shadows",

    # "golden_hour_left": "relit by golden-hour light from the left, brightening the left side of the person while leaving the other half in dark shadow.",  
    # "golden_hour_right": "relit by golden-hour light from the right, brightening the right side of the person while leaving the other half in dark shadow.",  
        
    # "golden_hour_0": "relit by golden-hour light from 0 degrees, shining directly onto the front of the subject's face.", # not working
    # "golden_hour_90": "relit by golden-hour light from 90 degrees, hitting the subject from the right side, strong side lighting.", # not working
    # "golden_hour_180": "relit by golden-hour light from 180 degrees, shining from directly behind, creating a rim light and shadowed face.", # not working
    # "golden_hour_270": "relit by golden-hour light from 270 degrees, hitting the subject from the left side, strong side lighting.", # not working
}


relighting_prompts_7 = {
    # TODO: add explicit direction (from 6), and add more details (like 1 or 2, whichever works better)
}


# Register available prompt versions
relighting_prompt_versions = {
    "1": relighting_prompts, # default one. no need to specify in config
    "2": relighting_prompts_2,
    "3": relighting_prompts_3,
    "4": relighting_prompts_4, 
    "5": relighting_prompts_5,
    "6": relighting_prompts_6,
    "7": relighting_prompts_7,
    # Add future versions like "4": relighting_prompts_4 here
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
    
        outpaint_folder = os.path.join("outpaint", config.output_dir, relight_id, os.path.basename(subfolder_path))

        if config.relight_image_only:
            # Load only the relight depth map.
            relight_path = os.path.join(outpaint_folder, "depth_relight.png")
            depth_map = Image.open(relight_path).convert('RGB')
            return depth_map
        else:
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

def extract_foreground(prompt: str) -> str:
    index = prompt.lower().find("background")
    if index != -1:
        return prompt[:index].strip(",. ")
    return prompt.strip(",. ")


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

def prepare_canvas_and_mask(image, target_width, target_height, apply_fg_mask=False, body_mask=None, crop_to_foreground=False):
    '''
    Resizes and centers an image on a fixed-size canvas with black padding, optionally creating a matching mask.
    If crop_to_foreground is True and a body_mask is provided, the image is tightly cropped to the foreground.
    '''

    # --- Crop to foreground if enabled ---
    if crop_to_foreground and body_mask is not None:
        # Invert, since the foreground is black
        inverted_mask = ImageOps.invert(body_mask)
        bbox = inverted_mask.getbbox()
        if bbox is not None:
            left, upper, right, lower = bbox
            image = image.crop((left, upper, right, lower))
            body_mask = body_mask.crop((left, upper, right, lower))

    # --- Resize image and paste to canvas ---
    orig_w, orig_h = image.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    canvas.paste(image.resize((new_w, new_h), Image.LANCZOS), (x_offset, y_offset))

    # --- Create the mask ---
    if apply_fg_mask and body_mask is not None:
        body_mask_resized = body_mask.resize((new_w, new_h), Image.NEAREST)
        mask_canvas = Image.new("L", (target_width, target_height), color=255)
        mask_canvas.paste(body_mask_resized, (x_offset, y_offset))
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