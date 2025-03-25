import torch
from diffusers import FluxControlPipeline, FluxFillPipeline
from PIL import Image
import numpy as np
import argparse
import os
import glob
import json
import glob
from PIL import Image, ImageDraw, ImageFont
import yaml
from omegaconf import OmegaConf

'''
First, run the inference
    bash inf.sh
    cd relighting-comparison

(build a local webpage)
    cd outputs/output_3_15_*/
    python -m http.server 8000
'''

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


relighting_prompts = {
    # NOTE: add these into html files. 

    # DONE
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

    # # More complex prompts (more details about scene geometries and light sources) => SKIP FOR NOW
    # "golden_hour_2": "Relit by warm, golden-hour sunlight streaming through scattered tree branches, casting long, soft-edged shadows. The golden light reflects off surfaces, creating a rich, glowing warmth, while the dappled illumination highlights contours of the terrain and architecture.",
    # "moonlight_2": "Relit by soft, bluish moonlight entering through an open window, casting diffused, elongated shadows. The light gently reflects off smooth surfaces, giving a cool, silvery glow, while deeper shadows soften edges, enhancing the tranquil nighttime atmosphere.",
    # "noon_sunlight_2": "Relit by bright, overhead noon sunlight, casting short, sharply defined shadows. The intense light highlights textures like rough pavement and reflective glass, while shaded areas create stark contrast, emphasizing a crisp, high-contrast atmosphere.",
    # "neon_lights_2": "Relit by vibrant neon signs glowing above rain-slicked streets, casting dynamic reflections in pink, blue, and purple hues. The neon glow seeps into narrow alleys, illuminating edges and silhouettes with a futuristic, high-energy ambiance.",
    # "candlelight_2": "Relit by flickering candlelight, casting warm, golden hues that shift subtly across nearby surfaces. The soft light dances on wooden textures and fabric folds, while distant shadows blur gently, creating an intimate and cozy setting.",
    # "spotlight_2": "Relit by a harsh, focused spotlight from above, creating extreme contrast with bright highlights and deep, hard-edged shadows. The intense beam isolates the subject, leaving surrounding areas in near-total darkness.",
    # "thunderstorm_2": "Relit by sudden lightning flashes illuminating a stormy sky, casting intense, high-contrast light. The brief bursts expose rain-soaked textures and silhouettes of buildings before vanishing into deep shadows.",
    # "meteor_shower_2": "Relit by streaking meteors across the night sky, casting short-lived, shifting glows. The momentary bursts of light reflect off metal and glass surfaces, creating fleeting highlights against a deep cosmic darkness.",
    # "volcanic_glow_2": "Relit by molten lava's fiery red-orange glow, casting intense, flickering shadows. The hot light glows against jagged rocks, while thick, rising smoke softens distant edges with an ominous haze.",
    # "foggy_morning_2": "Relit by diffused morning light filtering through thick fog, muting colors and softening edges. The fog scatters the light, creating a uniform glow, while silhouettes of trees and buildings fade into the misty distance.",
}

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to the base configuration file")
parser.add_argument("--exp_config", type=str, required=True, help="Path to the experiment-specific configuration file")
parser.add_argument("--relight_type", type=str, required=True, help="Specify relighting type")
parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
args = parser.parse_args()

# Load and merge configurations using OmegaConf
base_cfg = OmegaConf.load(args.base_config)
exp_cfg = OmegaConf.load(args.exp_config)
config = OmegaConf.merge(base_cfg, exp_cfg)

# Inject CLI arguments into the merged configuration
config.relight_type = args.relight_type
config.gpu = args.gpu
config.output_dir = os.path.splitext(os.path.basename(args.exp_config))[0]

print(OmegaConf.to_yaml(config))

# **🔹 Load pre-trained model on the specified GPU**
device = f"cuda:{config.gpu}"

if config.flux_type == "FluxFillPipeline":
    raise NotImplementedError("FluxFillPipeline is not implemented yet.")
elif config.flux_type == "FluxControlPipeline":
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
else:
    raise ValueError(f"Invalid flux_type: {config.flux_type}")

pipe.set_progress_bar_config(disable=True)

# Track processed count
count = 0

# Iterate through each subfolder
for subfolder in sorted(os.listdir(config.input_dir)):
    subfolder_path = os.path.join(config.input_dir, subfolder)
    annotation_path = glob.glob(os.path.join(subfolder_path, "*.txt"))[0]
    source_image_path = glob.glob(os.path.join(subfolder_path, "bdy_*"))[0]
    source_image = Image.open(source_image_path).convert('RGB')
    
    with open(annotation_path, "r") as f:
        base_prompt = f.read().strip()
    
    relight_id = config.relight_type
    relight_prompt = relighting_prompts[relight_id]
    
    final_prompt = (
        f"A photo of a person in a 2 by 1 grid. "
        f"On the left, {base_prompt} "
        f"On the right, {relight_prompt}."
    )
    
    # Load depth image based on depth_mode (TODO: get depth1 and depth2 from outpainting pipeline)
    if config.flux_type == "FluxControlPipeline":
        if config.depth_mode == "filtered":
            depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered.png")
        elif config.depth_mode == "filtered_pad":
            depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered_pad.png")
        elif config.depth_mode == "raw":
            depth_path = os.path.join(subfolder_path, "pre_processing/depth.png")
        else:
            raise ValueError(f"Unknown depth_mode: {config.depth_mode}")
                    
        output_dir = config.output_dir
        depth_map = Image.open(depth_path).convert('RGB')
        depth_map_2x1 = Image.fromarray(np.hstack([np.array(depth_map), np.array(depth_map)]))
    
        if config.match_source_resolution:
            print("using source resolution!")
            height, width = source_image.height, source_image.width
            print(f"height: {height}, width: {width}")
        else:
            height, width = config.height, config.width
    
        image = pipe(
            prompt=final_prompt,
            control_image=depth_map_2x1,
            height=height,
            width=width * 2,
            guidance_scale=config.cfg,
            num_inference_steps=config.num_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(42),
        ).images[0]

        # NEW: Resize generated image if flag is set
        if config.resize_generated:
            print("Resizing generated image to match source resolution (2x width).")
            image = image.resize((source_image.width * 2, source_image.height), Image.BILINEAR)

    elif config.flux_type == "FluxFillPipeline":
        raise NotImplementedError("FluxFillPipeline is not implemented yet.")
    
    output_dir_relight = os.path.join("outputs", output_dir, relight_id)
    os.makedirs(output_dir_relight, exist_ok=True)
    
    output_filename = f"{os.path.basename(subfolder_path)}.png"
    output_path = os.path.join(output_dir_relight, output_filename)
    
    concatenated_image = concat_images_side_by_side(source_image, image)
    concatenated_image.save(output_path)
    print(f"[{count+1}] Saved: {output_path}")
    
    count += 1
    
    if config.max_images and count >= config.max_images:
        print(f"Reached max_images limit ({config.max_images}). Stopping.")
        break