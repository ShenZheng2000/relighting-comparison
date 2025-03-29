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
from utils import concat_images_side_by_side

'''
First, run the inference
    bash inf.sh
    cd relighting-comparison

(build a local webpage)
    cd outputs/XXXXXXXX/
    python -m http.server 8000
'''


# NOTE: stick to this prompts for now 
relighting_prompts = {
    # NOTE: add these into html files. 
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


# # NOTE: hardcode these prompts for now!!!!!!!!
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

# use relighting_prompts_2
if config.relighting_prompts_2:
    relighting_prompts = relighting_prompts_2

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

    # NOTE: set up this prompt (more constraint on person identity)
    final_prompt_2 = (
        f"A 2x1 image grid;"
        f"On the left, {base_prompt} "
        f"On the right, the same person {relight_prompt}."
    )

    if config.final_prompt_2:
        print("using final_prompt_2")
        final_prompt = final_prompt_2
    
    # Load depth image based on depth_mode
    if config.flux_type == "FluxControlPipeline":
        if config.depth_mode == "filtered":
            depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered.png")
        elif config.depth_mode == "filtered_pad":
            depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered_pad.png")
        elif config.depth_mode == "raw":
            depth_path = os.path.join(subfolder_path, "pre_processing/depth.png")
        elif "outpaint" in config.depth_mode:
            HARDCODE_PATH = f"../outpaint/{config.depth_mode}"
            depth_path_base = os.path.join(HARDCODE_PATH, relight_id, subfolder, "depth_base.png")
            depth_path_relight = os.path.join(HARDCODE_PATH, relight_id, subfolder, "depth_relight.png")
        else:
            raise ValueError(f"Unknown depth_mode: {config.depth_mode}")

        output_dir = config.output_dir      

        if "outpaint" in config.depth_mode:
            depth_map_base = Image.open(depth_path_base).convert('RGB')
            depth_map_relight = Image.open(depth_path_relight).convert('RGB')
            assert depth_map_base.size == depth_map_relight.size, "Depth maps must have the same size!"
            depth_map_2x1 = Image.fromarray(np.hstack([np.array(depth_map_base), np.array(depth_map_relight)]))
        else:
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