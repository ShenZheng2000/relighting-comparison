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
parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing images and annotations")
parser.add_argument("--height", type=int, default=768) 
parser.add_argument("--width", type=int, default=768) 
parser.add_argument("--cfg", type=float, default=5) 
parser.add_argument("--num_steps", type=int, default=30)
parser.add_argument("--max_images", type=lambda x: int(x) if x.isdigit() else None, default=None, help="Maximum number of images to process")
parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
parser.add_argument("--relight_type", type=str, required=True, choices=relighting_prompts.keys(), help="Specify which relighting type to use")
parser.add_argument("--gpu", type=int, required=True, help="Specify which GPU to use (e.g., 0 for cuda:0)")
parser.add_argument("--flux_type", type=str, default="FluxControlPipeline") # [FluxControlPipeline, FluxFillPipeline]
parser.add_argument("--depth_fg_only", action="store_true", help="Use only the foreground of the depth map")
args = parser.parse_args()


# **🔹 Load pre-trained model on the specified GPU**
device = f"cuda:{args.gpu}"

if args.flux_type == "FluxFillPipeline":
    raise NotImplementedError("FluxFillPipeline is not implemented yet.")
elif args.flux_type == "FluxControlPipeline":
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev",
        torch_dtype=torch.bfloat16
    ).to(device)  # ✅ Correct
else:
    raise ValueError(f"Invalid flux_type: {args.flux_type}")

pipe.set_progress_bar_config(disable=True)

# Track processed count
count = 0

# Iterate through each subfolder
for subfolder in sorted(os.listdir(args.input_dir)):

    subfolder_path = os.path.join(args.input_dir, subfolder)

    # Locate annotation files
    annotation_path = glob.glob(os.path.join(subfolder_path, "*.txt"))[0]

    # Locate and load source image
    source_image_path = glob.glob(os.path.join(subfolder_path, "bdy_*"))[0]
    source_image = Image.open(source_image_path).convert('RGB')

    with open(annotation_path, "r") as f:
        base_prompt = f.read().strip()

    # **🔹 Only process the specified relighting type**
    relight_id = args.relight_type
    relight_prompt = relighting_prompts[relight_id]

    final_prompt = (
        f"A photo of a person in a 2 by 1 grid. "
        f"On the left, {base_prompt} "
        f"On the right, {relight_prompt}."
    )

    # Load and process depth image
    if args.flux_type == "FluxControlPipeline":

        # NOTE: add depth_fg_only option
        if args.depth_fg_only:
            print("using depth_fg_only")
            depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered.png")
            output_dir = f"{args.output_dir}_depth_fg_only"
        else:
            depth_path = os.path.join(subfolder_path, "pre_processing/depth.png")
            output_dir = args.output_dir
            
        depth_map = Image.open(depth_path).convert('RGB')
        depth_map_2x1 = Image.fromarray(np.hstack([np.array(depth_map), np.array(depth_map)]))

        # **🔹 Generate image on the specified GPU**
        image = pipe(
            prompt=final_prompt,
            control_image=depth_map_2x1,
            height=args.height,
            width=args.width * 2,
            guidance_scale=args.cfg,
            num_inference_steps=args.num_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]

    # Load and process body mask
    elif args.flux_type == "FluxFillPipeline":

        raise NotImplementedError("FluxFillPipeline is not implemented yet.")

    # Create relighting-specific output directory
    output_dir_relight = os.path.join("outputs", output_dir, relight_id)
    os.makedirs(output_dir_relight, exist_ok=True)

    # Save output image with folder name as filename
    output_filename = f"{os.path.basename(subfolder_path)}.png"
    output_path = os.path.join(output_dir_relight, output_filename)

    # concat source and generated images
    concatenated_image = concat_images_side_by_side(source_image, image)
    concatenated_image.save(output_path)
    print(f"[{count+1}] Saved: {output_path}")

    count += 1

    # Stop if max_images is reached
    if args.max_images and count >= args.max_images:
        print(f"Reached max_images limit ({args.max_images}). Stopping.")
        break  # Exit inner loop