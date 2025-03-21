import torch
from diffusers import FluxControlPipeline, FluxFillPipeline
from PIL import Image
import numpy as np
import argparse
import os
import glob
import json
import glob
'''
First, run the inference
    bash inf.sh
    cd relighting-comparison

(build a local webpage)
    cd outputs/output_3_15_*/
    python -m http.server 8000
'''


relighting_prompts = {
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

    # More complex prompts (with more scene geometries and light sources)
    "golden_hour_2": "Relit by warm, golden-hour sunlight streaming through scattered tree branches, casting long, soft-edged shadows. The golden light reflects off surfaces, creating a rich, glowing warmth, while the dappled illumination highlights contours of the terrain and architecture.",
    "moonlight_2": "Relit by soft, bluish moonlight entering through an open window, casting diffused, elongated shadows. The light gently reflects off smooth surfaces, giving a cool, silvery glow, while deeper shadows soften edges, enhancing the tranquil nighttime atmosphere.",
    "noon_sunlight_2": "Relit by bright, overhead noon sunlight, casting short, sharply defined shadows. The intense light highlights textures like rough pavement and reflective glass, while shaded areas create stark contrast, emphasizing a crisp, high-contrast atmosphere.",
    "neon_lights_2": "Relit by vibrant neon signs glowing above rain-slicked streets, casting dynamic reflections in pink, blue, and purple hues. The neon glow seeps into narrow alleys, illuminating edges and silhouettes with a futuristic, high-energy ambiance.",
    "candlelight_2": "Relit by flickering candlelight, casting warm, golden hues that shift subtly across nearby surfaces. The soft light dances on wooden textures and fabric folds, while distant shadows blur gently, creating an intimate and cozy setting.",
    "spotlight_2": "Relit by a harsh, focused spotlight from above, creating extreme contrast with bright highlights and deep, hard-edged shadows. The intense beam isolates the subject, leaving surrounding areas in near-total darkness.",
    "thunderstorm_2": "Relit by sudden lightning flashes illuminating a stormy sky, casting intense, high-contrast light. The brief bursts expose rain-soaked textures and silhouettes of buildings before vanishing into deep shadows.",
    "meteor_shower_2": "Relit by streaking meteors across the night sky, casting short-lived, shifting glows. The momentary bursts of light reflect off metal and glass surfaces, creating fleeting highlights against a deep cosmic darkness.",
    "volcanic_glow_2": "Relit by molten lava's fiery red-orange glow, casting intense, flickering shadows. The hot light glows against jagged rocks, while thick, rising smoke softens distant edges with an ominous haze.",
    "foggy_morning_2": "Relit by diffused morning light filtering through thick fog, muting colors and softening edges. The fog scatters the light, creating a uniform glow, while silhouettes of trees and buildings fade into the misty distance.",
}

# TODO_Later: try identity loss (src vs. gen) to maintain person identity

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





# NOTE: create json and symlink (only once for each experiments)

# # Define the path for JSON output
# json_path = os.path.join("outputs", args.output_dir, "image_list.json")

# # Build the list of categories from your relighting prompts
# categories = list(relighting_prompts.keys())

# # Get all folder names from input_dir (each folder represents one image set)
# folder_names = sorted([
#     os.path.basename(d)
#     for d in os.listdir(args.input_dir)
#     if os.path.isdir(os.path.join(args.input_dir, d))
# ])

# information = []
# for folder in folder_names:
#     folder_path = os.path.join(args.input_dir, folder)
#     # List all files in the folder
#     files = os.listdir(folder_path)
#     # Filter files that start with "bdy_"
#     bdy_files = [f for f in files if f.startswith("bdy_")]
#     if not bdy_files:
#         # If no matching file is found, skip this folder or handle as needed
#         continue
#     # Pick the first match (alphabetically sorted for consistency)
#     bdy_file = sorted(bdy_files)[0]
#     # Construct the relative paths:
#     # For input, use the folder name and the discovered file name.
#     # For output, assume the convention is foldername + ".png".
#     info_item = {
#         "input": f"{folder}/{bdy_file}",
#         "output": f"{folder}.png"
#     }
#     information.append(info_item)

# # Combine the two parts into one dictionary
# data = {
#     "categories": categories,
#     "information": information
# }

# # Save the JSON
# with open(json_path, "w") as f:
#     json.dump(data, f, indent=4)

# # Determine the output directory (where the JSON is stored)
# output_dir = os.path.join("outputs", args.output_dir)
# # Determine the symlink path: it will be created inside the output directory,
# # with the same basename as the input directory.
# symlink_path = os.path.join(output_dir, os.path.basename(args.input_dir))

# # If the symlink already exists, remove it first.
# if os.path.lexists(symlink_path):
#     os.remove(symlink_path)

# # Create the symlink. Note that os.symlink(source, link_name) creates a symbolic link.
# os.symlink(args.input_dir, symlink_path)
# # ------------------------------------------------------

# exit()






# **🔹 Load pre-trained model on the specified GPU**
device = f"cuda:{args.gpu}"

if args.flux_type == "FluxFillPipeline":
    # pipe = FluxFillPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-Fill-dev",
    #     torch_dtype=torch.bfloat16
    # ).to(device)  # ✅ Correct
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

    with open(annotation_path, "r") as f:
        base_prompt = f.read().strip()

    # **🔹 Only process the specified relighting type**
    relight_id = args.relight_type
    modified_prompt = relighting_prompts[relight_id]

    final_prompt = (
        f"A photo of a person in a 2 by 1 grid. "
        f"On the left, {base_prompt} "
        f"On the right, {modified_prompt}."
    )

    # Load and process depth image
    if args.flux_type == "FluxControlPipeline":

        # NOTE: add depth_fg_only option
        if args.depth_fg_only:
            depth_path = os.path.join(subfolder_path, "pre_processing/depth_filtered.png")
        else:
            depth_path = os.path.join(subfolder_path, "pre_processing/depth.png")
            
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

    # Load and process body mask (TODO: this is not working. needs debug)
    elif args.flux_type == "FluxFillPipeline":

        # # Load body image
        # body_path = glob.glob(os.path.join(subfolder_path, "bdy_*.*"))[0]
        # body_image = Image.open(body_path).convert("RGB")

        # # Locate body mask (inverted white =========> black)
        # mask_path = os.path.join(subfolder_path, "pre_processing/black_fg_mask.png")
        # body_mask = Image.open(mask_path).convert("L")

        # image = pipe(
        #     prompt=modified_prompt, # NOTE: use relight prompts only
        #     image=body_image,
        #     mask_image=body_mask, 
        #     height=args.height,
        #     width=args.width,
        #     guidance_scale=args.cfg,
        #     num_inference_steps=args.num_steps,
        #     max_sequence_length=512,
        #     generator=torch.Generator("cpu").manual_seed(42)
        # ).images[0]
        raise NotImplementedError("FluxFillPipeline is not implemented yet.")

    # Create relighting-specific output directory
    output_dir_relight = os.path.join("outputs", args.output_dir, relight_id)
    os.makedirs(output_dir_relight, exist_ok=True)

    # Save output image with folder name as filename
    output_filename = f"{os.path.basename(subfolder_path)}.png"
    output_path = os.path.join(output_dir_relight, output_filename)

    # Save the image
    image.save(output_path)
    print(f"[{count+1}] Saved: {output_path}")

    count += 1

    # Stop if max_images is reached
    if args.max_images and count >= args.max_images:
        print(f"Reached max_images limit ({args.max_images}). Stopping.")
        break  # Exit inner loop