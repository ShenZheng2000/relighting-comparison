from PIL import Image, ImageDraw
import numpy as np
import os
import torch
from utils import prepare_canvas_and_mask, process_depth_map
from diffusers import FluxFillPipeline
from transformers import pipeline
from image_gen_aux import DepthPreprocessor


use_v2 = False
depth_model = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")

# use_v2 = True
# depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# ---------------------------
# Example Standalone Script
# ---------------------------
def main():

    # ----------- Outpainting Stage ----------- #
    print("Loading outpainting pipeline...")
    device = f"cuda"
    pipe_outpaint = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe_outpaint.set_progress_bar_config(disable=True)


    # Configuration parameters (modify as needed)
    # target_width = 768
    # target_height = 768

    # NOTE: use a number that is divisible by both 16 (flux) and 14 (depth anything)
    target_width = 784
    target_height = 784


    cfg_outpaint = 30  # guidance scale
    num_inference_steps = 50

    # Paths to source assets (update these paths)
    source_image_path = "/home/shenzhen/Datasets/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/bdy_1.png"
    body_mask_path = "/home/shenzhen/Datasets/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/pre_processing/black_fg_mask_groundedsam2.png"
    base_prompt = "background features a wooden side table with a lamp and framed pictures on the wall, light-colored walls, wooden floor."  # Replace with your prompt
    outpaint_folder = "output_outpaint_debug/09WOMEN_WOMEN_BLOUSE_167"

    # source_image_path = "/home/shenzhen/Relight_Projects/relighting-comparison/data/dataset_with_garment_debug_100/Adsb_Women_Skirts_008/bdy_2.webp"
    # body_mask_path = "/home/shenzhen/Relight_Projects/relighting-comparison/data/dataset_with_garment_debug_100/Adsb_Women_Skirts_008/pre_processing/black_fg_mask_groundedsam2.png"
    # base_prompt = "background of a weathered brick wall, urban setting, sunlight casting shadows."  # Replace with your prompt
    # outpaint_folder = "output_outpaint_debug/Adsb_Women_Skirts_008"

    os.makedirs(outpaint_folder, exist_ok=True)

    # Load source image and body mask
    source_image = Image.open(source_image_path).convert("RGB")
    body_mask = Image.open(body_mask_path).convert("L")

    # TODO: uncomment later for outpainting
    # -----------------------------
    # Step 1: Prepare Canvas & Mask
    # -----------------------------
    # Base version (no fg mask)
    print("Preparing canvas and mask...")
    canvas_base_no, mask_base_no, _, _, _ = prepare_canvas_and_mask(
        source_image, target_width, target_height, apply_fg_mask=False
    )
    
    # -----------------------------
    # Step 2: Run Outpainting (Base)
    # -----------------------------
    print("Running outpainting...")
    img_out_base_no = pipe_outpaint(
        prompt=base_prompt,
        image=canvas_base_no,
        mask_image=mask_base_no,
        height=target_height,
        width=target_width,
        guidance_scale=cfg_outpaint,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]

    # Save the base canvas and the outpainted image
    canvas_base_path = os.path.join(outpaint_folder, "canvas_base_no.png")
    outpaint_base_path = os.path.join(outpaint_folder, "img_out_base_no.png")

    # TODO: uncomment later for outpainting
    canvas_base_no.save(canvas_base_path)
    img_out_base_no.save(outpaint_base_path)
    print("Saved base canvas and outpainted image.")

    # -----------------------------
    # Step 3: Process Depth Maps
    # -----------------------------
    depth_canvas_path = os.path.join(outpaint_folder, "depth_canvas.png")
    depth_outpaint_path = os.path.join(outpaint_folder, "depth_outpaint.png")

    process_depth_map(canvas_base_path, depth_canvas_path, depth_model, use_v2)
    process_depth_map(outpaint_base_path, depth_outpaint_path, depth_model, use_v2)

    print("Saved depth maps.")

    # -----------------------------
    # Step 4: Overlay Depth Maps
    # -----------------------------
    # Helper function to overlay depth map on an image.
    def overlay_depth(image_path, depth_path, output_path, alpha=0.5):
        base_img = Image.open(image_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("RGB")
        # Blend depth map on top of the base image.
        overlay = Image.blend(base_img, depth_img, alpha=alpha)
        overlay.save(output_path)
    
    overlay_canvas_path = os.path.join(outpaint_folder, "overlay_canvas_depth.png")
    overlay_outpaint_path = os.path.join(outpaint_folder, "overlay_outpaint_depth.png")
    overlay_depth(canvas_base_path, depth_canvas_path, overlay_canvas_path)
    overlay_depth(outpaint_base_path, depth_outpaint_path, overlay_outpaint_path)
    print("Saved overlay images:")
    print(" -", overlay_canvas_path)
    print(" -", overlay_outpaint_path)

if __name__ == "__main__":
    main()
