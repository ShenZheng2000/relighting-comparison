# '''
# First, run the inference
#     bash inf.sh

# Then, start a local webpage
#     cd outputs/
#     python -m http.server 8000
# '''
# import os
# import glob
# import numpy as np
# import torch
# from PIL import Image, ImageDraw, ImageFont
# from utils import relighting_prompts, relighting_prompts_2
# from utils import concat_images_side_by_side, parse_arguments, load_config, load_pipeline, load_depth_map


# def process_subfolder(subfolder_path, config, pipe, prompts):
#     """Process a single subfolder: load inputs, run inference, and save output."""
#     # Load source image and base prompt.
#     annotation_path = glob.glob(os.path.join(subfolder_path, "*.txt"))[0]
#     source_image_path = glob.glob(os.path.join(subfolder_path, "bdy_*"))[0]
#     source_image = Image.open(source_image_path).convert('RGB')
#     with open(annotation_path, "r") as f:
#         base_prompt = f.read().strip()

#     relight_id = config.relight_type
#     relight_prompt = prompts[relight_id]

#     # Build final prompts.
#     final_prompt = (
#         f"A photo of a person in a 2 by 1 grid. "
#         f"On the left, {base_prompt} "
#         f"On the right, {relight_prompt}."
#     )
#     final_prompt_2 = (
#         f"A 2x1 image grid; "
#         f"On the left, {base_prompt} "
#         f"On the right, the same person {relight_prompt}."
#     )
#     if config.final_prompt_2:
#         print("using final_prompt_2")
#         final_prompt = final_prompt_2

#     # Load and prepare depth map.
#     if config.flux_type == "FluxControlPipeline":
#         depth_map_2x1 = load_depth_map(subfolder_path, config, relight_id)

#         # Determine image dimensions.
#         if config.match_source_resolution:
#             print("using source resolution!")
#             height, width = source_image.height, source_image.width
#             print(f"height: {height}, width: {width}")
#         else:
#             height, width = config.height, config.width

#         # Run inference.
#         image = pipe(
#             prompt=final_prompt,
#             control_image=depth_map_2x1,
#             height=height,
#             width=width * 2,
#             guidance_scale=config.cfg,
#             num_inference_steps=config.num_steps,
#             max_sequence_length=512,
#             generator=torch.Generator("cpu").manual_seed(42),
#         ).images[0]

#         if config.resize_generated:
#             print("Resizing generated image to match source resolution (2x width).")
#             image = image.resize((source_image.width * 2, source_image.height), Image.BILINEAR)
#     else:
#         raise NotImplementedError("FluxFillPipeline is not implemented yet.")

#     # Save the concatenated result.
#     output_dir_relight = os.path.join("outputs", config.output_dir, relight_id)
#     os.makedirs(output_dir_relight, exist_ok=True)
#     output_filename = f"{os.path.basename(subfolder_path)}.png"
#     output_path = os.path.join(output_dir_relight, output_filename)
#     concatenated_image = concat_images_side_by_side(source_image, image)
#     concatenated_image.save(output_path)
#     print(f"Saved: {output_path}")


# def main():
#     args = parse_arguments()
#     config = load_config(args)
#     pipe = load_pipeline(config)

#     # Choose which prompts to use.
#     prompts = relighting_prompts_2 if config.relighting_prompts_2 else relighting_prompts

#     count = 0
#     for subfolder in sorted(os.listdir(config.input_dir)):
#         subfolder_path = os.path.join(config.input_dir, subfolder)
#         if os.path.isdir(subfolder_path):
#             process_subfolder(subfolder_path, config, pipe, prompts)
#             count += 1
#             if config.max_images and count >= config.max_images:
#                 print(f"Reached max_images limit ({config.max_images}). Stopping.")
#                 break


# if __name__ == "__main__":
#     main()