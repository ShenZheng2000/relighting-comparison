# import os
# from PIL import Image

# # Settings
# root_dir = "outputs"
# folders = [f"exp_4_10_v{i}" for i in range(11)] + ["exp_4_10_v42"]
# relight_id = "golden_hour"
# crop_size = 784
# output_dir = "debug_outputs"
# os.makedirs(output_dir, exist_ok=True)

# # Get filenames from the first folder
# first_folder = os.path.join(root_dir, folders[0], relight_id)
# filenames = sorted([f for f in os.listdir(first_folder) if f.endswith(".png")])

# for filename in filenames:
#     relight_images = []
#     success = True
#     for folder in folders:
#         img_path = os.path.join(root_dir, folder, relight_id, filename)
#         if not os.path.exists(img_path):
#             print(f"Missing: {img_path}")
#             success = False
#             break
#         img = Image.open(img_path)
#         w, h = img.size
#         left = w - crop_size
#         relight = img.crop((left, 0, w, crop_size))
#         relight_images.append(relight)
    
#     if not success or len(relight_images) != 12:
#         continue

#     # Build 2x6 grid
#     grid_img = Image.new("RGB", (6 * crop_size, 2 * crop_size))
#     for idx, relight in enumerate(relight_images):
#         row, col = divmod(idx, 6)
#         grid_img.paste(relight, (col * crop_size, row * crop_size))
    
#     save_path = os.path.join(output_dir, filename)
#     grid_img.save(save_path)
#     print(f"Saved: {save_path}")