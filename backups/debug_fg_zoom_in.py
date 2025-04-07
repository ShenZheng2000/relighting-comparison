from PIL import ImageOps

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


from PIL import Image, ImageDraw
import numpy as np

# Load your image (replace with actual path)
image = Image.open("/home/shenzhen/Relight_Projects/relighting-comparison/data/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/bdy_1.png")  # must be a PIL image

# Create a fake body mask for demonstration (e.g., a white square in the center)
body_mask = Image.open("/home/shenzhen/Relight_Projects/relighting-comparison/data/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/pre_processing/black_fg_mask_groundedsam2.png").convert("L")

# Desired canvas size
target_width = 784
target_height = 784

# Call the function
canvas, mask_canvas, scale, x_offset, y_offset = prepare_canvas_and_mask(
    image,
    target_width,
    target_height,
    apply_fg_mask=True,
    body_mask=body_mask,
    crop_to_foreground=True
)

# Save or visualize results
canvas.save("resized_centered_image.png")
mask_canvas.save("foreground_mask.png")