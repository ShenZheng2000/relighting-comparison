import cv2
import numpy as np

# Load RGB image and two masks (assume they are grayscale)
image = cv2.imread('/home/shenzhen/Relight_Projects/personalize-anything/data/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/bdy_1.png')
mask1 = cv2.imread('/home/shenzhen/Relight_Projects/personalize-anything/data/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/pre_processing/black_fg_mask.png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('/home/shenzhen/Relight_Projects/personalize-anything/data/dataset_with_garment_debug_100/09WOMEN_WOMEN_BLOUSE_167/pre_processing/black_fg_mask_groundedsam2.png', cv2.IMREAD_GRAYSCALE)

def overlay_mask(rgb, mask, color=(0, 0, 255), alpha=0.5):
    """Overlay black-foreground-white-background mask on RGB image with a color."""
    overlay = rgb.copy()
    # Foreground = where mask is black (value close to 0)
    foreground = mask < 128
    color_layer = np.zeros_like(rgb)
    color_layer[:] = color
    overlay[foreground] = cv2.addWeighted(rgb[foreground], 1 - alpha, color_layer[foreground], alpha, 0)
    return overlay

# Overlay each mask
overlay1 = overlay_mask(image, mask1)
overlay2 = overlay_mask(image, mask2)

# Concatenate side by side
result = np.concatenate((overlay1, overlay2), axis=1)

# Save or show
cv2.imwrite('vis_mask_compare.png', result)
# or for display (in notebook or local script):
# cv2.imshow('Overlay Comparison', result)
# cv2.waitKey(0); cv2.destroyAllWindows()
