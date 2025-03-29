from PIL import Image

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