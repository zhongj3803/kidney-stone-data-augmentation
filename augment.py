# import itk
# import cv2
# import numpy as np
# import random


# def read_masks_and_image(binary_mask_path, color_mask_path, original_image_path):
#     """
#     Reads a binary mask, a color mask, and the original image from the specified file paths using ITK.
    
#     Args:
#         binary_mask_path (str): Path to the binary mask image file.
#         color_mask_path (str): Path to the color mask image file.
#         original_image_path (str): Path to the original image file.

#     Returns:
#         tuple: A tuple containing the binary mask, color mask, and original image.
#                Binary mask is a 2D NumPy array with values 0 and 255.
#                Color mask is a 3D NumPy array (RGB).
#                Original image is a 3D NumPy array (RGB).
#     """
#     # Read binary mask
#     binary_mask = itk.imread(binary_mask_path, itk.UC)
#     binary_mask_np = itk.GetArrayFromImage(binary_mask)

#     # Ensure binary mask values are only 0 and 255
#     binary_mask_np = np.where(binary_mask_np > 127, 255, 0).astype(np.uint8)

#     # Read color mask
#     color_mask = itk.imread(color_mask_path, itk.RGBPixel[itk.UC])
#     color_mask_np = itk.GetArrayFromImage(color_mask)

#     # Read original image
#     original_image = itk.imread(original_image_path, itk.RGBPixel[itk.UC])
#     original_image_np = itk.GetArrayFromImage(original_image)

#     return binary_mask_np, color_mask_np, original_image_np


# def apply_blur(image_np, kernel_size=7):
#     """
#     Applies medium-heavy blur to an image using ITK.
#     Args:
#         image_np (numpy array): Input image as a NumPy array.
#         kernel_size (int): Size of the kernel used for blurring.
#     Returns:
#         numpy array: Blurred image.
#     """
#     print(image_np.shape)
#     image = itk.GetImageFromArray(image_np)
#     blur_filter = itk.SmoothingRecursiveGaussianImageFilter.New(image, Sigma=kernel_size / 2.0)
#     blur_filter.Update()
#     blurred_image_np = itk.GetArrayFromImage(blur_filter.GetOutput())
#     return blurred_image_np



# def apply_color_jitter(image_np, brightness=0.3, contrast=0.3, saturation=0.3):
#     """
#     Applies color jitter to an image using ITK.
#     Args:
#         image_np (numpy array): Input image as a NumPy array (RGB).
#         brightness (float): Maximum change in brightness.
#         contrast (float): Maximum change in contrast.
#         saturation (float): Maximum change in saturation.
#     Returns:
#         numpy array: Color-jittered image.
#     """
#     # Brightness adjustment
#     brightness_factor = 1 + random.uniform(-brightness, brightness)
#     jittered_image = np.clip(image_np * brightness_factor, 0, 255).astype(np.uint8)

#     # Contrast adjustment
#     contrast_factor = 1 + random.uniform(-contrast, contrast)
#     jittered_image = np.clip((jittered_image - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)

#     # Saturation adjustment (modify color intensity)
#     saturation_factor = 1 + random.uniform(-saturation, saturation)
#     hsv_image = itk.GetArrayFromImage(itk.RGBToLuminanceImageFilter.New(itk.GetImageFromArray(jittered_image)).GetOutput())
#     jittered_image = np.clip(jittered_image * saturation_factor + hsv_image[:, :, None] * (1 - saturation_factor), 0, 255).astype(np.uint8)

#     return jittered_image


# def apply_random_perspective(image_np, max_warp=0.1):
#     """
#     Applies a random perspective transformation using ITK.
#     Args:
#         image_np (numpy array): Input image as a NumPy array (RGB).
#         max_warp (float): Maximum percentage of image dimensions to warp.
#     Returns:
#         numpy array: Perspective-transformed image.
#     """
#     height, width = image_np.shape[:2]
#     delta = max_warp * min(width, height)

#     # Define warp points
#     src_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
#     dst_points = src_points + np.random.uniform(-delta, delta, src_points.shape).astype(np.float32)

#     # Compute transform matrix
#     transform = itk.GetArrayFromImage(
#         itk.PerspectiveTransform[itk.D].New().SetSourceLandmarks(itk.PointsList(src_points))
#         .SetTargetLandmarks(itk.PointsList(dst_points))
#         .Compute()
#     )
#     image = itk.GetImageFromArray(image_np)
#     warped_image = itk.ResampleImageFilter.New(image, Transform=transform).GetOutput()

#     return itk.GetArrayFromImage(warped_image)


# def apply_random_adjust_sharpness(image_np, alpha=1.5):
#     """
#     Randomly adjusts sharpness of an image using ITK.
#     Args:
#         image_np (numpy array): Input image as a NumPy array (RGB).
#         alpha (float): Sharpening factor. >1 increases sharpness.
#     Returns:
#         numpy array: Sharpness-adjusted image.
#     """
#     image = itk.GetImageFromArray(image_np)

#     # Sharpening kernel
#     kernel = itk.Array2D.D.New([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpen_filter = itk.ConvolutionImageFilter.New(image, KernelImage=kernel)
#     sharpen_filter.Update()

#     sharp_image_np = itk.GetArrayFromImage(sharpen_filter.GetOutput())

#     # Blend the original and sharpened image
#     blended_image_np = np.clip(image_np * (1 - alpha) + sharp_image_np * alpha, 0, 255).astype(np.uint8)

#     return blended_image_np



# def apply_augmentations(image, augmentations):
#     """
#     Applies a series of augmentations to an image.

#     Args:
#         image (numpy array): Input image.
#         augmentations (list): List of augmentation functions to apply.
#         apply_all (bool): If True, applies all augmentations sequentially.
#                           If False, applies a random subset of augmentations.

#     Returns:
#         numpy array: Augmented image.
#     """
#     augmented_image = image.copy()
#     for augmentation in augmentations:
#         augmented_image = augmentation(augmented_image)

#     return augmented_image

# def apply_augmentations_to_binary_image(original_image, binary_mask, augmentations):
#     """
#     Applies augmentations to regions defined by a binary mask in the original image.
    
#     Args:
#         original_image (numpy array): Original image (RGB).
#         binary_mask (numpy array): Binary mask (0 and 255).
#         augmentations (list): List of augmentation functions to apply.

#     Returns:
#         numpy array: Final image with augmented regions combined with unaltered regions.
#     """
#     # Ensure binary_mask is in the correct shape for RGB
#     binary_mask_3d = np.stack([binary_mask] * 3, axis=-1)  # Convert to 3D for RGB operations

#     # Create the masked (colored) image
#     colored_image = (original_image * (binary_mask_3d / 255)).astype(np.uint8)

#     # Apply augmentations to the colored image
#     augmented_colored_image = colored_image.copy()
#     for augmentation in augmentations:
#         augmented_colored_image = augmentation(augmented_colored_image)

#     # Invert the binary mask for non-augmented regions
#     inverted_mask = 255 - binary_mask_3d

#     # Combine the augmented regions with the unaltered regions from the original image
#     final_image = ((original_image * (inverted_mask / 255)) +
#                    (augmented_colored_image * (binary_mask_3d / 255))).astype(np.uint8)

#     return final_image

# def apply_augmentations_to_colored_segmented_image(original_image, segmented_image, augmentations):
#     """
#     Applies augmentations to the entire segmented image and combines the augmented regions 
#     with unaltered regions from the original image.
    
#     Args:
#         original_image (numpy array): Original image (RGB).
#         segmented_image (numpy array): Colored segmented image (RGB).
#         augmentations (list): List of augmentation functions to apply.

#     Returns:
#         numpy array: Final image with augmented regions combined with unaltered regions.
#     """
#     # Apply augmentations to the entire segmented image
#     augmented_segmented_image = segmented_image.copy()
#     for augmentation in augmentations:
#         augmented_segmented_image = augmentation(augmented_segmented_image)

#     # Create a binary mask from the segmented image (non-zero pixels are part of the segment)
#     binary_mask = np.any(segmented_image > 0, axis=-1).astype(np.uint8) * 255
#     binary_mask_3d = np.stack([binary_mask] * 3, axis=-1)  # Convert to 3D for RGB operations

#     # Invert the binary mask for non-augmented regions
#     inverted_mask = 255 - binary_mask_3d

#     # Combine the augmented regions with the unaltered regions from the original image
#     final_image = ((original_image * (inverted_mask / 255)) +
#                    (augmented_segmented_image * (binary_mask_3d / 255))).astype(np.uint8)

#     return final_image




# # List of augmentation functions
# augmentations = [
#     lambda img: apply_blur(img, kernel_size=7),
#     lambda img: apply_color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.3),
#     lambda img: apply_random_perspective(img, max_warp=0.1),
#     lambda img: apply_random_adjust_sharpness(img, alpha=1.5)
# ]

# # Read masks and original image
# binary_mask_np, color_mask_np, original_image_np = read_masks_and_image(
#     binary_mask_path='./data/test/1/VUMC040dust.MP4_frame312/mask_VUMC040dust.MP4_frame312.jpg',
#     color_mask_path='./data/test/1/VUMC040dust.MP4_frame312/color_VUMC040dust.MP4_frame312.jpg',
#     original_image_path='./data/test/1/VUMC040dust.MP4_frame312/VUMC040dust.MP4_frame312.jpg'
# )

# binary_augmented_image = apply_augmentations_to_binary_image(original_image_np, binary_mask_np, augmentations)
# color_augmented_image = apply_augmentations_to_colored_segmented_image(original_image_np, color_mask_np, augmentations)
# itk.imwrite(itk.GetImageFromArray(binary_augmented_image), 'thing.png')
# itk.imwrite(itk.GetImageFromArray(color_augmented_image), 'thing1.png')


import cv2
import numpy as np
import random
import os

def read_masks_and_image(binary_mask_path, colored_segmented_image_path, original_image_path):
    """
    Loads a binary mask and a colored segmented image.

    Args:
        binary_mask_path (str): Path to the binary mask image.
        colored_segmented_image_path (str): Path to the colored segmented image.

    Returns:
        tuple: (binary_mask, colored_segmented_image)
            - binary_mask (np.ndarray): Binary mask as a 2D numpy array.
            - colored_segmented_image (np.ndarray): Colored segmented image as a 3D numpy array in RGB format.
    """
    # Read the binary mask as a grayscale image
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    if binary_mask is None:
        raise FileNotFoundError(f"Failed to load binary mask from {binary_mask_path}")
    
    # Threshold to ensure binary (values 0 or 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Read the colored segmented image as a color image
    colored_segmented_image = cv2.imread(colored_segmented_image_path, cv2.IMREAD_COLOR)
    if colored_segmented_image is None:
        raise FileNotFoundError(f"Failed to load colored segmented image from {colored_segmented_image_path}")
    
    # Convert to RGB (from OpenCV's default BGR format)
    colored_segmented_image = cv2.cvtColor(colored_segmented_image, cv2.COLOR_BGR2RGB)

    # Read the original image as a color image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"Failed to load original image from {original_image_path}")
    
    # Convert the original image to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    return binary_mask, colored_segmented_image, original_image


def apply_blur(image_np, kernel_size=7):
    """Applies Gaussian blur to an image using OpenCV."""
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)


def apply_color_jitter(image_np, brightness=0.3, contrast=0.3, saturation=0.3):
    """
    Applies color jitter to the input image by adjusting brightness, contrast, and saturation.

    Args:
        image_np (np.ndarray): Input image in RGB format.
        brightness (float): Max adjustment factor for brightness (default: 0.3).
        contrast (float): Max adjustment factor for contrast (default: 0.3).
        saturation (float): Max adjustment factor for saturation (default: 0.3).

    Returns:
        np.ndarray: Color-jittered image in RGB format.
    """
    # Convert image to float32 for processing
    image = image_np.astype(np.float32) / 255.0

    # Random brightness adjustment
    brightness_factor = 1 + random.uniform(-brightness, brightness)
    image = np.clip(image * brightness_factor, 0, 1)

    # Random contrast adjustment
    contrast_factor = 1 + random.uniform(-contrast, contrast)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = np.clip((image - mean) * contrast_factor + mean, 0, 1)

    # Random saturation adjustment
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + random.uniform(-saturation, saturation)), 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Convert back to uint8
    return (image * 255).astype(np.uint8)

def apply_random_perspective(image_np, max_warp=0.1):
    """
    Applies a random perspective transformation to the input image.

    Args:
        image_np (np.ndarray): Input image in RGB format.
        max_warp (float): Maximum warp factor as a fraction of image dimensions (default: 0.1).

    Returns:
        np.ndarray: Image with random perspective transformation applied.
    """
    # Get image dimensions
    h, w, _ = image_np.shape

    # Define random shifts for corners
    warp_x = int(max_warp * w)
    warp_y = int(max_warp * h)
    src_points = np.float32([
        [random.randint(0, warp_x), random.randint(0, warp_y)],  # Top-left
        [w - random.randint(0, warp_x), random.randint(0, warp_y)],  # Top-right
        [random.randint(0, warp_x), h - random.randint(0, warp_y)],  # Bottom-left
        [w - random.randint(0, warp_x), h - random.randint(0, warp_y)]  # Bottom-right
    ])
    dst_points = np.float32([
        [0, 0],  # Top-left
        [w, 0],  # Top-right
        [0, h],  # Bottom-left
        [w, h]   # Bottom-right
    ])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective warp
    warped_image = cv2.warpPerspective(image_np, matrix, (w, h))

    return warped_image


def apply_random_adjust_sharpness(image_np, alpha=1.5):
    """
    Applies random sharpness adjustment to the input image by blending the original and a sharpened version.

    Args:
        image_np (np.ndarray): Input image in RGB format.
        alpha (float): Sharpness blending factor. Higher values increase sharpness (default: 1.5).

    Returns:
        np.ndarray: Image with adjusted sharpness.
    """
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Apply the sharpening kernel to create a sharpened version of the image
    sharpened_image = cv2.filter2D(image_np, -1, kernel)

    # Blend the original and sharpened image using alpha
    blended_image = cv2.addWeighted(image_np, 1 - alpha, sharpened_image, alpha, 0)

    # Ensure the resulting image is within valid pixel range
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    return blended_image


def apply_augmentations(image, augmentations):
    """Applies a series of augmentations to an image."""
    augmented_image = image.copy()
    for augmentation in augmentations:
        augmented_image = augmentation(augmented_image)
    return augmented_image


def apply_augmentations_to_binary_image(original_image, binary_mask, augmentations):
    """Applies augmentations to regions defined by a binary mask in the original image."""
    binary_mask_3d = np.stack([binary_mask] * 3, axis=-1)
    colored_image = (original_image * (binary_mask_3d / 255)).astype(np.uint8)
    augmented_colored_image = apply_augmentations(colored_image, augmentations)
    inverted_mask = 255 - binary_mask_3d
    return ((original_image * (inverted_mask / 255)) +
            (augmented_colored_image * (binary_mask_3d / 255))).astype(np.uint8)


def apply_augmentations_to_colored_segmented_image(original_image, segmented_image, augmentations):
    """Applies augmentations to the entire colored segmented image."""
    augmented_segmented_image = apply_augmentations(segmented_image, augmentations)
    binary_mask = np.any(segmented_image > 0, axis=-1).astype(np.uint8) * 255
    binary_mask_3d = np.stack([binary_mask] * 3, axis=-1)
    inverted_mask = 255 - binary_mask_3d
    return ((original_image * (inverted_mask / 255)) +
            (augmented_segmented_image * (binary_mask_3d / 255))).astype(np.uint8)


# Main Execution

frame = './data/test/1/VUMC040dust.MP4_frame312/'
orig = ''
for file_name in os.listdir(frame):
    if not (file_name.startswith("mask_") or file_name.startswith("color_")):
        orig = file_name
binary_mask_path = frame + 'mask_' + orig
color_mask_path = frame + 'color_' + orig
original_image_path = frame + orig

binary_mask_np, color_mask_np, original_image_np = read_masks_and_image(
    binary_mask_path, color_mask_path, original_image_path
)

augmentations = [
    lambda img: apply_blur(img, kernel_size=7),
    lambda img: apply_color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.3),
    lambda img: apply_random_perspective(img, max_warp=0.1),
    lambda img: apply_random_adjust_sharpness(img, alpha=1.5)
]

binary_augmented_image = apply_augmentations_to_binary_image(original_image_np, binary_mask_np, augmentations)
color_augmented_image = apply_augmentations_to_colored_segmented_image(original_image_np, color_mask_np, augmentations)

# Save results
cv2.imwrite('original.png', cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))
cv2.imwrite('binary_augmented_image.png', cv2.cvtColor(binary_augmented_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('color_augmented_image.png', cv2.cvtColor(color_augmented_image, cv2.COLOR_RGB2BGR))
