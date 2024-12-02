import itk
import os
import itk.itkNotImageFilterPython
import numpy as np
import random

def read_masks_and_image(binary_mask_path: str, colored_segmentation_path: str, original_image_path: str):
    """
    Reads a binary mask, color segmentation mask, and original image using ITK.
    
    Parameters:
        binary_mask_path (str): Path to the binary mask file.
        color_mask_path (str): Path to the color segmentation mask file.
        original_image_path (str): Path to the original image file (e.g., PNG).
    
    Returns:
        tuple: A tuple containing the binary mask, color mask, and original image.
    """
    # Read the binary mask (scalar image)
    binary_mask = itk.imread(binary_mask_path, itk.UC)  # Unsigned Char for binary mask

    # Read the color segmentation mask (RGB image)
    colored_segmentation = itk.imread(colored_segmentation_path, itk.RGBPixel[itk.UC])  # RGB image

    # Read the original image (e.g., PNG)
    original_image = itk.imread(original_image_path, itk.RGBPixel[itk.UC])  # RGB image

    return binary_mask, colored_segmentation, original_image

# WORKS
def apply_blur(image):
    copy = itk.image_duplicator(image)
    components = itk.VectorIndexSelectionCastImageFilter.New(copy)
    blurred_channels = []

    for i in range(3):  # RGB channels
        components.SetIndex(i)
        components.Update()

        # Apply Gaussian blurring on each channel
        channel = components.GetOutput()
        blur_filter = itk.SmoothingRecursiveGaussianImageFilter.New(channel)
        blur_filter.SetSigma(2.0)  # Set Gaussian standard deviation
        blur_filter.Update()
        blurred_channels.append(blur_filter.GetOutput())

    # Merge blurred channels back into RGB
    join_filter = itk.ComposeImageFilter.New(*blurred_channels)
    join_filter.Update()

    # Save the output
    blurred = join_filter.GetOutput()
    itk.imwrite(blurred, 'blur.png')
    return blurred

# WORKS
def apply_color_jitter(image, brightness_range=(0.5, 1.5), contrast_range=(0.5, 2), saturation_range=(0.5, 2)):
    # Get the size of the image
    size = image.GetLargestPossibleRegion().GetSize()
    height, width = size[1], size[0]

    # Clone the input image to create an output image
    output_image = itk.image_duplicator(image)

    # Generate random factors
    brightness_factor = random.uniform(*brightness_range)
    contrast_factor = random.uniform(*contrast_range)
    saturation_factor = random.uniform(*saturation_range)

    # Iterate through the image pixels
    for y in range(height):
        for x in range(width):
            # Get the RGB pixel at the current position
            rgb_pixel = image.GetPixel([x, y])
            r, g, b = rgb_pixel

            # Calculate grayscale value
            gray = (r + g + b) / 3.0

            # Apply brightness adjustment
            r = r * brightness_factor
            g = g * brightness_factor
            b = b * brightness_factor

            # Apply contrast adjustment
            r = gray + contrast_factor * (r - gray)
            g = gray + contrast_factor * (g - gray)
            b = gray + contrast_factor * (b - gray)

            # Apply saturation adjustment
            r = gray + saturation_factor * (r - gray)
            g = gray + saturation_factor * (g - gray)
            b = gray + saturation_factor * (b - gray)

            # Clip values to valid range (0-255 for 8-bit images)
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))

            # Set the adjusted pixel value in the output image
            output_image.SetPixel([x, y], itk.RGBPixel[itk.UC]([r, g, b]))

    itk.imwrite(output_image, 'jittered.png')
    return output_image

# DOESN'T WORK
def apply_random_perspective(image, distortion_scale=0.5):
    # Get image size and spacing
    size = image.GetLargestPossibleRegion().GetSize()
    spacing = image.GetSpacing()

    # Define source and destination points
    source_points = [
        [0, 0],
        [size[0] - 1, 0],
        [0, size[1] - 1],
        [size[0] - 1, size[1] - 1]
    ]

    destination_points = []
    for point in source_points:
        dest_point = [
            point[0] + random.uniform(-distortion_scale, distortion_scale) * size[0],
            point[1] + random.uniform(-distortion_scale, distortion_scale) * size[1],
        ]
        destination_points.append(dest_point)

    # Convert to ITK PointSet
    point_set_type = itk.PointSet[itk.D, 2]
    source_point_set = point_set_type.New()
    destination_point_set = point_set_type.New()

    for i, (source, dest) in enumerate(zip(source_points, destination_points)):
        source_point = itk.Point[itk.D, 2](source)
        destination_point = itk.Point[itk.D, 2](dest)
        source_point_set.SetPoint(i, source_point)
        destination_point_set.SetPoint(i, destination_point)

    # Create the perspective transform
    transform = itk.ThinPlateSplineKernelTransform[itk.D, 2].New()
    transform.SetSourceLandmarks(source_point_set)
    transform.SetTargetLandmarks(destination_point_set)
    transform.ComputeWMatrix()

    # Resample the image with the transform
    resample_filter = itk.ResampleImageFilter.New(image)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputSpacing(spacing)
    resample_filter.SetSize(size)
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.Update()

    return resample_filter.GetOutput()
    
# DOESN'T WORK
def apply_random_adjust_sharpness(image, min_factor=0.5, max_factor=2.0):
        # Convert the image to grayscale for sharpening
        # Randomize the sharpness adjustment factor
    sharpness_factor = random.uniform(min_factor, max_factor)

    # Convert RGB image to grayscale for edge detection
    grayscale_image = itk.rgb_to_luminance_image_filter(image)

    # Apply Laplacian filter for edge enhancement
    laplacian_filter = itk.LaplacianImageFilter.New(Input=grayscale_image)
    laplacian_filter.Update()
    laplacian_image = laplacian_filter.GetOutput()

    # Convert Laplacian image to RGB for blending
    laplacian_rgb = itk.cast_image_filter(laplacian_image, ttype=(type(laplacian_image), type(image)))

    # Blend the original and sharpness-enhanced image
    adjust_filter = itk.AddImageFilter.New(image, laplacian_rgb)
    adjust_filter.SetConstant2(sharpness_factor - 1)
    adjust_filter.Update()
    adjusted_image = adjust_filter.GetOutput()

    return adjusted_image

# inverts the given mask
def inverse_of_mask(binary_mask):
    itk.imwrite(binary_mask, 'mask.png')
    copy = itk.image_duplicator(binary_mask)
    inverted_arr = itk.array_from_image(copy)
    for i in range(len(inverted_arr)):
        for j in range(len(inverted_arr[0])):
            if inverted_arr[i][j] > 127:
                inverted_arr[i][j] = 0
            else:
                inverted_arr[i][j] = 255
    inverted_image = itk.image_from_array(inverted_arr)
    itk.imwrite(inverted_image, 'inverted.png') 
    return inverted_image

# gets the non region of interest
def get_nonroi(mask_inverse, original_image):
    nonroi = itk.image_duplicator(original_image)
    inverse_arr = itk.array_from_image(mask_inverse)
    for i in range(len(inverse_arr)):
        for j in range(len(inverse_arr[0])):
            if inverse_arr[i][j] == 0:
                nonroi.SetPixel((j, i), itk.RGBPixel[itk.UC]([0, 0, 0]))
    itk.imwrite(nonroi, 'nonroi.png')
    itk.imwrite(original_image, 'original.png')
    return nonroi

# combines the augmented roi with the non roi
def combine(roi_augmented, nonroi):
    combined = itk.image_duplicator(nonroi)
    # kinda lazy so i hardcoded
    for i in range(512):
        for j in range(512):
            if nonroi.GetPixel((j, i)) == itk.RGBPixel[itk.UC]([0, 0, 0]):
                combined.SetPixel((j, i), roi_augmented.GetPixel((j, i)))
    itk.imwrite(combined, 'combined.png')
    return combined
    

# this is the ultimate function that will take the mask, segmentation, and original image to produce the augmented version
# it does so by determining the non-roi (using the mask) and then combining this with the augmented segmentation
# it looks weird because the mask does not align with the segmentation
def apply_augmentations(binary_mask, segmentation, original_image, augmentations):
    res = itk.image_duplicator(segmentation)
    # inverse_mask = inverse_of_mask(binary_mask)
    # nonroi = get_nonroi(inverse_mask, original_image)
    for augmentation in augmentations:
        res = augmentation(res)

    # combined = combine(res, nonroi)
    # return combined    


# Main Execution

augmentations = [
    # lambda img: apply_blur(img),
    # lambda img: apply_color_jitter(img, brightness_range=(0.5, 1.5), contrast_range=(0.5, 2), saturation_range=(0.5, 2)),
    lambda img: apply_random_perspective(img, distortion_scale=0.5), # TODO
    # lambda img: apply_random_adjust_sharpness(img, min_factor=0.5, max_factor=2.0)
]

frame = './data/test/1/VUMC040dust.MP4_frame312/'
orig = ''
for file_name in os.listdir(frame):
    if not (file_name.startswith("mask_") or file_name.startswith("color_")):
        orig = file_name
binary_mask_path = frame + 'mask_' + orig
segmentation_path = frame + 'color_' + orig
original_image_path = frame + orig

binary_mask, segmentation, original_image = read_masks_and_image(
    binary_mask_path, segmentation_path, original_image_path
)

# inverse_mask = inverse_of_mask(binary_mask)
# nonroi = get_nonroi(inverse_mask, original_image)
# combined = combine(segmentation, nonroi)


augmented = apply_augmentations(binary_mask, segmentation, original_image, augmentations)

# itk.imwrite(original_image, 'original.png')
# itk.imwrite(augmented, 'augmented.png')