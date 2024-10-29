import os
import cv2
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision
import matplotlib.pyplot as plt
from rembg import remove
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import math
import ffmpeg


def count_items_in_folder(folder_path):
    """
    Count the number of items in a folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        int: The number of items in the folder.

    Raises:
        FileNotFoundError: If the folder does not exist.
        PermissionError: If the user does not have permission to access the folder.
    """
    try:
        items = os.listdir(folder_path)
        return len(items)
    except FileNotFoundError:
        return "Folder not found"
    except PermissionError:
        return "Permission denied"


def split_folder_by_percentage(input_folder, test_dir, train_dir, percentage):
    """
    Splits a folder into two sub-folders based on a percentage.

    Args:
        input_folder (str): The path to the input folder.
        test_dir (str): The path to the test folder.
        train_dir (str): The path to the train folder.
        percentage (float): The percentage of items to put in the test folder.

    Returns:
        None

    Raises:
        None
    """
    # Ensure the output folders exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Get a list of all files in the input folder
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Calculate the split index
    split_index = int(len(files) * percentage / 100)
    if len(files) == 0:
        return

    # Split the files
    test = files[:split_index]
    train = files[split_index:]

    # Move the 10% files to the 10% folder
    for file in test:
        src_file = os.path.join(input_folder, file)
        dst_file = os.path.join(test_dir, file)
        shutil.move(src_file, dst_file)

    # Move the 90% files to the 90% folder
    for file in train:
        src_file = os.path.join(input_folder, file)
        dst_file = os.path.join(train_dir, file)
        shutil.move(src_file, dst_file)


def resize_images(input_folder, size=(1024, 1024)):
    """
    Resizes all images in the specified input folder to the given size and saves them as PNG files.

    Args:
        input_folder (str): The path to the folder containing the images.
        size (tuple, optional): The size to resize the images to. Defaults to (1024, 1024).

    Returns:
        None

    Raises:
        Exception: If there is an error resizing an image.

    """
    # List files in input folder
    files = os.listdir(input_folder)

    for file in files:
        # Check if file is an image
        if file.lower().endswith(('jpg', 'jpeg', 'gif', 'png', 'jfif', 'webp')):
            try:
                # Open image
                img = Image.open(os.path.join(input_folder, file))
                # Resize image
                img_resized = img.resize(size)
                # Save resized image as PNG
                output_filename = os.path.splitext(file)[0] + ".png"
                img_resized.save(os.path.join(input_folder, output_filename))

                # Remove original file only if it was not already a PNG
                if not file.lower().endswith('.png'):
                    os.remove(os.path.join(input_folder, file))

                print(f"Resized {file} successfully and replaced with {output_filename}.")
            except Exception as e:
                print(f"Error resizing {file}: {e}")
        else:
            print(f"{file} is not an image file (jpg, jpeg, png, gif), skipping.")


def apply_projective_transformation(image_path, src_points, dst_points):
    """
    Apply a projective transformation to an image.

    Args:
        image_path (str): The path to the image file.
        src_points (List[List[float]]): The source points for the transformation.
        dst_points (List[List[float]]): The destination points for the transformation.

    Returns:
        numpy.ndarray: The transformed image.

    Raises:
        ValueError: If the image is not found or the path is incorrect.

    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert points to numpy arrays
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Compute the transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Get image dimensions
    h, w = image.shape[:2]

    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(image, matrix, (w, h))

    return transformed_image


def ensure_folder_exists(folder_path):
    """
    Ensure that a folder exists. If it doesn't exist, create it.

    Args:
        folder_path (str): The path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def calc_iou(new_mask, comparison_dataset):
    """
    Calculate the Intersection over Union (IoU) score between a new mask and a set of comparison masks.

    Args:
        new_mask (numpy.ndarray): The new mask to compare.
        comparison_dataset (list of numpy.ndarray): The set of comparison masks.

    Returns:
        float: The IoU score, which is the average of the intersection over union areas.
    """
    iou_score = 0
    for j, mask in enumerate(comparison_dataset):
        intersection = cv2.bitwise_and(new_mask, mask)
        union = cv2.bitwise_or(new_mask, mask)
        intersection_area = np.sum(intersection > 0)
        union_area = np.sum(union > 0)
        iou_score += intersection_area / union_area if union_area != 0 else 0
    iou = iou_score/(j+1)
    return iou


def create_shape(image, mask, color, shape_type, roundness=0.0, increase_factor=1.1, border_padding=5):
    """
    Creates a shape based on the given image, mask, color, and shape type.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The binary mask of the shape.
        color (List[int]): The color of the shape.
        shape_type (str): The type of the shape. Can be 'circle', 'oval', 'octagon', or 'triangle'.
        roundness (float, optional): The roundness of the shape. Defaults to 0.0.
        increase_factor (float, optional): The factor to increase the shape. Defaults to 1.1.
        border_padding (int, optional): The padding of the shape from the border. Defaults to 5.

    Returns:
        np.ndarray: The colored shape.

    Notes:
        - If the contour has less than 5 points, it prints "Contour has less than 5 points, cannot fit ellipse." and returns a zero-filled image.
        - If no contours are found, it prints "No contours found." and returns a zero-filled image.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return np.zeros_like(image)

    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 5:  # Minimum 5 points required for fitEllipse
        print("Contour has less than 5 points, cannot fit ellipse.")
        return np.zeros_like(image)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        cx, cy = contour[0][0]

    height, width = image.shape[:2]

    if shape_type in ['circle', 'oval']:
        # Fit an ellipse to the contour
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse

        # Calculate aspect ratio
        aspect_ratio = min(axes) / max(axes)

        if shape_type == 'circle' or (shape_type == 'oval' and aspect_ratio > 0.9):
            # Use circular shape
            radius = max(axes) / 2
            radius = int(radius * increase_factor)
            max_radius = min(cx, cy, width - cx, height - cy) - border_padding
            radius = min(radius, max_radius)
            shape_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(shape_mask, (int(center[0]), int(center[1])), radius, 255, thickness=-1)
        else:
            # Use oval shape
            major_axis = max(axes) * increase_factor / 2
            minor_axis = min(axes) * increase_factor / 2

            # Adjust axes to ensure they don't touch the border
            max_axis = min(cx, cy, width - cx, height - cy) - border_padding
            major_axis = min(major_axis, max_axis)
            minor_axis = min(minor_axis, max_axis)

            shape_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # Correct the angle to match the original orientation
            corrected_angle = -angle  # Negate the angle to correct mirroring

            cv2.ellipse(shape_mask, (int(center[0]), int(center[1])), (int(major_axis), int(minor_axis)),
                        corrected_angle, 0, 360, 255, thickness=-1)

    else:
        # Code for octagon and triangle remains the same
        num_sides = 8 if shape_type == 'octagon' else 3
        for epsilon in np.linspace(0.01, 0.1, 100):
            approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
            if len(approx) == num_sides:
                break

        scaled_approx = np.array([[[int((x[0][0] - cx) * increase_factor + cx),
                                    int((x[0][1] - cy) * increase_factor + cy)]] for x in approx])
        scaled_approx = np.clip(scaled_approx, border_padding,
                                [width - border_padding - 1, height - border_padding - 1])

        shape_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        cv2.drawContours(shape_mask, [scaled_approx], -1, 255, thickness=cv2.FILLED)

    if roundness > 0:
        kernel_size = int(min(shape_mask.shape[0], shape_mask.shape[1]) * roundness)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed_shape = cv2.GaussianBlur(shape_mask, (kernel_size, kernel_size), 0)
        _, rounded_shape = cv2.threshold(smoothed_shape, 127, 255, cv2.THRESH_BINARY)
    else:
        rounded_shape = shape_mask

    colored_shape = np.zeros_like(image)
    colored_shape[rounded_shape > 0] = color

    return colored_shape


def process_image(image, color, label):
    """
    Process an image to create a final result with a colored shape on a black background.

    Args:
        image (np.ndarray): The input image.
        color (List[int]): The color of the shape.
        label (str): The label of the shape. Can be 'stop_sign', 'give_way_sign', or 'no_entry_sign'.

    Returns:
        np.ndarray: The final result with the colored shape on a black background.

    Notes:
        - The function creates a binary mask for the non-black color in the input image.
        - It then creates a shape based on the mask and the color.
        - The shape is either an octagon, triangle, or oval, depending on the label.
        - The function also creates a mask for the specific color.
        - Finally, it creates the final result by setting the pixels in the mask to the specified color.

    """
    
    # Define the lower and upper bounds for the color
    lower_bound = np.clip(color, 0, 255)
    upper_bound = np.clip(color, 0, 255)
    # Create mask for the non-black color
    binary_mask = cv2.inRange(image, lower_bound, upper_bound)

    # Create shape
    if label == 'stop_sign':
        shape = create_shape(image, binary_mask, color, 'octagon', increase_factor=1, border_padding=5)  # Octagon
    elif label == 'give_way_sign':
        shape = create_shape(image, binary_mask, color, 'triangle', roundness=0.3, increase_factor=1.2, border_padding=5)
    else:  # no_entry_sign
        shape = create_shape(image, binary_mask, color, 'oval', increase_factor=1, border_padding=5)

    # Create a mask for the specific color (pink for 'give_way_sign')
    lower_color = np.array([color[0]-10, color[1]-10, color[2]-10])
    upper_color = np.array([color[0]+10, color[1]+10, color[2]+10])
    color_mask = cv2.inRange(shape, lower_color, upper_color)

    # Create the final result: colored shape on black background
    final_result = np.zeros_like(image)
    final_result[color_mask > 0] = color

    return final_result


def create_mask(image, traffic_label, live, main_folder_path):
    """
    Create a mask for a specific traffic sign in an image.

    Args:
        image (np.ndarray): The input image.
        traffic_label (str): The label of the traffic sign. Can be 'stop_sign', 'give_way_sign', or 'no_entry_sign'.
        live (bool): Whether the image is live or not.
        main_folder_path (str): The path to the main folder.

    Returns:
        bool: Whether a mask was detected or not.
        np.ndarray: The mask image.

    Notes:
        - The function creates a binary mask for the specific traffic sign in the input image.
        - It uses the traffic label to determine the color of the mask.
        - The function also creates a mask for the specific color.
        - The function applies morphological operations to clear noise in the mask.
        - The function determines which mask has the highest IOU.
        - The function returns the detected mask and the mask image.
    """
    color_mask = {'stop_sign': [255, 255, 0],
                  'give_way_sign': [255, 0, 255],
                  'no_entry_sign': [0, 255, 255]
                  }
    color = color_mask[traffic_label]
    comparison_mask_path = f'{main_folder_path}/comparison_images/{traffic_label}'
    comparison_mask_dataset = [f'{comparison_mask_path}/mask1.png', f'{comparison_mask_path}/mask2.png',
                               f'{comparison_mask_path}/mask3.png', f'{comparison_mask_path}/mask4.png']
    dataset_masks = [cv2.imread(mask, cv2.COLOR_BGR2RGB) for mask in comparison_mask_dataset]
    dataset_masks = [cv2.resize(mask, (image.shape[1], image.shape[0]))for mask in dataset_masks]

    detected_polygon, detect_polygon_mask = detect_polygon(image, color)
    if detected_polygon:
        # plt.imshow(detect_polygon_mask)
        # plt.title('detect_polygon_mask')
        # plt.show()
        detect_polygon_iou = calc_iou(detect_polygon_mask, dataset_masks)
        if detect_polygon_iou < 0.19:
            detected_polygon = False
    detected_rembg, rembg_mask = rembg_mask_create(image, color)
    if detected_rembg:
        # plt.imshow(rembg_mask)
        # plt.title('rembg_mask')
        # plt.show()
        rembg_iou = calc_iou(rembg_mask, dataset_masks)
        if rembg_iou < 0.1:
            detected_rembg = False
    detected = True
    if detected_polygon and detected_rembg is True:

        combined_mask = cv2.bitwise_or(detect_polygon_mask, rembg_mask)
        # Apply morphological operations to clear noise
        kernel1 = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((9, 9), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel2)
        # plt.imshow(combined_mask)
        # plt.title('combined_mask')
        # plt.show()
        combined_iou = calc_iou(combined_mask, dataset_masks)

        # Determine which mask has the highest IOU
        iou_values = {'detect_polygon_mask': detect_polygon_iou, 'rembg_mask': rembg_iou, 'combined_mask': combined_iou}
        chosen_mask_name = max(iou_values, key=iou_values.get)
        chosen_mask = eval(chosen_mask_name)
        # plt.imshow(chosen_mask)
        # plt.title('chosen_mask')
        # plt.show()

        process_mask = process_image(chosen_mask, color, traffic_label)
        process_mask_iou = calc_iou(process_mask, dataset_masks)
        if process_mask_iou >= max(iou_values.values()) or max(iou_values.values()) - process_mask_iou < 0.025:
            final_mask = process_mask
            # print(f'chosen mask is {chosen_mask_name} after process')
        else:
            final_mask = chosen_mask
            # print(f'chosen mask is {chosen_mask_name}')

    elif detected_polygon is False and detected_rembg is True:
        chosen_mask = rembg_mask
        # plt.imshow(chosen_mask)
        # plt.title('chosen_mask')
        # plt.show()

        process_mask = process_image(chosen_mask, color, traffic_label)

        process_mask_iou = calc_iou(process_mask, dataset_masks)
        if process_mask_iou >= rembg_iou or rembg_iou - process_mask_iou < 0.025:
            final_mask = process_mask
            # print(f'chosen mask is rembg_mask after process (polygon has failed)')
        else:
            final_mask = chosen_mask
            # print(f'chosen mask is rembg_mask (polygon has failed)')

    elif detected_rembg is False and detected_polygon is True:
        chosen_mask = detect_polygon_mask
        # plt.imshow(chosen_mask)
        # plt.title('chosen_mask')
        # plt.show()

        process_mask = process_image(chosen_mask, color, traffic_label)
        process_mask_iou = calc_iou(process_mask, dataset_masks)
        if process_mask_iou >= detect_polygon_iou or detect_polygon_iou - process_mask_iou < 0.025:
            final_mask = process_mask
            # print(f'chosen mask is detect_polygon_mask after process (rembg has failed)')
        else:
            final_mask = chosen_mask
            # print(f'chosen mask is detect_polygon_mask (rembg has failed)')

    else:
        # print(f'rembg and polygon have failed')
        return False, None, None

    if live is True:
        masks = np.any(final_mask != [0, 0, 0], axis=-1)
        image_with_mask = image.copy()
        image_with_mask[masks] = color
        return detected, final_mask, image_with_mask

    else:
        background = np.all(final_mask == [0, 0, 0], axis=-1)
        image_black_back = image.copy()
        image_black_back[background] = [0, 0, 0]
        return detected, final_mask, image_black_back


def rembg_mask_create(img, color):
    """
    Create a mask for a specific color in an image using the `rembg` library.

    Args:
        img (np.ndarray): The input image.
        color (List[int]): The RGB color to create the mask for.

    Returns:
        Tuple[bool, np.ndarray]: A tuple containing a boolean indicating if a mask was detected and the mask image.

    Notes:
        - The function converts the input image to the RGB color format.
        - It uses the `rembg` library to create a binary mask for the specific color.
        - The function applies morphological operations to clear noise in the mask.
        - The function determines the area of the mask and checks if it is greater than a threshold.
        - If the area is greater than the threshold, the function sets the pixels in the mask to the specified color.

    """
    detected = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB FORMAT
    mask = remove(img, only_mask=True, post_process_mask=True)
    if mask.ndim == 2:
        mask = np.stack((mask,) * 3, axis=-1)
        mask[np.any(mask != [0, 0, 0], axis=-1)] = color

    # Apply morphological operations to clear noise
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    # plt.imshow(mask)
    # plt.show()

    area = np.sum(np.all(mask == color, axis=-1))
    if area > 300:
        detected = True
        mask[np.any(mask != color, axis=-1)] = [0, 0, 0]

    return detected, mask


def detect_polygon(image, color):
    """
    Detect a polygon in an image based on a specified color.

    Args:
        image (np.ndarray): The input image.
        color (List[int]): The RGB color to detect the polygon for.

    Returns:
        bool: Whether a polygon was detected or not.
        np.ndarray: The binary mask of the detected polygon.

    Notes:
        - The function converts the input image to grayscale.
        - It applies Gaussian blur to reduce noise.
        - It performs edge detection using Canny.
        - It finds contours in the edge-detected image.
        - It enlarges the contour slightly by dilating it.
        - It returns the detected polygon and the binary mask.

    """
    detected = False
    # plt.imshow(image)
    # plt.show()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 8)
    # Apply morphological operations to clear noise
    # kernel1 = np.ones((3, 3), np.uint8)
    # kernel2 = np.ones((9, 9), np.uint8)
    # adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel1)
    # adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel2)

    # plt.imshow(adaptive_thresh, cmap='binary')
    # plt.show()

    # Perform edge detection using Canny with automatic thresholds
    v = np.mean(adaptive_thresh)
    lower = int(max(0, 0.6 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(adaptive_thresh, lower, upper, L2gradient=True)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    # Enlarge the contour slightly by dilating it
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, [c], -1, 255, thickness=cv2.FILLED)
    dilated_contour_mask = cv2.dilate(contour_mask, np.ones((9, 9), np.uint8))  # Adjust the kernel size as needed
    dilated_contours, _ = cv2.findContours(dilated_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    enlarged_c = max(dilated_contours, key=cv2.contourArea)
    # enlarged_c_mask = np.zeros_like(gray)
    # cv2.drawContours(enlarged_c_mask, [enlarged_c], -1, 255, thickness=cv2.FILLED)
    # plt.imshow(enlarged_c_mask)
    # plt.show()
    # Create a blank canvas to draw the contour
    #    contour_canvas = np.zeros_like(image)

    # Function to generate a random color
    #    def get_random_color():
    #        return tuple(np.random.randint(0, 255, 3).tolist())

    # Dictionary to store contours and their colors
    #    contour_colors = {}

    # Draw each contour with a different color
    #    for i, contour in enumerate(contours):
    #        color = get_random_color()
    #        contour_colors[i] = color
    #        cv2.drawContours(contour_canvas, [contour], -1, color, 2)

    # Display the contour
    #    plt.imshow(cv2.cvtColor(contour_canvas, cv2.COLOR_BGR2RGB))
    #    plt.title('Contour')
    #    plt.show()

    # Create a mask for the background
    mask = np.zeros_like(image, dtype=np.uint8)

    area = cv2.contourArea(enlarged_c)  # Calculate the area of the contour
    if area > 400:
        detected = True
        cv2.fillPoly(mask, [enlarged_c], tuple(color))

    if detected:
        # Apply Gaussian blur to the mask to smooth edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Apply morphological operations to clear noise
        kernel1 = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
        # plt.imshow(mask)
        # plt.show()

        mask[np.any(mask != color, axis=-1)] = [0, 0, 0]
        return detected, mask
    else:
        return detected, _


def process_prediction(save_image_and_masks, input_image, prediction, i, classification, device,
                       pred, live, video_name, road_sign_count, main_folder_path):
    """
    Process a prediction to create a final result with a colored shape on a black background.

    Args:
        save_image_and_masks (bool): Whether to save the image and mask.
        input_image (np.ndarray): The input image.
        prediction (dict): The prediction dictionary.
        i (int): The index of the prediction.
        classification (torch.nn.Module): The classification model.
        device (torch.device): The device to use.
        pred (int): The prediction index.
        live (bool): Whether the image is live or not.
        video_name (str): The name of the video.
        road_sign_count (int): The count of road signs.
        main_folder_path (str): The path to the main folder.

    Returns:
        None

    Notes:
        - The function processes a prediction to create a final result.
        - It extracts the necessary information from the prediction dictionary.
        - It creates a mask for the specific color.
        - The function creates a shape based on the mask and the color.
        - The shape is either an octagon, triangle, or oval, depending on the label.
        - The function also creates a mask for the specific color.
        - Finally, it creates the final result by setting the pixels in the mask to the specified color.

    """
    image_original = input_image.copy()
    # Extract the image dimensions
    height, width, _ = input_image.shape

    # Extract bounding box coordinates
    x_center = prediction["x"]
    y_center = prediction["y"]
    w_abs = prediction["width"]
    h_abs = prediction["height"]

    # Calculate top-left and bottom-right coordinates of the bounding box
    x_min = int(x_center - w_abs / 2)
    y_min = int(y_center - h_abs / 2)
    x_max = int(x_center + w_abs / 2)
    y_max = int(y_center + h_abs / 2)

    cropped_region_original = image_original[max(0, y_min - 10):min(y_max + 10, image_original.shape[0]),
                                             max(0, x_min - 10):min(x_max + 10, image_original.shape[1])]

    # plt.imshow(cropped_region_original)
    # plt.show()
    cropped_region_pil = Image.fromarray(cropped_region_original)

    # Convert the image to PyTorch tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # Resize to the size your model expects
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cropped_tensor = transform(cropped_region_pil).unsqueeze(0).to(device)

    gap_threshold = 0.8  # Set your desired gap threshold
    with torch.no_grad():
        output = classification(cropped_tensor)
        probabilities = torch.softmax(output, dim=1)
        # Get the top two probabilities
        top_two_probs, top_two_classes = torch.topk(probabilities, 2, dim=1)
        # Calculate the gap between the highest and the second-highest probability
        gap = (top_two_probs[0][0] - top_two_probs[0][1]).item()
        # Check if the gap is greater than the threshold
        if gap >= gap_threshold:
            predicted_class = top_two_classes[0][0].item()
        else:
            predicted_class = 2
            # print(f"Prediction not confident enough (gap: {gap:.2f}), class is none")

    class_names = ['give_way_sign', 'no_entry_sign', 'none', 'stop_sign']
    predicted_sign = class_names[predicted_class]

    if predicted_sign != 'none':
        detected, cropped_mask, cropped_image = create_mask(cropped_region_original, predicted_sign,
                                                            live, main_folder_path)
        if live is False:
            if detected is True:
                cropped_mask = cv2.resize(cropped_mask, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                cropped_image = cv2.resize(cropped_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                output_path = f'{save_image_and_masks}/cropped_images/{predicted_sign}/image{i}.png'
                if not os.path.exists(output_path):
                    cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                else:
                    output_path = f'{save_image_and_masks}/cropped_images/{predicted_sign}/image{i}_{pred}.png'
                    cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                save_image_path = f'{save_image_and_masks}/cropped_masks/{predicted_sign}/mask{i}.png'
                if not os.path.exists(save_image_path):
                    cv2.imwrite(save_image_path, cv2.cvtColor(cropped_mask, cv2.COLOR_RGB2BGR))
                else:
                    save_image_path = f'{save_image_and_masks}/cropped_masks/{predicted_sign}/mask{i}_{pred}.png'
                    cv2.imwrite(save_image_path, cv2.cvtColor(cropped_mask, cv2.COLOR_RGB2BGR))
                # print(f"1 {predicted_sign}(s) detected in the image.")
            else:
                pass
                # print('No road sign was detected, skip to next image')

        else:
            if detected is True:
                location_file = f'{save_image_and_masks}/image_location/{video_name}/frame{i}_location.txt'
                is_new_object = True
                # Read existing locations
                if os.path.exists(location_file):
                    with open(location_file, 'r') as f:
                        data = f.readlines()

                    for j in range(0, len(data), 5):  # Assuming 4 lines per object and 1 blank line
                        prev_x_min = int(data[j].strip().split(': ')[1])
                        prev_y_min = int(data[j + 1].strip().split(': ')[1])
                        prev_x_max = int(data[j + 2].strip().split(': ')[1])
                        prev_y_max = int(data[j + 3].strip().split(': ')[1])

                        # Check if the new bounding box is close to any previous one
                        if is_close(x_min, y_min, x_max, y_max, prev_x_min, prev_y_min, prev_x_max, prev_y_max):
                            is_new_object = False
                            break

                    # If the object is new, save the location
                if is_new_object:
                    with open(location_file, 'a') as f:
                        f.write(f'x_min: {x_min}\n')
                        f.write(f'y_min: {y_min}\n')
                        f.write(f'x_max: {x_max}\n')
                        f.write(f'y_max: {y_max}\n\n')

                    # plt.imshow(cropped_image)
                    # plt.title('cropped_image')
                    # plt.show()

                    mask_on_image = image_original.copy()
                    mask_on_image[max(0, y_min - 10):min(y_max + 10, image_original.shape[0]),
                                  max(0, x_min - 10):min(x_max + 10, image_original.shape[1])] = cropped_image

                    # plt.imshow(mask_on_image[max(0, y_min - 10):min(y_max + 10, image_original.shape[0]),
                    #                         max(0, x_min - 10):min(x_max + 10, image_original.shape[1])])
                    # plt.title('mask_on_image')
                    # plt.show()

                    output_path = f'{save_image_and_masks}/frame_with_mask/{video_name}'
                    os.path.exists(output_path)
                    output_image_path = f'{output_path}/frame_{i}.png'
                    cv2.imwrite(output_image_path, cv2.cvtColor(mask_on_image, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    output_image = mask_on_image.copy()
                    return road_sign_count, output_image

    else:
        road_sign_count -= 1
        # print('not wonted road sign')

    output_image = input_image.copy()
    return road_sign_count, output_image


def is_close(x_min1, y_min1, x_max1, y_max1, x_min2, y_min2, x_max2, y_max2, threshold=30):
    """
    Check if two numbers are close to each other.

    Args:
        a (float): The first number.
        b (float): The second number.
        rel_tol (float, optional): The relative tolerance. Defaults to 1e-09.
        abs_tol (float, optional): The absolute tolerance. Defaults to 0.0.

    Returns:
        bool: Whether the numbers are close or not.

    Notes:
        - The function checks if the absolute difference between a and b is less than or equal to the absolute tolerance.
        - If the absolute difference is greater than the absolute tolerance, the function checks if the relative difference
          between a and b is less than or equal to the relative tolerance.

    """
    # Calculate the center of both bounding boxes
    center1_x = (x_min1 + x_max1) / 2
    center1_y = (y_min1 + y_max1) / 2
    center2_x = (x_min2 + x_max2) / 2
    center2_y = (y_min2 + y_max2) / 2

    # Calculate the distance between the centers
    distance = math.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)

    # Check if the distance is below the threshold
    return distance < threshold


def save_images_from_loader(loader, save_path):
    """
    Save images from the DataLoader to the specified path, creating class-specific subfolders.

    Args:
        loader (DataLoader): The DataLoader to extract images from.
        save_path (str): The path to save the images.

    Returns:
        None

    Notes:
        - The function extracts images from the DataLoader.
        - It creates subfolders in the save_path for each class.
        - It saves the images as PNG files in the respective class-specific subfolders.

    """
    ensure_folder_exists(save_path)
    image_save = f'{save_path}/images'
    mask_save = f'{save_path}/masks'

    if os.path.exists(image_save):
        shutil.rmtree(image_save)
    if os.path.exists(mask_save):
        shutil.rmtree(mask_save)

    os.makedirs(image_save)
    os.makedirs(mask_save)

    # Create a dictionary to keep track of class names and indices
    class_names = loader.dataset.dataset.get_class_names()
    for class_name in class_names:
        image_class_folder = os.path.join(image_save, class_name)
        ensure_folder_exists(image_class_folder)
        mask_class_folder = os.path.join(mask_save, class_name)
        ensure_folder_exists(mask_class_folder)

    for i, (masks, images, labels) in enumerate(loader):
        for j in range(images.size(0)):
            # Get the class index from the label
            class_idx = labels[j].item()  # Assuming labels are indices
            class_name = class_names[class_idx]

            # Save image
            img = transforms.ToPILImage()(images[j].cpu())
            img.save(os.path.join(image_save, class_name, f'image_{i * loader.batch_size + j}.png'))

            # Save corresponding mask
            mask = transforms.ToPILImage()(masks[j].cpu())
            mask.save(os.path.join(mask_save, class_name, f'mask_{i * loader.batch_size + j}.png'))


def compute_ssim(original, reconstructed):
    """
    Compute the Structural Similarity Index Measure (SSIM) between the original and reconstructed images.

    Parameters:
        original (torch.Tensor): The original image tensor in NCHW format.
        reconstructed (torch.Tensor): The reconstructed image tensor in NCHW format.

    Returns:
        float: The mean SSIM value.

    Notes:
        - The original and reconstructed images are converted to NHWC format.
        - The SSIM is computed for each pair of corresponding images.
        - The mean SSIM value is returned.

    """
    original = original.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NCHW -> NHWC
    reconstructed = reconstructed.cpu().detach().numpy().transpose(0, 2, 3, 1)  # Convert to NCHW -> NHWC
    ssim_values = [ssim(orig, rec, win_size=11, channel_axis=2, data_range=1.0) for orig, rec in
                   zip(original, reconstructed)]
    return np.mean(ssim_values)


def create_folder_path(video_name, main_folder_path, videos_path):
    """
    Create the folder paths for the given video.

    Args:
        video_name (str): The name of the video.
        main_folder_path (str): The path to the main folder.
        videos_path (str): The path to the videos folder.

    Returns:
        tuple: A tuple containing the paths to the frames, video, frame with mask, video mask, mask frames,
               frame reconstruct, image location, and video reconstruct folders.

    """
    # part1 folders create
    part1_files_path = f'{main_folder_path}/part1'
    ensure_folder_exists(part1_files_path)
    video_path = f'{videos_path}/{video_name}.yuv'
    frames_path = f'{part1_files_path}/frames/{video_name}'
    ensure_folder_exists(frames_path)
    image_location_path = f'{part1_files_path}/image_location/{video_name}'
    ensure_folder_exists(image_location_path)
    frame_with_mask_path = f'{part1_files_path}/frame_with_mask/{video_name}'
    ensure_folder_exists(frame_with_mask_path)
    videos_mask_path = f'{part1_files_path}/video_mask'
    ensure_folder_exists(videos_mask_path)

    # part2 folders create
    part2_files_path = f'{main_folder_path}/part2'
    ensure_folder_exists(part2_files_path)
    video_mask_name = f'{video_name}_mask'

    mask_frames_path = f'{part2_files_path}/frames/{video_mask_name}'
    ensure_folder_exists(mask_frames_path)
    frames_reconstruct_path = f'{part2_files_path}/frame_reconstruct/{video_mask_name}'
    ensure_folder_exists(frames_reconstruct_path)
    video_reconstruct_path = f'{part2_files_path}/video_reconstruct'
    ensure_folder_exists(video_reconstruct_path)
    return (frames_path, video_path, frame_with_mask_path, videos_mask_path, mask_frames_path,
            frames_reconstruct_path, image_location_path, video_reconstruct_path)


def extract_frames_using_ffmpeg(video_path, output_folder):
    """
    Extracts frames from a video using FFMPEG.

    Args:
        video_path (str): The path to the video file.
        output_folder (str): The path to the folder where the extracted frames will be saved.

    Returns:
        None

    Raises:
        FileNotFoundError: If the video file does not exist.
        PermissionError: If the user does not have permission to read the video file.

    Notes:
        - The function uses FFMPEG to extract frames from the video.
        - The output folder is created if it does not exist.
        - The frames are saved in the output folder with the pattern 'frame_%d.png'.
        - The function uses the full path to the FFMPEG executable.

    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Output pattern for the frames
    output_pattern = os.path.join(output_folder, 'frame_%d.png')

    # Run FFMPEG to extract frames from the video
    (
        ffmpeg
        .input(video_path)  # No need for framerate here
        .output(output_pattern)
        .run(cmd=r'D:\ffmpeg-7.0.2-full_build-shared\bin\ffmpeg.exe')  # Full path to ffmpeg.exe
    )


def create_video_using_ffmpeg(folder_path, output_file, frame_rate=30):
    """
    Creates a video using FFMPEG.

    Args:
        folder_path (str): The path to the folder containing the frames.
        output_file (str): The path to the output video file.
        frame_rate (int, optional): The frame rate of the video. Defaults to 30.

    Raises:
        ValueError: If the folder_path does not exist.

    Returns:
        None

    Notes:
        - The function uses FFMPEG to create a video from the frames.
        - The frames are expected to be in the folder_path with the pattern 'frame_%d.png'.
        - The output video file is saved at the specified output_file path.
        - The function uses the full path to the FFMPEG executable.

    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder '{folder_path}' does not exist.")

    # Frame input pattern
    frame_pattern = os.path.join(folder_path, 'frame_%d.png')

    # Run FFMPEG with the full path to ffmpeg.exe
    (
        ffmpeg
        .input(frame_pattern, framerate=frame_rate)
        .output(output_file, vcodec='libx264', pix_fmt='yuv420p')
        .run(cmd=r'D:\ffmpeg-7.0.2-full_build-shared\bin\ffmpeg.exe')  # Full path to ffmpeg.exe
    )


class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(dataset1) == len(dataset2), "Datasets must be of the same length"

        # Extract class names from dataset1 (assuming both datasets have the same classes)
        self.class_names = dataset1.classes

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        img1, _ = self.dataset1[idx]
        img2, label = self.dataset2[idx]
        return img1, img2, label

    def get_class_names(self):
        return self.class_names
