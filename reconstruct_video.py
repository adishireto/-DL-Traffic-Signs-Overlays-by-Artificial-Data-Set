import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
# from PIL import Image

# functions import
from functions import count_items_in_folder, extract_frames_using_ffmpeg, create_video_using_ffmpeg
from torch_autoencoder_model import Autoencoder


def reconstruct_video(video_name, model_path, videos_mask_path, mask_frames_path,
                      frames_reconstruct_path, image_location_path, video_reconstruct_path):
    """
    Reconstructs a video from a mask video file.

    Args:
        video_name (str): The name of the video.
        model_path (str): The path to the autoencoder model.
        videos_mask_path (str): The path to the folder containing the mask video.
        mask_frames_path (str): The path to the folder where the extracted frames will be saved.
        frames_reconstruct_path (str): The path to the folder where the reconstructed frames will be saved.
        image_location_path (str): The path to the folder containing the location information.
        video_reconstruct_path (str): The path to the reconstructed video.

    Returns:
        None

    Notes:
        - The function reconstructs a video from a mask video file.
        - It extracts frames from the mask video using FFMPEG.
        - It processes each frame to reconstruct the original image.
        - It saves the reconstructed frames in the specified folder.
        - It creates a video from the reconstructed frames using FFMPEG.

    """
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    autoencoder.eval()
    video_mask_file = f'{videos_mask_path}/{video_name}_mask.yuv'
    extract_frames_using_ffmpeg(video_mask_file, mask_frames_path)
    frames_number = count_items_in_folder(mask_frames_path)

    for i in range(0, frames_number):
        # Load the original image
        image_path = f'{mask_frames_path}/frame_{i}.png'
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        location_file_path = f'{image_location_path}/frame{i}_location.txt'

        if os.path.exists(location_file_path):
            with open(location_file_path, 'r') as f:
                data = f.readlines()
                print(f'frame {i}')
                # Initialize the reconstructed frame before the loop
                reconstructed_frame = input_image.copy()

                for j in range(0, len(data), 5):  # Assuming 4 lines per object and 1 blank line
                    # Parse the data
                    x_min = int(data[j].strip().split(': ')[1])
                    y_min = int(data[j + 1].strip().split(': ')[1])
                    x_max = int(data[j + 2].strip().split(': ')[1])
                    y_max = int(data[j + 3].strip().split(': ')[1])

                    cropped_region_original = input_image[max(0, y_min - 10):min(y_max + 10, input_image.shape[0]),
                                                          max(0, x_min - 10):min(x_max + 10, input_image.shape[1])]
                    plt.imshow(cropped_region_original)
                    # plt.show()

                    mask = cropped_region_original.copy()

                    # Define the excluded colors
                    excluded_colors = np.array([
                        [255, 255, 0],
                        [255, 0, 255],
                        [0, 255, 255]
                    ])

                    # Initialize mask condition to true for all pixels
                    mask_condition = np.ones(cropped_region_original.shape[:2], dtype=bool)

                    # Update mask condition to exclude specific colors
                    for color in excluded_colors:
                        mask_condition &= (cropped_region_original != color).any(axis=-1)

                    # Apply the mask
                    mask[mask_condition] = [0, 0, 0]
                    kernel1 = np.ones((3, 3), np.uint8)
                    kernel2 = np.ones((7, 7), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

                    # Convert the image to PyTorch tensor
                    transform = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((256, 256)),
                                                    transforms.ToTensor()])
                    cropped_tensor = transform(mask).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = autoencoder(cropped_tensor)

                    reconstructed_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Scale to [0, 255]

                    # Resize the reconstructed image to match the original cropped region size
                    resized_image = cv2.resize(reconstructed_image,
                                               (cropped_region_original.shape[1], cropped_region_original.shape[0]))

                    # Create a boolean mask where the resized_image is not black
                    non_black_mask = np.any(mask != [0, 0, 0], axis=-1)

                    # Apply this mask to update only the non-black pixels in cropped_region_original
                    cropped_region_original[non_black_mask] = resized_image[non_black_mask]

                    reconstructed_frame[max(0, y_min - 10):min(y_max + 10, input_image.shape[0]),
                                        max(0, x_min - 10):min(x_max + 10, input_image.shape[1])] = cropped_region_original

                frame_reconstruct_path = f'{frames_reconstruct_path}/frame_{i}.png'
                cv2.imwrite(frame_reconstruct_path, cv2.cvtColor(reconstructed_frame, cv2.COLOR_RGB2BGR))

    create_video_using_ffmpeg(frames_reconstruct_path, video_reconstruct_path, frame_rate=30)
