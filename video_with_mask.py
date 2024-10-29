import cv2
import torch
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# functions import
from functions import count_items_in_folder, process_prediction, extract_frames_using_ffmpeg, create_video_using_ffmpeg
from models import CNNModel


def video_with_mask(classification_path, frames_path, video_path, frame_with_mask_path,
                    video_name, main_folder_path, videos_mask_path):
    """
    Generates a video with masks using the provided classification model and YOLO model.

    Args:
        classification_path (str): The path to the classification model.
        frames_path (str): The path to the folder containing the frames.
        video_path (str): The path to the video.
        frame_with_mask_path (str): The path to the folder where the frames with masks will be saved.
        video_name (str): The name of the video.
        main_folder_path (str): The path to the main folder.
        videos_mask_path (str): The path to the folder where the video with masks will be saved.

    Returns:
        None

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load the classification
    files_path = f'{main_folder_path}/part1'

    classification = CNNModel().to(device)
    classification.load_state_dict(torch.load(classification_path, map_location=device))
    classification.eval()

    # YOLO model for road sign
    custom_configuration = InferenceConfiguration(confidence_threshold=0.05)
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="SKyXZA5fmgpDTa7iHlWD")

    model_configs = [
        {"model_id": "universe-data-test/3", "class_name": ["road-sign"]},
        {"model_id": "logistics-sz9jr/2", "class_name": ["road sign"]},
    ]
    extract_frames_using_ffmpeg(video_path, frames_path)
    frames_number = count_items_in_folder(frames_path)

    for i in range(0, frames_number):
        # Load the original image
        image_path = f'{frames_path}/frame_{i}.png'
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        print(f'frame {i}')

        found_road_sign = False
        for config in model_configs:
            model_id = config["model_id"]
            class_name = config["class_name"]

            with CLIENT.use_configuration(custom_configuration):
                result = CLIENT.infer(image_path, model_id=model_id)

            if 'predictions' in result and not result['predictions']:
                continue
            else:
                predictions = result['predictions']
                road_sign_count = 0

                for pred in range(len(predictions)):
                    prediction = predictions[pred]

                    if prediction["class"] in class_name:
                        road_sign_count += 1
                        road_sign_count, new_image = process_prediction(files_path, input_image, prediction, i,
                                                                        classification, device, pred,
                                                                        True, video_name, road_sign_count,
                                                                        main_folder_path)
                        input_image = new_image
                if road_sign_count > 2:
                    found_road_sign = True
                    break

        if not found_road_sign:
            continue

    create_video_using_ffmpeg(frame_with_mask_path, videos_mask_path, frame_rate=30)
