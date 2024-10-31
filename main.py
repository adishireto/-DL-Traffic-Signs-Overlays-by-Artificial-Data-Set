import os
from functions import create_folder_path, ensure_folder_exists
from video_with_mask import video_with_mask
from reconstruct_video import reconstruct_video


# Load files for video with mask
main_folder_path = r'traffic_sign_overlays'
video_name = 'videofile_noentry'
videos_path = r'finel_project_github/videos'
ensure_folder_exists(videos_path)
# Load the classification
classification_path = r'finel_project_github/pytorch_multi_classification_11_09.pth'
classification_file = os.path.join(classification_path)
autoencoder_path = r'finel_project_github/torch_autoencoder_final.pth'
model_path = os.path.join(autoencoder_path)


(frames_path, video_path, frame_with_mask_path,
 videos_mask_path, mask_frames_path, frames_reconstruct_path,
 image_location_path, video_reconstruct_path) = create_folder_path(video_name, main_folder_path, videos_path)


video_with_mask(classification_file, frames_path, video_path, frame_with_mask_path,
                video_name, main_folder_path, videos_mask_path)


reconstruct_video(video_name, model_path, videos_mask_path, mask_frames_path,
                  frames_reconstruct_path, image_location_path, video_reconstruct_path)
