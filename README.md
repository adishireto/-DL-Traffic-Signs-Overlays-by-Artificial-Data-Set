# DL Traffic Signs Overlays by Artificial Data Set

This project focuses on improving traffic sign recognition for autonomous vehicles, particularly under challenging conditions. It aims to enhance video quality and minimize bandwidth usage for remote driving by developing an algorithm that generates colored masks for traffic signs. A trained autoencoder model reconstructs these signs from their corresponding masks, addressing issues like varying angles, distances, and visibility impairments such as glare or vandalism. Our approach utilizes a database of real-world images to train the model, ensuring robustness in diverse driving scenarios.

![example of the system.](assets/example_of_the_system.png)

*Figure 1: Example of the system on our traffic sign*

## Project Goals
Real-World Application: Ensure effective recognition and reconstruction of traffic signs in unpredictable environments, prioritizing visual clarity.
Diverse Sign Recognition: Train the system to handle a wide range of road signs, aiming for low reconstruction loss and high similarity (SSIM) between original and reconstructed images.
Robustness to Visibility Conditions: Develop the system to function reliably under various visibility challenges, including adverse weather and lighting.
Demonstration Videos: Create two videos: one showcasing colored masks in place of traffic signs and another demonstrating the autoencoder's reconstruction process.

## Purpose
The project aims to improve traffic sign visibility on a remote driver's screen by classifying signs into color-coded masks. This method enhances clarity while optimizing bandwidth by transmitting simplified mask data instead of high-resolution images, making it suitable for real-world applications.


# System Definition
The project consists of two main components:

- Vehicle Side: A video capture and mask creation system, which processes and compresses traffic sign data for transmission.
- Remote Driver Side: A video receiving and traffic sign reconstruction system, where the signs are identified from the masks and restored to a high clarity suitable for remote driving.
  
![system block diagram](assets/system_block_diagram.png)

*Figure 2: system block diagram*


## Process Details


![system block diagram](assets/Pre-Processing_Real-time_image.png)

*Figure 3: Pre-Processing Real-time image*

Vehicle Side:
- Video Capture: Receives continuous video from the vehicle's camera.
- Frame Extraction: Extracts individual frames for processing.
- Traffic Sign Detection and Mask Creation: Identifies traffic signs in each frame using YOLO, classifies into three main types: Stop, Give Way, and No Entry then overlays color-coded masks (e.g., yellow for "stop" signs) on the frames.
- Video with Masks Creation: Compiles the processed frames with masks back into a video format for efficient transmission to the remote driver.

![system block diagram](assets/Image_with_the_colored_mask.png)

*Figure 4: Image with the colored mask*
 
Remote Driver Side:
- Receiving and Frame Extraction: Processes the masked video frames.
- Traffic Sign Reconstruction with Autoencoder: Each traffic sign mask is passed through a trained autoencoder, which reconstructs the sign based on learned features from the training set.
- Video Compilation with Restored Signs: Combines the reconstructed frames into a video, enhancing sign visibility and clarity for the remote driver.

  
![Reconstruct image after the autoencoder](assets/Reconstruct_image_after_the_autoencoder.png)

*Figure 5: Reconstruct image after the autoencoder*

## Models Created
Traffic Sign Classification Model:
- Data Collection: Images of real-world traffic signs were curated and categorized into foure main types: Stop, Give Way, No Entry and other (that cointain all other traffic signs). Additional images were sourced and augmented to increase robustness under various conditions.
- Model Architecture: A CNN was designed and trained on these labeled images, achieving high accuracy in classifying traffic signs. The model demonstrates strong performance in identifying and classifying signs from a distance and under adverse conditions.
- Training Process: The model was trained with the Adam optimizer, using a learning rate of 0.0001 and Cross-Entropy Loss to handle classification tasks. Training was conducted over 20 epochs with early stopping to prevent overfitting. The final model demonstrated a high level of accuracy (98.3%) across all classes, as verified by a confusion matrix and metrics like accuracy, precision, recall, and F1-score.

![Confusin matrix of classification model](assets/Confusin_matrix.png)

*Figure 6: Confusin matrix of classification model*


Autoencoder Model:
- Purpose: The autoencoder reconstructs traffic signs from the color-coded masks generated on the vehicle side.
- Architecture: The autoencoder includes an encoder to compress masked input into a latent representation and a decoder to reconstruct the image. Skip connections between corresponding encoder and decoder layers ensure that spatial details from the original image are retained, improving reconstruction quality.
- Training Process: The model was trained with L1 and MSE loss functions, with high SSIM scores indicating successful reconstruction of the signs in varied conditions.

![Visualization of Autoencoder architecture](assets/visualization_of_the_autoencoder_archirecture.png)

*Figure 7: Visualization of Autoencoder architecture*

![Visualization of Autoencoder architecture](assets/Results_autoencoder_model.png)

*Figure 8: Results autoencoder model*

## System Results
The project includes two demonstration videos:

- Mask Overlay Video: Displays the vehicle's processed view with masked signs.
- Reconstructed Sign Video: Shows the remote driver’s view with fully reconstructed signs.
