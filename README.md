# DL-Traffic-Signs-Overlays-by-Artificial-Data-Set

This project focuses on improving traffic sign recognition for autonomous vehicles, particularly under challenging conditions. It aims to enhance video quality and minimize bandwidth usage for remote driving by developing an algorithm that generates colored masks for traffic signs. A trained autoencoder model reconstructs these signs from their corresponding masks, addressing issues like varying angles, distances, and visibility impairments such as glare or vandalism. Our approach utilizes a database of real-world images to train the model, ensuring robustness in diverse driving scenarios.

# Project Goals
Real-World Application: Ensure effective recognition and reconstruction of traffic signs in unpredictable environments, prioritizing visual clarity.
Diverse Sign Recognition: Train the system to handle a wide range of road signs, aiming for low reconstruction loss and high similarity (SSIM) between original and reconstructed images.
Robustness to Visibility Conditions: Develop the system to function reliably under various visibility challenges, including adverse weather and lighting.
Demonstration Videos: Create two videos: one showcasing colored masks in place of traffic signs and another demonstrating the autoencoder's reconstruction process.

# Purpose
The project aims to improve traffic sign visibility on a remote driver's screen by classifying signs into color-coded masks. This method enhances clarity while optimizing bandwidth by transmitting simplified mask data instead of high-resolution images, making it suitable for real-world applications.

# Final Product Specifications
The project consists of two main components:

1. Autonomous Vehicle
An algorithm processes video from the vehicle’s camera, recognizing and classifying traffic signs, creating colored masks, and compressing the masked frames into a video for transmission.

2. Remote Driver
An algorithm receives the compressed video, detects the masks, and uses a trained autoencoder to reconstruct the traffic signs, overlaying them on the original video for enhanced clarity.
