import asyncio
import json
import os
import sys
import time
from datetime import datetime
import cv2
from pypylon import pylon as py
from ImageCap import PylonCameras
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np

#THIS IS THE WORKING CODE FOR ASSEMAI

# Define constants
SIZE = (1080, 720)
BASE_IMG_LOC = 'static'
crop_box = (300, 250, 500, 320)
output_file = 'model_output.txt'

# Initialize the model
device = torch.device('cpu')
from efficientnet_pytorch import EfficientNet

# Load the model architecture with pre-trained weights
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
# model = models.efficientnet_b0(weights=None)  # Initialize model architecture
state_dict = torch.load('KI_new_seperate_loss.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()  # Set the model to evaluation mode
model.to(device)

# Ensure the base image directory exists
os.makedirs(BASE_IMG_LOC, exist_ok=True)

def save_image(image, file_path):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    pil_image.save(file_path)
def process_image(image):
    pil_image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    input_image = transform(pil_image).unsqueeze(0).to('cpu')  # Add batch dimension and move to device

    # Perform model inference
    with torch.no_grad():
        output = model(input_image)

    print("Raw Outputs (Logits):", output)

    # Convert logits to probabilities using softmax
    probs = torch.softmax(output, dim=1)
    print("Probabilities:", probs)

    # Get the predicted class
    _, predicted_class = torch.max(probs, 1)
    print("Predicted Class:", predicted_class.item())

    # Mapping the predicted class to its label (optional, if you have a label mapping)
    class_labels = ["No Anomaly", "NoNose", "NoNose,NoBody2", "NoNose,NoBody2,NoBody1", "NoBody1"]
    predicted_label = class_labels[predicted_class.item()]
    print(f"Predicted Label: {predicted_label}")

    return predicted_label

def capture_and_save_images(cap):
    count = 0
    batch = f'BATCH{count // 2 + 1}'
    os.makedirs(os.path.join(BASE_IMG_LOC, batch), exist_ok=True)

    # Capture images from both cameras
    for i in range(2):
        res = cap.cameras[i].RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            idx, device = cap.get_image_device(res)
            img = cap.converters[i].Convert(res)
            image = img.GetArray()
            image = cap.set_img_size(image, SIZE)
            image = cap.adjust_white_balance(image)

            # Save the raw image
            raw_path = os.path.join(BASE_IMG_LOC, batch, f'{count:06d}_camera{i}.png')
            save_image(image, raw_path)

            # Crop the image
            x_start, y_start, x_end, y_end = crop_box
            cropped_image = image[y_start:y_end, x_start:x_end]

            # Save the cropped image
            cropped_path = os.path.join(BASE_IMG_LOC, batch, f'{count:06d}_camera{i}_cropped.png')
            save_image(cropped_image, cropped_path)

            # Process the cropped image
            if i == 1:
                output = process_image(cropped_image)

                # Save the output to a file
                with open(output_file, 'a') as f:  # Use 'a' to append for multiple images
                    f.write(f"Image {count} from Camera {i} Output: {output}\n")

                    # Optionally print the output
                print(f"Model Output for Image {count} from Camera {i}:", output)

            count += 1

    print("Image capture and processing complete.")
    return output
# async def run_main():
#     cap = PylonCameras(num_devices=2)
#     cap.grab('LatestOnly')
#     await capture_and_save_images(cap)
#
#     # Ensure the camera is released
#     cap.cameras.Close()
#
# asyncio.run(run_main())
