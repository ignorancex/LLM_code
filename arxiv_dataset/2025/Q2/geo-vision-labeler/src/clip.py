# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPClassifier:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None):
        """
        Initialize the CLIP model and processor.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def classify_image(self, image, classes, context):
        """
        Classify an image using CLIP based on image-text similarity.

        Args:
            image (PIL.Image): The image to classify.
            classes (list of str): Class labels (e.g., ["buildings", "no buildings"]).
            context (str): Contextual information to prepend to class text.

        Returns:
            tuple: (predicted class label, list of class probabilities)
        """

        # Build class prompts
        text_inputs = [
            f"{context.strip()} Label '{label}'.".strip() for label in classes
        ]
        
        # Prepare inputs and run model
        inputs = self.processor(
            text=text_inputs,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()
        return classes[pred_idx], probs.tolist()
