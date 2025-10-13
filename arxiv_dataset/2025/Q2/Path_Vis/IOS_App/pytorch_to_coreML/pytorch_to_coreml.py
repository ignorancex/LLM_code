import urllib
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torchvision
import json

from torchvision import transforms
from PIL import Image

import coremltools as ct
from modelclass import SegmentationModel

# Load the PyTorch model
model = SegmentationModel(2)
model.load_state_dict(torch.load("best.pth", map_location=torch.device("cpu")))

# Dummy input - adjust according to your model input shape
input_batch = torch.randn(1, 3, 640, 640)

preprocess = transforms.Compose(
    [
        # Resize to match the model's input size
        transforms.Resize((640, 640)),
        transforms.Pad(30),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),  # Convert to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ]
)

# Trace the model
trace = torch.jit.trace(model, input_batch)



mlmodel = ct.convert(
    trace,
    inputs=[
        ct.ImageType(name="image",
            shape=(1, 3, 640, 640)
        )
    ],
    minimum_deployment_target=ct.target.macOS13,
)

mlmodel.save("SegmentationModel_no_metadata.mlpackage")

mlmodel = ct.models.MLModel("SegmentationModel_no_metadata.mlpackage")

labels_json = {"labels": ["pathology_slide", "null"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata["com.apple.coreml.model.preview.params"] = json.dumps(
    labels_json
)

mlmodel.save("SegmentationModel_with_metadata.mlpackage")
