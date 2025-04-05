# pylint: disable=missing-module-docstring, missing-function-docstring
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

IMG_SIZE = 5

mat = np.array(
    [
        [
            [82, 10, 160, 149, 128],
            [93, 34, 32, 169, 62],
            [114, 170, 48, 209, 104],
            [43, 197, 122, 157, 245],
            [165, 249, 118, 240, 116],
        ],
        [
            [38, 53, 146, 243, 160],
            [55, 137, 67, 25, 197],
            [58, 70, 224, 235, 192],
            [53, 13, 83, 38, 32],
            [109, 151, 20, 50, 64],
        ],
        [
            [71, 112, 222, 68, 226],
            [215, 198, 104, 73, 171],
            [144, 248, 253, 232, 66],
            [171, 197, 33, 15, 116],
            [161, 139, 3, 90, 126],
        ],
    ],
    dtype=np.uint8,
)


def preprocess_torch(img: np.ndarray) -> np.ndarray:
    """
    Prepare an image for input to the model.

    Args:
        img (np.ndarray): The input image for preprocessing.

    Returns:
        np.ndarray: Preprocessed image.
    """
    # Transpose the array from (3, height, width) to (height, width, 3)
    img = np.transpose(img, (1, 2, 0))
    test_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return test_transforms(Image.fromarray(img))


def preprocess_onnx(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img, (1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255
    for i in range(3):
        x[..., i] = (x[..., i] - mean[i]) / std[i]
    return x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]


out = preprocess_onnx(mat)
print(out)
