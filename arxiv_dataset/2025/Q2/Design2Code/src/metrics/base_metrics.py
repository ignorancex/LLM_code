from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from PIL import Image
from .screenshot_capture import *
import mimetypes

class Metrics(ABC):
    @abstractmethod
    def __call__(self, reference_path: str, hypothesis_path: str) -> Any:
        """
        Abstract method to be implemented by subclasses.

        Args:
            reference (str): Path to the reference image.
            hypothesis (str): Path to the hypothesis image.

        Returns:
            Any: The computed metric value.
        """
        raise NotImplementedError("Subclass must implement this method")

class VisualMetrics(Metrics):
    def _is_html(self, path: str) -> bool:
        """
        Check if the given path is an HTML file based on its extension.
        """
        mime_type, _ = mimetypes.guess_type(path)
        return mime_type == 'text/html' or path.lower().endswith('.html')
    
    def _screenshot_capture_from_html(self, html_path):
        return take_screenshot_from_url(html_path)

    def _load_img(self, path: str) -> np.ndarray:
        """
        Load an image from a path and convert it to a numpy array.

        Args:
            path (str): Path to the image file.

        Returns:
            np.ndarray: Image as a numpy array.
        """
        # Open the image file
        image = Image.open(path)
        # Convert image to numpy array
        return np.asarray(image)

    def _load_img_as_gray(self, path: str) -> np.ndarray:
        """
        Load an image from a path and convert it to grayscale.

        Args:
            path (str): Path to the image file.

        Returns:
            np.ndarray: Grayscale image as a numpy array normalized to [0, 1].
        """
        # Open image and convert it to grayscale
        image = Image.open(path).convert('L')
        # Convert to numpy array and normalize to [0, 1]
        return np.asarray(image) / 255.0
