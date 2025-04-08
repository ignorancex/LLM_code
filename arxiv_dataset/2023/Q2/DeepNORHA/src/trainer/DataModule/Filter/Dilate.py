import cv2

from .Filter import Filter


class Dilate(Filter):
    """Dilate

    Dilates the image with a structuring element.
    
    Example:
    - Type: Dilate
      dilation: 3
      iterations: 1
    """
    def __init__(self, dilation, iterations = None, **args):
        super().__init__(**args)
        self._structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        self._iterations = iterations or 1

    def apply_one(self,  image, i, prepared):
        return cv2.dilate(image, self._structuring_element, iterations=self._iterations)
        