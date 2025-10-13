from .base_metrics import VisualMetrics
from skimage.metrics import structural_similarity as ssim

import numpy as np
import cv2

class MSEMetrics(VisualMetrics):
    def __init__(self):
        self.masked = False


    def __call__(self, reference_path: str, hypothesis_path: str) -> float:
        """
        Compute the Mean Squared Error (MSE) between two images.

        Args:
            reference_path (str): Path to the reference file(html or image).
            hypothesis_path (str): Path to the hypothesis file(html or image).

        Returns:
            float: The Mean Squared Error between the two images.
        """

        """if self._is_html(reference_path):
            ref_img = self._screenshot_capture_from_html(reference_path)
        if self._is_html(hypothesis_path):
            hyp_img = self._screenshot_capture_from_html(hypothesis_path)"""


        if self.masked:
            ref_img = self._load_img_as_gray(reference_path)
            hyp_img = self._load_img_as_gray(hypothesis_path)

        else:
            ref_img = self._load_img(reference_path)
            hyp_img = self._load_img(hypothesis_path)
        
        """# Ensure images are in the same shape
        if ref_img.shape != hyp_img.shape:
            raise ValueError("Images must have the same dimensions.")"""
        # Get dimensions
        ref_height, ref_width = ref_img.shape[:2]
        hyp_height, hyp_width = hyp_img.shape[:2]

        # Resize images if needed
        if ref_img.shape[:2] != hyp_img.shape[:2]:
            if ref_height * ref_width > hyp_height * hyp_width:
                # Resize ref_img to the size of hyp_img
                new_size = (hyp_width, hyp_height)
                ref_img = cv2.resize(ref_img, new_size, interpolation=cv2.INTER_AREA)
            else:
                # Resize hyp_img to the size of ref_img
                new_size = (ref_width, ref_height)
                hyp_img = cv2.resize(hyp_img, new_size, interpolation=cv2.INTER_AREA)

        
        # Compute MSE
        return np.mean((ref_img - hyp_img)**2)
    

class MaskedMSEMetrics(MSEMetrics):
    def __init__(self):
        self.masked = True


class IoUMetrics(VisualMetrics):
    def __call__(self, reference_path: str, hypothesis_path: str) -> float:
        """
        Compute the Intersection over Union (IoU) between two binary masks derived from images.

        Args:
            reference_path (str): Path to the reference image.
            hypothesis_path (str): Path to the hypothesis image.

        Returns:
            float: The Intersection over Union score between the two masks.
        """

        """ if self._is_html(reference_path):
            ref_img = self.take_screenshot_from_html_file(reference_path)
        if self._is_html(hypothesis_path):
            hyp_img = self.take_screenshot_from_html_file(hypothesis_path)"""

        ref_img = self._load_img_as_gray(reference_path)
        hyp_img = self._load_img_as_gray(hypothesis_path)
        
        
        
        """# Ensure masks are in the same shape
        if ref_mask.shape != hyp_mask.shape:
            raise ValueError("Masks must have the same dimensions.")"""
        
        ref_height, ref_width = ref_img.shape[:2]
        hyp_height, hyp_width = hyp_img.shape[:2]

        # Resize images if needed
        if ref_img.shape[:2] != hyp_img.shape[:2]:
            if ref_height * ref_width > hyp_height * hyp_width:
                # Resize ref_img to the size of hyp_img
                new_size = (hyp_width, hyp_height)
                ref_img = cv2.resize(ref_img, new_size, interpolation=cv2.INTER_AREA)
            else:
                # Resize hyp_img to the size of ref_img
                new_size = (ref_width, ref_height)
                hyp_img = cv2.resize(hyp_img, new_size, interpolation=cv2.INTER_AREA)


        # Convert images to binary masks
        ref_mask = (ref_img > 0).astype(np.uint8)
        hyp_mask = (hyp_img > 0).astype(np.uint8)

        
        # Compute IoU
        intersection = np.logical_and(ref_mask, hyp_mask).sum()
        union = np.logical_or(ref_mask, hyp_mask).sum()
        
        return intersection / union if union != 0 else 0


class SSIMMetrics(VisualMetrics):
    def __call__(self, reference_path: str, hypothesis_path: str) -> float:
        """
        Computes the SSIM score between the reference and hypothesis images.

        Args:
            reference_path (str): Path to the reference image.
            hypothesis_path (str): Path to the hypothesis image.

        Returns:
            float: The computed SSIM score.
        """
        # Load images (this is a placeholder, you would use an actual image loading library)
        ref_img = self._load_img(reference_path)
        hyp_img = self._load_img(hypothesis_path)

        ref_height, ref_width = ref_img.shape[:2]
        hyp_height, hyp_width = hyp_img.shape[:2]

        # Resize images if needed
        if ref_img.shape[:2] != hyp_img.shape[:2]:
            if ref_height * ref_width > hyp_height * hyp_width:
                # Resize ref_img to the size of hyp_img
                new_size = (hyp_width, hyp_height)
                ref_img = cv2.resize(ref_img, new_size, interpolation=cv2.INTER_AREA)
            else:
                # Resize hyp_img to the size of ref_img
                new_size = (ref_width, ref_height)
                hyp_img = cv2.resize(hyp_img, new_size, interpolation=cv2.INTER_AREA)


        # Compute SSIM
        ssim_score = ssim(ref_img, hyp_img, channel_axis=-1)

        return ssim_score
    

if __name__ == "__main__":
    from screenshot_capture import take_screenshot_from_url
    # Load two sample images
    ref_html_path = '/media/chung/Data/FPT/baseline/design2code-baseline/utils/graph_utils/testcases/design2code_1.html'
    hyp_html_path = '/media/chung/Data/FPT/baseline/design2code-baseline/utils/graph_utils/testcases/design2code_9.html'


    ref_img_path = "./graphics/reference.png"
    hyp_img_path = "./graphics/hypothesis.png"
    take_screenshot_from_url(ref_html_path, ref_img_path)
    take_screenshot_from_url(hyp_html_path, hyp_img_path)

    mse_metric = MSEMetrics(masked = False)
    maksed_mse_metric = MSEMetrics(masked = True)
    iou_metric = IoUMetrics()
    ssim_metric = SSIMMetrics()

    # Compute MSE
    mse_value = mse_metric(ref_img_path, hyp_img_path)
    print(f"MSE: {mse_value}")

    maksed_mse_value = maksed_mse_metric(ref_img_path, hyp_img_path)
    print(f"Masked MSE: {maksed_mse_value}")


    # Compute IoU
    iou_value = iou_metric(ref_img_path, hyp_img_path)
    print(f"IoU: {iou_value}")

    # Compute SSIM
    ssim_value = ssim_metric(ref_img_path, hyp_img_path)
    print(f"SSIM: {ssim_value}")



