import numpy as np
from python_color_transfer.color_transfer import ColorTransfer
from PIL import Image, ImageOps
import cv2
from joblib import Parallel, delayed
from albumentations.core.transforms_interface import ImageOnlyTransform

PT = ColorTransfer()


# get reference image
def open_img(img_path):
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = np.array(image)
    return image

def get_sample_reference():
    img_path = "/data/throat_instance_segmentation/images/018-0079/018-0079-01-11.png"
    image = open_img(img_path)
    return image

def get_sample_references():
    paths = [
       "/data/throat_instance_segmentation/images/018-0079/018-0079-01-11.png",
    ]
    images = [open_img(img_path) for img_path in paths]
    return images


# LAB mean Color Transfer
def color_transfer_lab(source_img, ref_img):
    modified_img = PT.lab_transfer(img_arr_in=source_img, img_arr_ref=ref_img)
    return modified_img


####################################
# Unused (experimental)
####################################


# Linear Monge-Kantorovitch - MKL Color transfer
EPS = 2.2204e-16

def MKL(A, B):
    Da2, Ua = np.linalg.eig(A)

    Da2 = np.diag(Da2)
    Da2[Da2 < 0] = 0
    Da = np.sqrt(Da2 + EPS)
    C = Da @ np.transpose(Ua) @ B @ Ua @ Da
    Dc2, Uc = np.linalg.eig(C)

    Dc2 = np.diag(Dc2)
    Dc2[Dc2 < 0] = 0
    Dc = np.sqrt(Dc2 + EPS)
    Da_inv = np.diag(1 / (np.diag(Da)))
    T = Ua @ Da_inv @ Uc @ Dc @ np.transpose(Uc) @ Da_inv @ np.transpose(Ua)
    return T

def color_transfer_MKL(source, target):
    assert len(source.shape) == 3, 'Images should have 3 dimensions'
    assert source.shape[-1] == 3, 'Images should have 3 channels'
    X0 = np.reshape(source, (-1, 3), 'F')
    X1 = np.reshape(target, (-1, 3), 'F')
    A = np.cov(X0, rowvar=False)
    B = np.cov(X1, rowvar=False)
    T = MKL(A, B)
    mX0 = np.mean(X0, axis=0)
    mX1 = np.mean(X1, axis=0)
    XR = (X0 - mX0) @ T + mX1
    IR = np.reshape(XR, source.shape, 'F')
    IR = np.real(IR)
    IR[IR > 1] = 1
    IR[IR < 0] = 0
    return IR


def rgb_to_lab(image):
    """Convert an image from RGB to LAB color space."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def lab_to_rgb(image):
    """Convert an image from LAB to RGB color space."""
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)


def sliced_wasserstein(source, target, n_projections=100):
    """
    Compute the Sliced Wasserstein Distance and transport plan.

    Parameters:
        source (ndarray): Source points (Nx3).
        target (ndarray): Target points (Mx3).
        n_projections (int): Number of random projections.

    Returns:
        transported_source (ndarray): Source points aligned to target.
    """
    source = source.astype(np.float32)
    target = target.astype(np.float32)

    # Generate random projections and normalize them
    projections = np.random.randn(n_projections, source.shape[1]).astype(np.float32)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    def project_and_sort(projection):
        # Project and sort source and target along the projection
        proj_source = np.dot(source, projection)
        proj_target = np.dot(target, projection)
        sorted_idx_source = np.argsort(proj_source)
        sorted_idx_target = np.argsort(proj_target)
        return sorted_idx_source, sorted_idx_target

    # Use parallel processing for independent projections
    results = Parallel(n_jobs=-1)(
        delayed(project_and_sort)(proj) for proj in projections
    )

    # Accumulate the transported source
    transported_source = np.zeros_like(source)
    for (sorted_idx_source, sorted_idx_target) in results:
        transported_source[sorted_idx_source] += source[sorted_idx_target]

    # Average over projections
    transported_source /= n_projections
    return transported_source

def color_transfer_sot(source_img, target_img, n_projections=100):
    """
    Perform color transfer from target_img to source_img using SOT.

    Parameters:
        source_img (ndarray): Source image (HxWx3).
        target_img (ndarray): Target image (HxWx3).
        n_projections (int): Number of projections for Sliced Wasserstein.

    Returns:
        result_img (ndarray): Color-transferred image.
    """

    # Ensure same shape
    target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))

    # Convert images to LAB color space
    source_lab = rgb_to_lab(source_img)
    target_lab = rgb_to_lab(target_img)

    # Reshape into Nx3
    source_pixels = source_lab.reshape(-1, 3)
    target_pixels = target_lab.reshape(-1, 3)

    # Apply Sliced Wasserstein
    transferred_pixels = sliced_wasserstein(source_pixels, target_pixels, n_projections)

    # Reshape back to original shape and convert to uint8
    transferred_lab = np.clip(transferred_pixels.reshape(source_lab.shape), 0, 255).astype(np.uint8)

    # Convert back to RGB
    result_img = lab_to_rgb(transferred_lab)
    return result_img


class LABColorTransferAugmentation(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        """
        LAB Color Transfer augmentation for Albumentations.

        Parameters:
        - always_apply: bool, whether to always apply the augmentation.
        - p: float, probability of applying the augmentation.
        """
        super(LABColorTransferAugmentation, self).__init__(always_apply=always_apply, p=p)
        self.reference_image = get_sample_reference()

    def apply(self, image, **params):
        return color_transfer_lab(image, self.reference_image)