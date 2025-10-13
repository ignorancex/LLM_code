import os
import numpy as np 
import nibabel as nib

def probHeterogeneous(attributes):
    """
    Assigns inverse-frequency-based probabilities to each value in attributes.

    ARGS INPUT: 
        attributes: conditions 

    ARGS OUTPUT: 
        weights: weights to balance the pick rate of the heterogeneous dataset
    """
    
    arr = np.round(attributes)
    
    unique_vals, counts = np.unique(arr, return_counts=True)

    print('CLASSES:', unique_vals)
    print('COUNTS: ', counts)

    weights_per_class = 1 / counts
    weights_per_class /= weights_per_class.sum()

    weights = np.array([weights_per_class[np.where(unique_vals == val)[0][0]] for val in arr])

    return weights / weights.sum()

def save_nifti_image(
    image,
    output_path,
    target_shape,
    affine=None
    ):
    """
    Save image(s) as NIfTI files, handling flexible input shapes.
    Used in inference script.

    ARGS INPUT:
        image: np.ndarray or torch.Tensor
        output_path: base file path (e.g., 'out/image.nii.gz')
        target_shape: expected (H, W, D)
        affine: 4x4 affine matrix (default = identity)
    """

    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()

    image = np.array(image)
    shape = image.shape

    # Normalize shape to (B, H, W, D, C)
    if len(shape) == 3:
        image = image[np.newaxis, ..., np.newaxis]
    elif len(shape) == 4:
        if shape[-1] == target_shape[-1]:  
            image = image[np.newaxis, ...]
        else:  
            image = image[..., np.newaxis]
    elif len(shape) == 5:
        pass  
    else:
        raise ValueError(f"Unsupported input shape: {shape}")

    B, H, W, D, C = image.shape

    if (H, W, D) != tuple(target_shape):
        raise ValueError(f"Expected spatial shape {target_shape}, but got {(H, W, D)}")

    if affine is None:
        affine = np.eye(4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for b in range(B):
        volume = image[b]  
        if C == 1:
            nib_img = nib.Nifti1Image(np.squeeze(volume), affine)
            nib.save(nib_img, output_path)
        else:
            nib_img = nib.Nifti1Image(volume, affine)  
            nib.save(nib_img, output_path)
