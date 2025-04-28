import numpy as np
import skimage

def get_recon_metrics(gt, recon):
    """
    This function calculates the following metrics for a given reconstruction with ground truth
       - NRMSE
       - PSNR
       - SSIM

     Args:
        gt (np.ndarray): N x M or N x M x C ground truth image (dtype: float)
        recon (np.ndarray): N x M or N x M x C reconstructed image (dtype: float)

    Returns: 
        dict: Dictonary containing above metrics
    """
    if gt.ndim == 2:
        gt = gt[:, :, None]
        recon = recon[:, :, None]

    metrics_dict = {}
    metrics_dict["NRMSE"] = mse(gt, recon)[2]
    metrics_dict["PSNR"] = psnr(gt, recon)
    metrics_dict["SSIM"] = ssim(gt, recon)

    return metrics_dict

def mse(gt, recon):
    """
    Calculate MSI, RMSE, and NRMSE between ground truth and reconstruction.

    Args:
        gt (np.ndarray): N x M x C ground truth image
        recon (np.ndarray): N x M x C reconstructed image

    Returns: 
        list of float32: [MSE, RMSE, NRMSE]
    """
    recon_vec = recon.reshape(-1)
    gt_vec = gt.reshape(-1)
    mse = np.mean(np.power(recon_vec - gt_vec, 2))
    rmse = np.sqrt(mse)
    nrmse = rmse / np.sqrt(np.mean(np.power(gt_vec, 2)))
    return [mse, rmse, nrmse]

def psnr(gt, recon):
    """
    This function computes the Peak Signal to Noise Ratio relative to the maximum pixel value of the ground truth..

     Args:
        gt (np.ndarray): N x M x C ground truth image (dtype: float)
        recon (np.ndarray): N x M x C reconstructed image (dtype: float)

    Returns: 
        float32: PSNR
    """
    MSE = mse(gt, recon)[0]
    out = 10 * np.log10(np.max(gt) ** 2 / MSE)
    return out

def ssim(gt, recon, k_1=0.01, k_2=0.03):
    """
    Compute the mean structural similarity index over the image using the skimage implementation

    Args:
        gt (np.ndarray): N x M x C ground truth image (dtype: float)
        recon (np.ndarray): N x M x C reconstructed image (dtype: float)

    Returns: 
        float32: SSIM
    """
    gt_range = np.max(gt) - np.min(gt)
    ssim = skimage.metrics.structural_similarity(gt, recon, channel_axis=2, data_range=gt_range, K1=k_1, K2=k_2)

    return ssim