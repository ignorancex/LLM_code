"""
Metrics utilities for StyleGAN fingerprinting evaluation.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from scipy import linalg



def save_metrics_text(metrics, output_dir):
    """
    Save metrics to a text file.
    
    Args:
        metrics (dict): Dictionary of metrics
        output_dir (str): Output directory
    """
    metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_IDX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling features
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_IDX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        # Load pretrained model
        self.inception = models.inception_v3(pretrained=True)
        self.inception.aux_logits = False
        self.inception.fc = nn.Identity()

        # Block definitions
        self.blocks = nn.ModuleList()
        
        # Block 0: input to maxpool1
        block0 = [
            self.inception.Conv2d_1a_3x3,
            self.inception.Conv2d_2a_3x3,
            self.inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                self.inception.Conv2d_3b_1x1,
                self.inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                self.inception.Mixed_5b,
                self.inception.Mixed_5c,
                self.inception.Mixed_5d,
                self.inception.Mixed_6a,
                self.inception.Mixed_6b,
                self.inception.Mixed_6c,
                self.inception.Mixed_6d,
                self.inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                self.inception.Mixed_7a,
                self.inception.Mixed_7b,
                self.inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = nn.functional.interpolate(x,
                                        size=(299, 299),
                                        mode='bilinear',
                                        align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def calculate_activation_statistics(images, model, batch_size=50, dims=2048, device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Tensor of images, of shape (N,C,H,W) and in range [0, 1]
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                    batch size batch_size. A reasonable batch size
                    depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    model.eval()
    
    n_batches = (images.size(0) + batch_size - 1) // batch_size
    n_used_imgs = n_batches * batch_size
    
    pred_arr = np.empty((n_used_imgs, dims))
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, images.size(0))
        
        batch = images[start:end].to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        
        pred_arr[start:end] = pred.cpu().numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(pred_arr[:images.size(0)], axis=0)
    sigma = np.cov(pred_arr[:images.size(0)], rowvar=False)
    
    return mu, sigma


def calculate_fid(images1, images2, batch_size=50, device='cpu'):
    """Calculate FID between two sets of images.
    
    Args:
        images1: First set of images, tensor of shape (N,C,H,W) in range [0,1]
        images2: Second set of images, tensor of shape (N,C,H,W) in range [0,1]
        batch_size: Batch size for processing
        device: Device to run calculations on
        
    Returns:
        FID score
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    
    mu1, sigma1 = calculate_activation_statistics(images1, model, batch_size, device=device)
    mu2, sigma2 = calculate_activation_statistics(images2, model, batch_size, device=device)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return float(fid) 


def extract_inception_features(images, batch_size=50, device='cpu'):
    """Extract features from the InceptionV3 model for FID calculation.
    
    Args:
        images: Images tensor of shape (N,C,H,W) in range [0,1]
        batch_size: Batch size for processing
        device: Device to run calculations on
        
    Returns:
        Features array of shape (N, 2048)
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    
    n_batches = (images.size(0) + batch_size - 1) // batch_size
    n_used_imgs = n_batches * batch_size
    
    pred_arr = np.empty((n_used_imgs, 2048))
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, images.size(0))
        
        batch = images[start:end].to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        
        pred_arr[start:end] = pred.cpu().numpy().reshape(pred.size(0), -1)
    
    return pred_arr[:images.size(0)]


def compute_fid_from_features(features1, features2):
    """Compute FID score from pre-extracted features.
    
    Args:
        features1: Features from first set of images, array of shape (N, 2048)
        features2: Features from second set of images, array of shape (N, 2048)
        
    Returns:
        FID score
    """
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    
    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return float(fid) 