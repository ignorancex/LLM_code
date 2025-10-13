#!/usr/bin/env python
# coding=utf-8
"""
Enhanced Loss Functions for Pan-sharpening
@Description: Modernized and fixed version of loss functions from utils.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GANLoss(nn.Module):
    """GAN Loss for adversarial training"""
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = input.new_tensor(self.real_label).expand_as(input)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = input.new_tensor(self.fake_label).expand_as(input)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class NewLoss(nn.Module):
    """Enhanced version of newLoss with better numerical stability"""
    def __init__(self, r1=1.1, r2=1000, offset=0.9995):
        super(NewLoss, self).__init__()
        self.r1 = r1
        self.r2 = r2
        self.offset = offset
        print("Enhanced NewLoss created")

    def forward(self, input, target):
        del_x = torch.abs(input - target)
        # Add numerical stability
        del_x = torch.clamp(del_x, min=1e-8)
        
        # Normalize to 0-1 range if needed
        if input.max() > 1.0:
            del_x = del_x / 255.0
        
        # Enhanced loss computation with better numerical stability
        y = ((del_x * 255) ** self.r1 / 255) * del_x
        return torch.mean(y)  # Use mean instead of sum for better scaling


class MEF_SSIM_Loss(nn.Module):
    """Multi-Exposure Fusion SSIM Loss"""
    def __init__(self, window_size=11, size_average=True):
        super(MEF_SSIM_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=(2, 3))

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.window is None or channel != self.channel:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        else:
            window = self.window
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class VGGLoss(nn.Module):
    """VGG Perceptual Loss - Fixed version compatible with 4-channel input"""
    def __init__(self, layer_type='22', rgb_range=255):
        super(VGGLoss, self).__init__()
        self.layer_type = layer_type
        self.rgb_range = rgb_range
        self.use_vgg = False
        
        try:
            # Try to use the VGG from utils.vgg first
            from utils.vgg import VGG
            self.vgg = VGG(layer_type, rgb_range=1)  # VGG expects [0,1] range
            self.use_vgg = True
            print(f"VGG{layer_type} loss initialized successfully")
        except ImportError:
            print("Warning: utils.vgg not available, using MSE fallback")
            self.loss = nn.MSELoss()
            self.use_vgg = False
    
    def forward(self, pred, target):
        if not self.use_vgg:
            # Fallback to MSE if VGG not available
            return self.loss(pred, target)
        
        # Handle 4-channel input by converting to 3-channel RGB
        if pred.size(1) == 4:
            # Convert 4-channel MS to 3-channel RGB (take first 3 channels)
            pred_rgb = pred[:, :3, :, :].contiguous()  
            target_rgb = target[:, :3, :, :].contiguous()
        elif pred.size(1) == 1:
            # Convert grayscale to RGB by repeating
            pred_rgb = pred.repeat(1, 3, 1, 1)
            target_rgb = target.repeat(1, 3, 1, 1)
        else:
            pred_rgb = pred
            target_rgb = target
            
        # Ensure input is in [0, 1] range for VGG
        if pred_rgb.max() > 1.1:  # Allow some tolerance for floating point precision
            pred_rgb = pred_rgb / self.rgb_range
            target_rgb = target_rgb / self.rgb_range
        
        # Clamp to [0, 1] range to be safe
        pred_rgb = torch.clamp(pred_rgb, 0, 1)
        target_rgb = torch.clamp(target_rgb, 0, 1)
        
        return self.vgg(pred_rgb, target_rgb)


# Additional custom loss functions from unisolver.py
class SSIMLoss(nn.Module):
    """SSIM Loss implementation"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=(2, 3))  # Average over spatial dimensions only

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.window is None or channel != self.channel:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        else:
            window = self.window
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG features"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        try:
            # Use pretrained VGG16 for perceptual features
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
            self.features = nn.Sequential(*list(vgg.features)[:16])  # Up to relu3_3
            self.features.eval()
            for param in self.features.parameters():
                param.requires_grad = False
            self.loss = nn.MSELoss()
            self.use_vgg = True
        except ImportError:
            print("Warning: torchvision not available, using MSE as perceptual loss")
            self.loss = nn.MSELoss()
            self.use_vgg = False
        
    def forward(self, pred, target):
        if not self.use_vgg:
            # Fallback to MSE if VGG not available
            return self.loss(pred, target)
        
        # Ensure input is 3-channel for VGG
        if pred.size(1) == 4:
            pred_rgb = pred[:, :3, :, :]  # Take first 3 channels
            target_rgb = target[:, :3, :, :]
        else:
            pred_rgb = pred
            target_rgb = target
            
        # Normalize to [0, 1] range if needed
        if pred_rgb.max() > 1.0:
            pred_rgb = pred_rgb / 255.0
            target_rgb = target_rgb / 255.0
        
        pred_features = self.features(pred_rgb)
        target_features = self.features(target_rgb)
        return self.loss(pred_features, target_features)


class FrequencyLoss(nn.Module):
    """Frequency domain loss using FFT"""
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, pred, target):
        # Convert to frequency domain
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Compute loss in frequency domain
        return self.loss(torch.abs(pred_fft), torch.abs(target_fft))


class GradientLoss(nn.Module):
    """Gradient Loss"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, pred, target):
        # Compute gradients
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        return self.loss(pred_grad_x, target_grad_x) + self.loss(pred_grad_y, target_grad_y)


class EdgeLoss(nn.Module):
    """Edge Loss using Sobel operator"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.loss = nn.L1Loss()
        
        # Sobel kernels - register as buffers for automatic device handling
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, pred, target):
        # Convert to grayscale if multi-channel
        if pred.size(1) > 1:
            pred_gray = torch.mean(pred, dim=1, keepdim=True)
            target_gray = torch.mean(target, dim=1, keepdim=True)
        else:
            pred_gray = pred
            target_gray = target
            
        # Apply Sobel operator - sobel kernels are automatically on correct device
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return self.loss(pred_edge, target_edge)


def make_loss(loss_type):
    """Enhanced loss factory function prioritizing utils.utils.py losses"""
    # Add focal loss mapping (maps to NewLoss which is focal loss for regression)
    if loss_type.upper() == "FOCAL":
        loss_type = "newloss"
    
    # First try to import and use utils.utils.py make_loss
    try:
        from utils.utils import make_loss as utils_make_loss
        try:
            loss = utils_make_loss(loss_type)
            print(f"Using verified loss from utils.utils: {loss_type}")
            return loss
        except (ValueError, Exception) as e:
            print(f"utils.utils.make_loss failed for {loss_type}: {e}")
            # Fall through to enhanced implementation
    except ImportError:
        print("Warning: Could not import utils.utils.make_loss, using enhanced implementation")
    
    # Enhanced loss factory as fallback
    loss_type = loss_type.upper()
    
    if loss_type == "MSE":
        return nn.MSELoss(reduction='mean')
    elif loss_type == "L1":
        return nn.L1Loss(reduction='mean')
    elif loss_type == "MEF_SSIM":
        return MEF_SSIM_Loss()
    elif loss_type in ["VGG22", "VGG54"]:
        layer_type = loss_type[3:]  # Extract "22" or "54"
        return VGGLoss(layer_type, rgb_range=255)
    elif loss_type == "NEWLOSS":
        return NewLoss()
    elif loss_type == "CE":
        return nn.BCELoss()
    elif loss_type == "GAN":
        return GANLoss()
    elif loss_type == "SMOOTH_L1":
        return nn.SmoothL1Loss()
    elif loss_type == "HUBER":
        return nn.HuberLoss()
    # Custom advanced losses
    elif loss_type == "FREQUENCY":
        return FrequencyLoss()
    elif loss_type == "GRADIENT":
        return GradientLoss()
    elif loss_type == "EDGE":
        return EdgeLoss()
    elif loss_type == "PERCEPTUAL":
        return PerceptualLoss()
    elif loss_type == "SSIM":
        return SSIMLoss()
    else:
        print(f"Warning: Unknown loss type {loss_type}, defaulting to L1Loss")
        return nn.L1Loss(reduction='mean')


def get_available_losses():
    """Get list of available loss types"""
    verified_losses = ["MSE", "L1", "MEF_SSIM", "VGG22", "VGG54", "NEWLOSS", "CE", "GAN"]
    custom_losses = ["FREQUENCY", "GRADIENT", "EDGE", "PERCEPTUAL", "SSIM", "SMOOTH_L1", "HUBER"]
    return {
        "verified": verified_losses,  # From utils.utils.py
        "custom": custom_losses,      # Enhanced implementations
        "all": verified_losses + custom_losses
    }


# For backward compatibility
class newLoss(NewLoss):
    """Backward compatibility alias"""
    pass
