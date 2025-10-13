from CODE.Model.inception import *
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from CODE.Model.models import *

class DefenseModule(nn.Module):
    def __init__(self, device):
        super(DefenseModule, self).__init__()
        self.defense_methods = ['jitter', 'random_zero', 'segment_zero', 'gaussian_noise', 'smooth', 'nothing']
        self.method_usage_count = {method: 0 for method in self.defense_methods}  
        self.device = device

    def forward(self, x):
        method = random.choice(self.defense_methods)
        self.method_usage_count[method] += 1
        if method == 'jitter':
            return self.jitter(x)
        elif method == 'random_zero':
            return self.RandomZero(x)
        elif method == 'segment_zero':
            return self.SegmentZero(x)
        elif method == 'gaussian_noise':
            return self.gaussian_noise(x)
        elif method == 'smooth':
            return self.smooth_time_series(x)
        elif method == 'nothing':
            return self.nothing(x)

    def jitter(self, x, p=0.75, noise_level=1):
        mask = torch.rand(x.shape, device=self.device) < p

        noise = (torch.randint(0, 2, x.shape, dtype=x.dtype, device=self.device) * 2 - 1) 
        noise = noise * noise_level 

        x = (x + noise * mask.float()).to(self.device)  
        return x

    def RandomZero(self, x, p=0.5):
        mask = torch.rand(x.shape, device=self.device) < p

        x_masked = x * (~mask)
        
        return x_masked

    def SegmentZero(self, x, total_zero_fraction=0.25, max_segment_fraction=0.05):
        total_length = x.size(-1)
        max_segment_length = max(int(total_length * max_segment_fraction), 1)
        total_zero_length = int(total_length * total_zero_fraction)
        
        mask = torch.ones_like(x)

        zeroed_length = 0
        
        while zeroed_length < total_zero_length:
            segment_length = min(torch.randint(1, max_segment_length + 1, (1,)).item(), total_zero_length - zeroed_length)

            start = torch.randint(0, total_length - segment_length + 1, (1,)).item()
            
            mask[..., start:start+segment_length] = 0

            zeroed_length += segment_length
        
        x_masked = x * mask
        
        return x_masked

    def gaussian_noise(self, x, mean=0, std=0.3):
        noise = torch.randn_like(x) * std + mean
        
        x_noisy = x + noise
        
        return x_noisy

    def gaussian_kernel(self, kernel_size=10, sigma=5):
        """Construct a 1D Gaussian kernel."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_range = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        kernel = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size)
        return kernel

    def smooth_time_series(self, x, kernel_size=10, sigma=5):
        """Smooth the time series with a Gaussian kernel."""
        kernel = self.gaussian_kernel(kernel_size, sigma).to(self.device)
        
        # Ensure x is in the format [batch_size, 1, sample_length]
        if len(x.shape) == 1:
            x = x.view(1, 1, -1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        padding_size = kernel_size // 2
        start_mean = x[:, :, :padding_size].mean(dim=-1, keepdim=True)
        end_mean = x[:, :, -padding_size:].mean(dim=-1, keepdim=True)
        
        start_padding = start_mean.expand(-1, -1, padding_size)
        end_padding = end_mean.expand(-1, -1, padding_size)
        
        padded_x = torch.cat([start_padding, x, end_padding], dim=-1)
        
        smoothed_x = F.conv1d(padded_x, kernel, padding=0)
        
        # Optionally, adjust the length of smoothed_x as needed.
        # smoothed_x = 0.5 * (smoothed_x[:, :, :-1] + smoothed_x[:, :, 1:])
        
        # Remove added dimensions if the input was 1D.
        if len(x.shape) == 2:
            smoothed_x = smoothed_x.squeeze(1)
        if len(x.shape) == 1:
            smoothed_x = smoothed_x.view(-1)
        
        return smoothed_x

    def nothing(self, x):
        return x

    def scaling(x, sigma=0.1):
    
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
        return np.multiply(x, factor[:,np.newaxis,:])

    def time_warp(x, y, sigma=0.2, knot=4):
        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(x.shape[1])
        
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
        warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
        
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (x.shape[1]-1)/time_warp[-1]
                ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
        return ret
        
    def print_method_usage(self):
        print("Defense methods usage count:")
        for method, count in self.method_usage_count.items():
            print(f"{method}: {count}")

class Classifier_INCEPTION_with_Defense(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Defense, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense(x)  
        x = self.classifier(x)
        return x

class LSTMFCN_with_Defense(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(LSTMFCN_with_Defense, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = LSTMFCN(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense(x)  
        x = self.classifier(x)
        return x

class ClassifierResNet18_with_Defense(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(ClassifierResNet18_with_Defense, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = ClassifierResNet18(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_Defense2(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Defense2, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense(x) 
        x = self.defense(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_Defense3(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Defense3, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense(x)  
        x = self.defense(x)
        x = self.defense(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_Jitter(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Jitter, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.classifier(x)
        return x

class LSTMFCN_with_Jitter(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(LSTMFCN_with_Jitter, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = LSTMFCN(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.classifier(x)
        return x

class ClassifierResNet18_with_Jitter(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(ClassifierResNet18_with_Jitter, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = ClassifierResNet18(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_RandomZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_RandomZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.RandomZero(x) 
        x = self.classifier(x)
        return x

class LSTMFCN_with_RandomZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(LSTMFCN_with_RandomZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = LSTMFCN(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.RandomZero(x)  
        x = self.classifier(x)
        return x

class ClassifierResNet18_with_RandomZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(ClassifierResNet18_with_RandomZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = ClassifierResNet18(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.RandomZero(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_SegmentZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_SegmentZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.SegmentZero(x) 
        x = self.classifier(x)
        return x

class LSTMFCN_with_SegmentZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(LSTMFCN_with_SegmentZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = LSTMFCN(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.SegmentZero(x)  
        x = self.classifier(x)
        return x

class ClassifierResNet18_with_SegmentZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(ClassifierResNet18_with_SegmentZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = ClassifierResNet18(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.SegmentZero(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_gaussian_noise(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_gaussian_noise, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.gaussian_noise(x) 
        x = self.classifier(x)
        return x

class LSTMFCN_with_gaussian_noise(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(LSTMFCN_with_gaussian_noise, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = LSTMFCN(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.gaussian_noise(x)
        x = self.classifier(x)
        return x

class ClassifierResNet18_with_gaussian_noise(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(ClassifierResNet18_with_gaussian_noise, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = ClassifierResNet18(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.gaussian_noise(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_smooth_time_series(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_smooth_time_series, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.smooth_time_series(x) 
        x = self.classifier(x)
        return x

class LSTMFCN_with_smooth_time_series(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(LSTMFCN_with_smooth_time_series, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = LSTMFCN(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.smooth_time_series(x)  
        x = self.classifier(x)
        return x

class ClassifierResNet18_with_smooth_time_series(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(ClassifierResNet18_with_smooth_time_series, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = ClassifierResNet18(input_shape=1, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.smooth_time_series(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_Jitter_logits(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Jitter_logits, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION_Logits(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_RandomZero_logits(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_RandomZero_logits, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION_Logits(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.RandomZero(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_SegmentZero_logits(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_SegmentZero_logits, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION_Logits(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.SegmentZero(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_gaussian_noise_logits(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_gaussian_noise_logits, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION_Logits(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.gaussian_noise(x) 
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_smooth_time_series_logits(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_smooth_time_series_logits, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION_Logits(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.smooth_time_series(x)  
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_Jitter2(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Jitter2, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.defense.jitter(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_RandomZero2(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_RandomZero2, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.RandomZero(x)  
        x = self.defense.RandomZero(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_gaussian_noise2(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_gaussian_noise2, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.gaussian_noise(x) 
        x = self.defense.gaussian_noise(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_SegmentZero2(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_SegmentZero2, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.SegmentZero(x) 
        x = self.defense.SegmentZero(x)
        x = self.classifier(x)
        return x
    
class Classifier_INCEPTION_with_smooth_time_series2(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_smooth_time_series2, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.smooth_time_series(x)  
        x = self.defense.smooth_time_series(x)
        x = self.classifier(x)
        return x
    
class Classifier_INCEPTION_with_JitterRandomZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_JitterRandomZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.defense.RandomZero(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_JitterSegmentZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_JitterSegmentZero, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.defense.SegmentZero(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_Jittergaussian_noise(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Jittergaussian_noise, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.defense.gaussian_noise(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_Jittersmooth_time_series(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_Jittersmooth_time_series, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.jitter(x)  
        x = self.defense.smooth_time_series(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_RandomZeroJitter(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_RandomZeroJitter, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.RandomZero(x)  
        x = self.defense.jitter(x)
        x = self.classifier(x)
        return x

class Classifier_INCEPTION_with_RandomZeroSegmentZero(nn.Module):
    def __init__(self, input_shape, nb_classes, device):
        super(Classifier_INCEPTION_with_RandomZeroJitter, self).__init__()
        self.defense = DefenseModule(device)
        self.classifier = Classifier_INCEPTION(input_shape=input_shape, nb_classes=nb_classes)

    def forward(self, x):
        x = self.defense.RandomZero(x)  
        x = self.defense.SegmentZero(x)
        x = self.classifier(x)
        return x
