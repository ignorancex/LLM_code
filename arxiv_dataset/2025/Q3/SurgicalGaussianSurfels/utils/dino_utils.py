import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import timm


class DINOFeatureExtractor:
    """DINO feature extractor for perceptual loss computation"""
    
    def __init__(self, model_name='vit_base_patch16_224', device='cuda'):
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """Extract DINO features from an image"""
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 3:
                image = image.permute(1, 2, 0)  # CHW -> HWC
            image = (image * 255).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(image.cpu().numpy())
        
        # Apply transformation
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
        
        return features.squeeze(0)


def extract_dino_features_offline(image_dir, output_dir):
    """Extract DINO features for all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = DINOFeatureExtractor()
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    print(f"Extracting DINO features for {len(image_files)} images...")
    
    for filename in tqdm(image_files):
        img_path = os.path.join(image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        features = extractor.extract_features(image)
        
        # Save features
        base_name = os.path.splitext(filename)[0]
        feature_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(feature_path, features.cpu().numpy())
    
    print(f"DINO features saved to {output_dir}")


def load_dino_features(feature_dir, image_name):
    """Load pre-extracted DINO features"""
    base_name = os.path.splitext(image_name)[0]
    feature_path = os.path.join(feature_dir, f"{base_name}.npy")
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"DINO features not found for {image_name}")
    
    features = np.load(feature_path)
    return torch.from_numpy(features).float()


def dino_perceptual_loss(pred_features, gt_features):
    """Compute perceptual loss using DINO features"""
    if pred_features.dim() == 1:
        pred_features = pred_features.unsqueeze(0)
    if gt_features.dim() == 1:
        gt_features = gt_features.unsqueeze(0)
    
    # Normalize features
    pred_features = F.normalize(pred_features, dim=-1)
    gt_features = F.normalize(gt_features, dim=-1)
    
    # Compute cosine similarity loss
    loss = 1 - F.cosine_similarity(pred_features, gt_features, dim=-1).mean()
    
    return loss 