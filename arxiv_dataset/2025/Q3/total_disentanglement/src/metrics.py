"""
Simple metrics for font generation evaluation.
"""

import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class MSE:
    """Mean Squared Error metric."""
    
    def calc_dist(self, img1, img2):
        """Calculate MSE between two images."""
        img1_np = img1.cpu().numpy() if torch.is_tensor(img1) else img1
        img2_np = img2.cpu().numpy() if torch.is_tensor(img2) else img2
        return np.mean((img1_np - img2_np) ** 2)


class L1:
    """L1 (Mean Absolute Error) metric."""
    
    def calc_dist(self, img1, img2):
        """Calculate L1 distance between two images."""
        img1_np = img1.cpu().numpy() if torch.is_tensor(img1) else img1
        img2_np = img2.cpu().numpy() if torch.is_tensor(img2) else img2
        return np.mean(np.abs(img1_np - img2_np))


class IoU:
    """Intersection over Union metric."""
    
    def calc_dist(self, img1, img2, threshold=0.5):
        """Calculate IoU distance (1 - IoU) between two binary images."""
        img1_np = img1.cpu().numpy() if torch.is_tensor(img1) else img1
        img2_np = img2.cpu().numpy() if torch.is_tensor(img2) else img2
        
        # Binarize images
        binary1 = (img1_np > threshold).astype(np.uint8)
        binary2 = (img2_np > threshold).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.sum(binary1 & binary2)
        union = np.sum(binary1 | binary2)
        
        if union == 0:
            return 0.0  # Both images are empty
        
        iou = intersection / union
        return 1.0 - iou  # Return distance (lower is better)


class max_hausdorff_dist:
    """Maximum Hausdorff distance metric."""
    
    def calc_dist(self, img1, img2, threshold=0.5):
        """Calculate maximum Hausdorff distance between edge points."""
        try:
            img1_np = img1.cpu().numpy() if torch.is_tensor(img1) else img1
            img2_np = img2.cpu().numpy() if torch.is_tensor(img2) else img2
            
            # Get edge points (points where pixel value > threshold)
            points1 = np.column_stack(np.where(img1_np > threshold))
            points2 = np.column_stack(np.where(img2_np > threshold))
            
            if len(points1) == 0 or len(points2) == 0:
                return 1.0  # Maximum distance if one image is empty
            
            # Calculate directed Hausdorff distances
            dist1 = directed_hausdorff(points1, points2)[0]
            dist2 = directed_hausdorff(points2, points1)[0]
            
            # Return maximum of the two directed distances
            return max(dist1, dist2)
        
        except Exception:
            # Fallback to MSE if Hausdorff calculation fails
            return np.mean((img1_np - img2_np) ** 2)


class chamfer_dist:
    """Chamfer distance metric."""
    
    def calc_dist(self, img1, img2, threshold=0.5):
        """Calculate Chamfer distance between edge points."""
        try:
            img1_np = img1.cpu().numpy() if torch.is_tensor(img1) else img1
            img2_np = img2.cpu().numpy() if torch.is_tensor(img2) else img2
            
            # Get edge points
            points1 = np.column_stack(np.where(img1_np > threshold))
            points2 = np.column_stack(np.where(img2_np > threshold))
            
            if len(points1) == 0 or len(points2) == 0:
                return 1.0  # Maximum distance if one image is empty
            
            # Calculate minimum distances from each point in set1 to set2
            dist1_to_2 = []
            for p1 in points1:
                distances = np.sqrt(np.sum((points2 - p1) ** 2, axis=1))
                dist1_to_2.append(np.min(distances))
            
            # Calculate minimum distances from each point in set2 to set1
            dist2_to_1 = []
            for p2 in points2:
                distances = np.sqrt(np.sum((points1 - p2) ** 2, axis=1))
                dist2_to_1.append(np.min(distances))
            
            # Chamfer distance is the mean of all minimum distances
            chamfer = (np.mean(dist1_to_2) + np.mean(dist2_to_1)) / 2.0
            return chamfer
        
        except Exception:
            # Fallback to MSE if Chamfer calculation fails
            return np.mean((img1_np - img2_np) ** 2)