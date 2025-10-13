import numpy as np
from tqdm import tqdm
from encoder import enc_preprocess
from PIL import Image
import os

def compute_lipschitz_vectorized(x, y, num_samples):
    """
    Args:
        x, y: Input images as numpy arrays (H,W,C)
        num_samples: Number of point pairs to sample
    Returns:
        Maximum Lipschitz constant ratio 
    """
    # Flatten the arrays to create lists of 3D points
    x_flat = x.reshape(-1, 3)
    y_flat = y.reshape(-1, 3)

    # Sample random point pairs
    indices = np.random.choice(len(x_flat), (num_samples, 2), replace=True)
    
    # Compute pairwise distances
    dist_x = np.linalg.norm(x_flat[indices[:, 0]] - x_flat[indices[:, 1]], axis=1)
    dist_y = np.linalg.norm(y_flat[indices[:, 0]] - y_flat[indices[:, 1]], axis=1)
    
    # Compute Lipschitz constants and avoid division by zero
    lipschitz_values = np.divide(dist_y, dist_x, out=np.zeros_like(dist_y), where=dist_x != 0)

    return np.max(lipschitz_values)

if __name__ == "__main__":
    num_samples = 50000  
    path_dir = "V7_encoder_epoch_700000"  # Your directory name
    print("\n",path_dir)

    # Setup paths
    stylization_path = f"data/results_unsplash/{path_dir}/" # YOUR DIR NAME
    content_path = "data/test_imgs/content/" # YOUR DIR NAME
    
    # Get sorted lists of image paths
    stylization_results = sorted([
        os.path.join(stylization_path, f) 
        for f in os.listdir(stylization_path) 
        if f.endswith('.png')
    ])
    
    content_images = sorted([
        os.path.join(content_path, f) 
        for f in os.listdir(content_path) 
        if f.endswith('.png')
    ])
    
    lipschitz_constants = []
    for idx in tqdm(range(len(stylization_results))):
        x = enc_preprocess(Image.open(stylization_results[idx]))
        y = enc_preprocess(Image.open(content_images[idx]))
        L_max = compute_lipschitz_vectorized(x.numpy(), y.numpy(), num_samples)
        lipschitz_constants.append(L_max)

    lipschitz_constants = np.array(lipschitz_constants)
    

    lipschitz_constants = np.array(lipschitz_constants)

    average_L = np.mean(lipschitz_constants)
    std_L = np.std([lipschitz_constants[np.random.randint(0, len(lipschitz_constants), len(lipschitz_constants))].mean() for _ in range(1000)]) #bootstrap std

    print("Average Lipschitz constant:", average_L)
    print("Standard deviation of Lipschitz constant:", std_L)
