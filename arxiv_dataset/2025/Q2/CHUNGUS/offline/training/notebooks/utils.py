import tqdm
import torch
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms


def resize_image(image, resolution):
    """ Resizes an image """
    if resolution is None:
        return image
    else:
        return skimage.transform.resize(image, resolution)


def read_image(file, resolution=None):
    """ Read an image (optionally resizing it) """
    image = skimage.io.imread(file)[:,:,:3] # :3 to remove potential alpha channel
    image = Image.fromarray(skimage.img_as_ubyte(resize_image(image, resolution)))
    return image


def image_to_tensor(image):
    """ Convert PIL image to a tensor """
    return transforms.ToTensor()(image)


def evaluate(csv_path, image_folder, network, thresholds, device):
    """ Evaluate a network on the data in the given label file
    
    :param csv_path: path to label csv file
    :param image_folder: folder that images are stored in
    :param network: network to use for inference
    :param thresholds: HDR thresholds to use (list of floats)
    :param device: device to inference (must be cuda device for FeatUp)
    """

    # Read csv
    csv = pd.read_csv(csv_path)
    
    # Store the results here
    results = {t: [] for t in thresholds}
    
    for entry_index in tqdm.tqdm(range(len(csv))):
        entry = csv.iloc[entry_index]
    
        # List where each entry is [i1, x1, y1, i2, x2, y2, label]
        annotations = [[int(l) for l in s[1:-1].split(',')] for s in entry['labels'].split(';')]
        
        # Read images
        imageA_tensor = image_to_tensor(read_image(image_folder / entry['imageA_name']))
        imageB_tensor = image_to_tensor(read_image(image_folder / entry['imageB_name']))
    
        # Perform inference
        network.eval()
        network.to(device)
        with torch.no_grad():
            predictionA = network(imageA_tensor.to(device), should_normalize=True, resize=True)['prediction'].cpu()
            predictionB = network(imageB_tensor.to(device), should_normalize=True, resize=True)['prediction'].cpu()

        # Ensure predictions have correct resolution
        assert(predictionA.shape[1] == entry['width'], predictionA.shape[0] == entry['height'])
        assert(predictionB.shape[1] == entry['width'], predictionB.shape[0] == entry['height'])
        predictions = torch.stack([predictionA, predictionB], dim=0)
        
        # Compute disagreement
        for threshold in thresholds:
            for a in annotations:
                p_A = predictions[a[0], a[2], a[1]].item() # first is index, then y, then x
                p_B = predictions[a[3], a[5], a[4]].item() # first is index, then y, then x
                l = a[6] # label
                diff = p_B - p_A
                # Decision
                if diff > threshold:
                    p = 1
                elif diff < -threshold:
                    p = -1
                else:
                    p = 0
                results[threshold].append(l != p)

    results = {k: np.array(v).mean() for k, v in results.items()}
    return results
