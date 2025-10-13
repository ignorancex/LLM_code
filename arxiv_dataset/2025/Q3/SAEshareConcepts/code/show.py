from argparse import ArgumentParser
from PIL import Image, ImageDraw
from glob import glob
import torch
import numpy as np
from tqdm import tqdm
from rich import print
import os

import utils

def get_last_layer(run):
    """
    Get n° of the last layer, to compute Comparative Sharedeness
    Usually 23,-1 or 12
    """
    subdirs = glob(f'sae_features/{run}/*')
    if len(subdirs) == 1: return subdirs[0]
    elif f'sae_features/{run}/layer-1.pt' in subdirs: return f'sae_features/{run}/layer-1.pt'
    return sorted(subdirs)[-1]

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--run', default=None, type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arr = torch.load(get_last_layer(args.source), map_location=device).to(torch.float32)

    if not os.path.isdir('imgs'): os.mkdir('imgs')

    arr = arr.T

    dataset = utils.get_dataloader_cocoimage(None, batch_size=1, shuffle=False).dataset
    _idx = 0
    for arridx in tqdm(range(arr.shape[0])):
        
        _idx += 1
        feature_torch = arr[arridx]

        feature = feature_torch.to_dense().cpu().detach().numpy()

        if np.count_nonzero(feature) < 16: continue
        
        top_9_indices = np.argsort(feature)[-(3*3):][::-1]


        tot = np.nansum([feature[t] for t in top_9_indices])
        if tot == 0: continue

        fact = [feature[i] for i in top_9_indices]
        if np.nansum(fact) == 0: continue
        tot = np.nansum(fact)

        grid_size=(3, 3)
        image_size=(100, 100)

        rows, cols = grid_size
        w, h = image_size
        grid = Image.new('RGB', size=(cols*w, rows*h), color='white')  # White background


        imgs = [dataset[int(i)] for i in top_9_indices]

        
        for i, image in enumerate(imgs):
            if i >= rows * cols:  # Ensure we don't exceed the grid size
                break
            
            # Resize and crop to maintain aspect ratio
            image.thumbnail(image_size)
            
            # Calculate position
            x = (i % cols) * w
            y = (i // cols) * h
            
            # Create a blank image for this cell
            cell = Image.new('RGB', image_size, 'white')
            
            # Paste the resized image into the center of the cell
            paste_x = (w - image.width) // 2
            paste_y = (h - image.height) // 2
            cell.paste(image, (paste_x, paste_y))

            grid.paste(cell, (x, y))
            
        del feature

        fname = f"imgs/feat{_idx-1}_{tot}.jpg"
        grid.save(fname)
        
    print("Done.")