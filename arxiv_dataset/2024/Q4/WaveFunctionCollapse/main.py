# import os
# import glob

import numpy as np
import rasterio

from WFC import WaveFunctionCollapse, observe_mode
from utils import save_state, load_state
from functions import height_to_slopes, slopes_to_height, augment_images

import os
import glob

# reading data
# sample_data_path = "data/N26E057.hgt"
# sample_data_path = "data/N27E060.hgt"
# sample_data_path = "data/N28E057.hgt"

# save_path = "./results/run"
save_path = "./results/run_multi_input"

hgt_list = glob.glob("data/*.hgt")
# hgt_list = [hhh for hhh in hgt_list if os.path.basename(hhh).startswith("N30")]

n_max = 2

for sample_data_path in hgt_list:#["data/N30E054.hgt", "data/N31E051.hgt", "data/N35E047.hgt"]:
    scale_factor = 1.0 / 8


    with rasterio.open(sample_data_path) as src:
        # Define the desired smaller shape (for example, half the original size)
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)

        # Read and resample the data to the new shape
        sample_data = [src.read(
            1,  # First band
            out_shape=(new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear  # Choose the resampling method
        ).astype(float)[250:350, 200:300]]
        
    sample_data = augment_images(sample_data, rot90=False, rot180=False, rot270=False, flip_h=True, flip_v=True)
    print("data path: ", sample_data_path)

    slopes = []
    
    for dat in sample_data:
        
        # calculating grads
        grad_x, grad_y = height_to_slopes(dat)

        # recreatinh hright from grad
        hh = slopes_to_height(grad_x, grad_y)
        
        
        if len(np.unique(hh-dat)) == 1:
            print("Recreating height succesful")
        else:
            print("Recreating height failed")

        # WFC Extraction
        slope = np.concatenate([grad_x[:-1, ..., np.newaxis], grad_y[..., :-1, np.newaxis]], axis=2)
        slopes.append(slope)

    dirname = os.path.splitext(os.path.basename(sample_data_path))[0]
    
    saved_wfc_path = os.path.join(save_path, dirname, "wfc_state")
    outs_path = os.path.join(save_path, dirname, "outs")
    os.makedirs(outs_path, exist_ok=True)
    
    wfc_terrain = WaveFunctionCollapse(slopes, (2,2,2), remove_low_freq=False, low_freq=1)

    if os.path.exists(saved_wfc_path):
        print("Found existing wfc...")
        wfc_terrain.load(saved_wfc_path)
    elif os.path.exists(saved_wfc_path + '.pkl'):
        print("Found existing pkl wfc...")
        wfc_terrain = load_state(saved_wfc_path)
    else:
        wfc_terrain.extract_patterns()
        wfc_terrain.match_patterns()
        os.makedirs(saved_wfc_path, exist_ok=True)
        wfc_terrain.save(saved_wfc_path)

    # running WFC
    n_out = 0
    try:
        while n_out < n_max:
            mode = observe_mode.MB
            mode_name = "MB" if mode == observe_mode.MB else "RANDOM"
            output_image = wfc_terrain.run(grid_size=((20, 20, 2)), mode=mode)
            
            save_state(output_image, os.path.join(save_path, dirname, "outs", f"wfc_out_{n_out+1}_{mode_name}.npy"), overwrite=True)
            save_state(wfc_terrain.gosal_image, os.path.join(save_path, dirname, "outs", f"wfc_gosal_{n_out+1}_{mode_name}.npy"), overwrite=True)
            n_out += 1
    except AttributeError:
        print("WFC version conflict. try creating the WFC object from the beginning.")
        continue

    print(f"done with {dirname}...")
    print(f"proceding to next file...")
    print("------------------------")
    print()    