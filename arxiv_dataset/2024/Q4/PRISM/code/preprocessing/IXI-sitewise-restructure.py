import os

in_dir = 'C:/Users/HP/Desktop/Trust Lab/ixi_dataset_Skull_Stripped/'
out_dir = 'C:/Users/HP/Desktop/Trust Lab/IXI-original-nifti-stripped/'

# modalities = ['T1', 'T2', 'PD']
modalities = ['T1']
sites = ['Guys', 'HH', 'IOP']
# sites = ['HH']

for modality in modalities:
    for site in sites:
        if not os.path.exists(f"{out_dir}{site}"):
            os.makedirs(f"{out_dir}{site}")
        for f in os.listdir(f"{in_dir}{modality}"):
            # Check if the file contains the string site
            if site in f:
                # Move the file to the corresponding site folder
                fname = f[:-20] + f[-16:]
                os.rename(f"{in_dir}{modality}/{f}", f"{out_dir}{site}/{fname}")
                # os.system(f'copy "{in_dir}{modality}/{f}" "{out_dir}{site}/{fname}"')

        # Create empty train and test folders for manually splitting the data afterwards
        if not os.path.exists(f"{out_dir}{site}/train"):
            os.makedirs(f"{out_dir}{site}/train")
        if not os.path.exists(f"{out_dir}{site}/test"):
            os.makedirs(f"{out_dir}{site}/test")