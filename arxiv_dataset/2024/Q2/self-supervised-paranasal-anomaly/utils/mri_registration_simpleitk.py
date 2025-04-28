import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
# - set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib


import SimpleITK as sitk
import os
import glob
from tqdm import tqdm

root_dir = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/_raw_bids_hno"
save_loc = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/nature-journal-dataset/{}.nii.gz"

# Get a list of subdirectories within the root directory
subdirectories = [name for name in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, name))]


for subdirectory in tqdm(subdirectories,total=len(subdirectories)):
    
    nested_sub_directory = os.path.join(root_dir, subdirectory) + "/ses-1/anat/*.gz"
    # Use glob to get a list of file paths matching the file extension in the directory
    file_paths = glob.glob(nested_sub_directory)

    file_paths = [x for x in file_paths if "FLAIR" in x]
    #Check if there is atleast 1 FLAIR image 
    if len(file_paths) > 0: 

        
        #set up the registration object, ready to do the registration:

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(sitk.ReadImage('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/_raw_bids_hno/sub-0a249e33/ses-1/anat/sub-0a249e33_ses-1_FLAIR.nii.gz'))
        elastixImageFilter.SetMovingImage(sitk.ReadImage(file_paths[0]))
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        elastixImageFilter.Execute()
        sitk.WriteImage(elastixImageFilter.GetResultImage(),save_loc.format(subdirectory))
        #Save rigid transformed image
        #nib.save(transformed, save_loc.format(subdirectory))

        







    





    