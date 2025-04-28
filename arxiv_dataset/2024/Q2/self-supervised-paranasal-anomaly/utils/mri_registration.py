import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
# - set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib


from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,
                                  MutualInformationMetric,
                                  AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                    RigidTransform3D,AffineTransform3D)

import os
import glob
from tqdm import tqdm

root_dir = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/_raw_bids_hno"
save_loc = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/nature-journal-dataset/{}.nii.gz"

# Get a list of subdirectories within the root directory
subdirectories = [name for name in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, name))]

template_img = nib.load('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/_raw_bids_hno/sub-0a249e33/ses-1/anat/sub-0a249e33_ses-1_FLAIR.nii.gz')
template_data = template_img.get_data()
template_affine = template_img.affine

# The mismatch metric
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

# The optimization strategy
level_iters = [10000, 1000, 10]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]


for subdirectory in tqdm(subdirectories,total=len(subdirectories)):
    
    nested_sub_directory = os.path.join(root_dir, subdirectory) + "/ses-1/anat/*.gz"
    # Use glob to get a list of file paths matching the file extension in the directory
    file_paths = glob.glob(nested_sub_directory)

    file_paths = [x for x in file_paths if "FLAIR" in x]
    #Check if there is atleast 1 FLAIR image 
    if len(file_paths) > 0: 

        moving_img = nib.load(file_paths[0])

        moving_data = moving_img.get_data()
        moving_affine = moving_img.affine
        #set up the registration object, ready to do the registration:
        affreg = AffineRegistration(metric=metric,
                             level_iters=level_iters,
                             sigmas=sigmas,
                             factors=factors)
        
        """
        #Translation transform
        transform = TranslationTransform3D()
        params0 = None
        print("Translation transform started")
        translation = affreg.optimize(template_data, moving_data, transform, params0,template_affine, moving_affine)

        #transformed = translation.transform(moving_data)

        print("Translation transform complete")

        #Rigid transform
        transform = RigidTransform3D()
        rigid = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=translation.affine)


        #transformed = rigid.transform(moving_data)

        print("Rigid transform complete")
        """
        params0 = None
        transform = AffineTransform3D()
        # Bump up the iterations to get an more exact fit
        affreg.level_iters = [10000, 1000, 10]
        affine = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine)
                                  #starting_affine=rigid.affine)

        transformed = affine.transform(moving_data)

        print("Affine transform complete")

        #Save rigid transformed image
        nib.save(transformed, save_loc.format(subdirectory))

        







    





    