import numpy as np
from tqdm import tqdm
from utils import test_images, test_single_nii, calculate_metric_percase
from data_reader import H5DataLoader
import h5py

data_file = 'test_t1t2.h5'

predictions_file = 'test_t1_output.h5'
#predictions_file = 'test_t1.h5'


test = h5py.File(data_file, 'r')['Y']

pred = h5py.File(predictions_file, 'r')['Y']

num_classes = 9
num_imgs = test.shape[0]
dice_coeffs = np.zeros([num_imgs,num_classes])
hausdorff_dist = np.zeros([num_imgs,num_classes])


for i in range(num_imgs):

    metric_list = []
    for lid in range(num_classes):
        dice_coeffs[i,lid], hausdorff_dist[i,lid] = calculate_metric_percase(test[i] == lid, pred[i] == lid)


print('Dice mean: ', np.nanmean(dice_coeffs,axis=0))
print('Dice std: ', np.nanstd(dice_coeffs,axis=0))

print('Hausdorff mean: ', np.nanmean(hausdorff_dist,axis=0))
print('Hausdorff std: ', np.nanstd(hausdorff_dist,axis=0))


    

