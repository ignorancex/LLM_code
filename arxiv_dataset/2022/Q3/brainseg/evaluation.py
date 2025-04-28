import argparse
import numpy as np
from tqdm import tqdm
from utils import test_images, test_single_nii, calculate_metric_percase
from data_reader import H5DataLoader
import h5py

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='BCE1',
                    help='ss')


args = parser.parse_args()

data_file = '/project/ajoshi_27/code_farm/brainseg/test_t1t2.h5'

predictions_file = '/scratch1/wenhuicu/brainseg/test_t1_output_' + args.name + '.h5'
#predictions_file = 'test_t1.h5'


test = h5py.File(data_file, 'r')['Y']

pred = h5py.File(predictions_file, 'r')['Y']

num_classes = 9
num_imgs = test.shape[0]
dice_coeffs = np.zeros([num_imgs,num_classes])
hausdorff_dist = np.zeros([num_imgs,num_classes])


for i in tqdm(range(num_imgs)):

    metric_list = []
    for lid in range(num_classes):
        dice_coeffs[i,lid], hausdorff_dist[i,lid] = calculate_metric_percase(test[i] == lid, pred[i] == lid)

print(args.name)
print(np.nanmean(dice_coeffs,axis=0), np.nanstd(dice_coeffs,axis=0))


    

