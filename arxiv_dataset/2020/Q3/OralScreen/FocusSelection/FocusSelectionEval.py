# -*- coding: utf-8 -*-
'''
Evaluate the Focus Selection (FS) performance with EMBM.
'''
from glob import glob
import os
import numpy as np
from tqdm import tqdm
import matlab.engine

z_offsets = ['z-2000', 'z-1600', 'z-1200', 'z-800', 'z-400', 'z0', 
             'z400', 'z800', 'z1200', 'z1600', 'z2000']

patch_dir = '../VOC2007/Patches/Z_expanded/'
#test_dir = '../VOC2007/Patches/Z_testset/'
#patch_poss = ['_'.join(os.path.basename(patch).split('_')[:-1]) for patch in glob(test_dir + '*.jpg')]
patch_poss = ['000024_0006', '000024_0007', '000024_0008', '000024_0009', '000024_0010', '000024_0011', '000024_0012', '000024_0013', '000024_0014', '000024_0015', '000024_0016', '000024_0017', '000024_0018', '000024_0019', '000024_0020', '000024_0021', '000024_0022', '000024_0023', '000024_0024', '000024_0025', '000024_0026', '000024_0027', '000024_0028', '000024_0029', '000024_0030', '000024_0031', '000024_0032', '000024_0033', '000024_0034', '000024_0035', '000024_0036', '000024_0037', '000024_0038', '000024_0039', '000024_0040', '000024_0041', '000024_0042', '000024_0043', '000024_0044', '000024_0045', '000024_0046', '000024_0047', '000024_0048', '000024_0049', '000024_0050', '000024_0051', '000024_0052', '000024_0053', '000024_0054', '000024_0055', '000024_0056', '000024_0057', '000024_0058', '000024_0059', '000024_0060', '000024_0061', '000024_0062', '000024_0063', '000024_0064', '000024_0065', '000024_0066', '000024_0067', '000024_0068', '000024_0069', '000024_0070', '000024_0071', '000057_0000', '000057_0001', '000057_0002', '000057_0003', '000057_0004', '000057_0005', '000057_0006', '000057_0007', '000057_0008', '000057_0009', '000057_0010', '000057_0011', '000057_0012', '000057_0013', '000057_0014', '000057_0015', '000057_0016', '000057_0017', '000057_0018', '000057_0019', '000057_0020', '000057_0021', '000057_0022', '000057_0023', '000057_0024', '000057_0025', '000057_0026', '000057_0027', '000057_0028', '000057_0029', '000057_0030', '000057_0031', '000057_0032', '000057_0033']

#%%
def create_groundtruth(delta_median=2, delta_range=0):
    annos = np.empty([100, 0], dtype=int)
    
    for anno_path in glob("../VOC2007/Patches/Z_eval/annotation_*.txt"):
        with open(anno_path) as f:
            anno = f.read()
        anno_line = np.asarray([int(i) for i in anno.split()]).reshape(100, 1)
        annos = np.concatenate((annos, anno_line), axis=-1)
    
    medians = np.median(annos, axis=-1)
#    mean = np.mean(annos, axis=-1)
#    var = np.var(annos, axis=-1)
#    ptp = np.ptp(annos, axis=-1)
    mins = np.min(annos, axis=-1)
    maxs = np.max(annos, axis=-1)
    
    # Calculate human performance
    tps_median_list = []
    tps_range_list = []
    for person in range((annos.shape[1])):
        tps_m = 0
        tps_r = 0
        _annos = np.delete(annos, person, axis=-1)
        _medians = np.median(_annos, axis=-1)
        _mins = np.min(_annos, axis=-1)
        _maxs = np.max(_annos, axis=-1)
        
        for i in range((_annos.shape[0])):
            if annos[i, person] >= _medians[i] - delta_median and annos[i, person] <= _medians[i] + delta_median:
                tps_m += 1
            if annos[i, person] >= _mins[i] - delta_range and annos[i, person] <= _maxs[i] + delta_range:
                tps_r += 1
                
        tps_median_list.append(tps_m)
        tps_range_list.append(tps_r)
        
    tp_median_hmavg = np.mean(tps_median_list)
    tp_range_hmavg = np.mean(tps_range_list)
    
    return mins, maxs, medians, tp_median_hmavg, tp_range_hmavg

   

gt_min, gt_max, gt_median, tp_median_hmavg, tp_range_hmavg = create_groundtruth()



#%%
def count_TP(k=0, m=3, delta_median=2, delta_range=0, showN=False):
    
    eng = matlab.engine.start_matlab()
    tp_range = 0
    tp_median = 0
    
    if showN == True:
        Negs = {}
    
    for i in range(len(patch_poss)):
        patch_pos = patch_poss[i]
        
    #    Select best focus using a method
# =============================================================================
# =============================================================================
#         scores = []
#         for z_offset in z_offsets:
#             patch_name = "{}_{}.jpg".format(patch_pos, z_offset)
#             patch_path = patch_dir + patch_name
#             # read image through matlab
#             patch = eng.imread(patch_path)
#             
#             # Pre-processing
#             
#             # MATLAB median filter
#             if m > 1:
#                 patch = eng.medfilt3(patch, matlab.int8([m, m, 1]))
#             
#             # calcute quality score of the patch by EMBM
#             Q_image = eng.EdgeModelBasedIQA(patch)
#             scores.append(Q_image)
# =============================================================================
# =============================================================================
        patches = []    # create a list of focuses
        for z_offset in z_offsets:
            patch_name = "{}_{}.jpg".format(patch_pos, z_offset)
            patch_path = patch_dir + patch_name
            
            # read image through matlab
            patch = eng.imread(patch_path)
            
            # Pre-processing
            # MATLAB median filter
            if m > 1:
                patch = eng.medfilt3(patch, matlab.int8([m, m, 1]))
            patches.append(patch)
            
        stds = []   # create a std list for the difference of patch focuses 
        for j in range(len(z_offsets) - 1):
            diff = np.asarray(patches[j + 1]) - np.asarray(patches[j])
            stds.append(np.std(diff))
            
        i_diff_peak = stds.index(max(stds)) # index of the difference peak
        i_start = max(0, i_diff_peak - k)
        i_stop = min(len(z_offsets) - 1, i_diff_peak + 1 + k)
        
        scores = [0] * len(patches)
        for j in range(len(patches)):
            if i_start <= j and j <= i_stop: # measure Q_image only near diff peak
                scores[j] = eng.EdgeModelBasedIQA(patches[j])
# =============================================================================
            
            
        i_best = scores.index(max(scores)) # index of the best score
        
        
        if showN == True:
            
            if i_best >= gt_min[i] - delta_range and i_best <= gt_max[i] + delta_range:
                tp_range += 1
            else:
                Negs[patch_pos] = "{}-{} \t\t{} \t\t{} \t\t{}".format(gt_min[i], gt_max[i], 
                    gt_median[i], i_best, 
                    min(abs(i_best - gt_min[i]), abs(i_best - gt_max[i]), abs(i_best - gt_median[i])))
#                print("{}\t\t {}-{}\t {}".format(patch_pos, gt_min[i], gt_max[i], i_best))
                
            if i_best >= gt_median[i] - delta_median and i_best <= gt_median[i] + delta_median:
                tp_median += 1
            else:
                Negs[patch_pos] = "{}-{} \t\t{} \t\t{} \t\t{}".format(gt_min[i], gt_max[i], 
                    gt_median[i], i_best, 
                    min(abs(i_best - gt_min[i]), abs(i_best - gt_max[i]), abs(i_best - gt_median[i])))
#                print("{}\t\t {}\t {}".format(patch_pos, gt_median[i], i_best))
                
        else:
            if i_best >= gt_min[i] - delta_range and i_best <= gt_max[i] + delta_range:
                tp_range += 1
            if i_best >= gt_median[i] - delta_median and i_best <= gt_median[i] + delta_median:
                tp_median += 1
        
    eng.quit()
    
    if showN == True:
        print("\nWrong samples \t\tGT_range \tGT_median \tPrediction \tError distance")
        for key in Negs:
            print("{} \t\t{}".format(key, Negs[key]))

    
    return tp_range, tp_median#, Negs
#%%
for k in range(2,11):
    tp_range, tp_median = count_TP(k=k, m=3)
    accuracy_range = tp_range / 100
    accuracy_median = tp_median / 100
    print('k\tacc_m\tacc_r\n{}\t{}\t{}'.format(k, accuracy_median, accuracy_range))
    
accuracy_median_hm = tp_median_hmavg / 100
accuracy_range_hm = tp_range_hmavg / 100


