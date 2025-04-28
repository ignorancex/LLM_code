import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_IHC(im_path):
    save_path = os.path.join('histogram/', im_path.split('/')[-1])
    print(im_path)
    all_hist = np.zeros(255)
    # mean_hist = np.zeros(255)
    im_num = 0
    # Compute the mean histogram
    for root, dirs, files in os.walk(im_path):
        for file in files:
            if not file.endswith('.png'):
                continue
            im_name = os.path.join(root, file)
            im = cv2.imread(im_name)
            # calculate mean value from RGB channels and flatten to 1D array
            vals = im.mean(axis=2).flatten()
            # plot histogram with 255 bins
            cur_hist, bins, patches = plt.hist(vals, 255, [0, 255], color='navy')
            # cur_hist, bins, patches = plt.hist(vals, 255)
            plt.xlabel("Illumination Level", fontsize=18, fontweight='bold', labelpad=0)
            plt.ylabel('Frequency', fontsize=18, fontweight='bold', labelpad=0)
            plt.xticks(size=18, fontweight='bold')
            plt.yticks(size=18, fontweight='bold')
            plt.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0)  # dpi值为200
            plt.show()

            all_hist = all_hist+cur_hist
            im_num = im_num+1
    mean_hist = all_hist/im_num

    # Plot the mean illumination map
    x_axi=range(0, 255)
    plt.bar(x_axi, mean_hist, align="center", color='red')
    plt.xticks(size=18, fontweight='bold')
    plt.yticks(size=18, fontweight='bold')
    plt.xlabel("Illumination Level", fontsize=18, fontweight='bold', labelpad=0)
    plt.ylabel('Frequency', fontsize=18, fontweight='bold', labelpad=0)

    plt.savefig('mean.png', dpi=200, bbox_inches='tight', pad_inches=0)  # dpi值为200
    plt.show()

    # Compute the histogram discrepancies of each image and Sum them together
    sum_hist = np.zeros(255)
    for root, dirs, files in os.walk(im_path):
        for file in files:
            if not file.endswith('.png'):
                continue
            im_name = os.path.join(root, file)
            im = cv2.imread(im_name)
            height, width = im.shape[:2]
            # calculate mean value from RGB channels and flatten to 1D array
            vals = im.mean(axis=2).flatten()
            # plot histogram with 255 bins
            cur_hist, bins, patches = plt.hist(vals, 255, [0, 255], color='navy')
            # cur_hist, bins, patches = plt.hist(vals, 255)

            # cur_hist, bins, patches = plt.hist(vals, 255)
            # discrepancy_hist = abs(cur_hist-mean_hist)
            discrepancy_hist = abs(cur_hist-mean_hist)
            # print(cur_hist)
            # print('Sum_cur_hist:' + str(sum(cur_hist)))
            # print('Sum_discrepancy:' + str(sum(discrepancy_hist)))
            sum_hist = sum_hist+discrepancy_hist
            # print('Sum_discrepancys:' + str(sum(sum_hist)))

    # Normalize the histogram discrepancies
    sum_hist_value=sum(sum_hist)
    IHD=sum_hist_value/(im_num*512*512)
    print(im_path + ' IHD: '+ str(IHD))

    IHC=1-IHD
    print(im_path + ' IHC: '+ str(IHC))

# im_paths=['demo', 'groups_0', 'groups_1', 'groups_2', 'groups_3', 'groups_4', 'groups_5', 'groups_6', 'groups_7', 'groups_8', 'groups_9', 'groups_10']
im_paths=['groups_6']

root='illumination'

for im_path in im_paths:
    im_path = os.path.join(root, im_path)
    compute_IHC(im_path)