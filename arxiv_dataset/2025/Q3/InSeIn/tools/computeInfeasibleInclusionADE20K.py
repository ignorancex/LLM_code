# import required libraries
import cv2
import numpy as np
import pandas as pd
import os
from itertools import permutations
from copy import deepcopy

dataset_path = "D:/Thesis/ADEChallengeData2016/annotations/training/"
ADE_data = pd.read_csv("D:/Thesis/ADEChallengeData2016/color_coding.csv")
ADE_data_modified = ADE_data[['Idx', 'Name']]
ADE_Idx = ADE_data_modified['Idx'].tolist()

constraint_pairs = [pair for pair in permutations(ADE_Idx,r=2)]

feasible_pairs = []
all_possible_pairs = []
invalid_pairs_trainId = []
for pair in constraint_pairs:
    id1 = pair[0]
    id2 = pair[1]
    name1 = ADE_data_modified.iloc[ADE_data_modified.index[ADE_data_modified['Idx']==id1]]['Name'].iloc[0].replace(';','_')
    name2 = ADE_data_modified.iloc[ADE_data_modified.index[ADE_data_modified['Idx']==id2]]['Name'].iloc[0].replace(';','_')
    count = 0
    for files in os.listdir(dataset_path):
        # read input image
        img = cv2.imread(dataset_path+files)

        includee = cv2.inRange(img, np.array([id1,id1,id1]), np.array([id1,id1,id1]))
        includer = cv2.inRange(img, np.array([id2,id2,id2]), np.array([id2,id2,id2]))
        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(includee,4,cv2.CV_32S)

        for i in range(1, totalLabels):
            if values[i, cv2.CC_STAT_AREA]>5:
               output = np.zeros(includee.shape, dtype="uint8")
               component_mask = ((label_ids == i).astype("uint8") * 255)
               output = cv2.bitwise_or(output, component_mask)
               inverted_image = cv2.bitwise_not(output)
               h, w = inverted_image.shape
               mask = np.zeros((h + 2, w + 2), np.uint8)
               cv2.floodFill(inverted_image, mask, (0, 0), 255)
               filled_holes = cv2.bitwise_not(inverted_image)
               filled_image = cv2.bitwise_or(output, filled_holes)
               kernel = np.ones((5,5), np.uint8)
               img_dilation = cv2.dilate(filled_image, kernel, iterations=1)
               contour = cv2.subtract(img_dilation, filled_image)
               backup = deepcopy(contour)
               result = cv2.bitwise_and(includer,contour, mask= None)

               if (backup[backup!=0].shape[0]==result[result!=0].shape[0]) and result[result!=0].shape[0]>5:
                   count +=1
    if count>0:
       feasible_pairs.append((id2,id1))
       print('feasible pairs-->'+name2+"--"+name1)
       count=0



infeasible_pairs = list(set(constraint_pairs) - set(feasible_pairs))
print('-------prepare invalid trainids---------')

invalid_pairs_trainId = [(pair[0]-1, pair[1]-1) for pair in infeasible_pairs]

print(invalid_pairs_trainId)
