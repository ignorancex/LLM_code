# import required libraries
import cv2
import numpy as np
from cityscapesscripts.helpers import labels
import os
from itertools import permutations
from copy import deepcopy

dataset_path = '~/ACDC/gt/'
splits = ['train','val']
eval_list = ['road', 'sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']

constraint_pairs = [pair for pair in permutations(eval_list,r=2)]

feasible_pairs = []
all_possible_pairs = []
invalid_pairs_trainId = []
for pair in constraint_pairs:
    name1 = pair[0]
    name2 = pair[1]
    id1 = labels.name2label[name1].id
    id2 = labels.name2label[name2].id
    all_possible_pairs.append((id2,id1))
    count = 0
    for condition in os.listdir(dataset_path):
        for folders in os.listdir(dataset_path+condition+'/'+splits[0]):
            for images in os.listdir(dataset_path+condition+'/'+splits[0]+'/'+folders):
                if "_labelIds" in images:
                   img = cv2.imread(dataset_path+condition+'/'+splits[0]+'/'+folders+'/'+images)

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



infeasible_label_ids = list(set(all_possible_pairs) - set(feasible_pairs))
print('-------prepare invalid trainids---------')
for pair in infeasible_label_ids:
    for label in labels.labels:
        if pair[0]==label.name:
            pair_label_id_0 = label.trainId
            print('trainId--->',pair_label_id_0)
            print('labelId--->',pair[0])
            print('class--->',label.name)
        if pair[1]==label.name:
            pair_label_id_1 = label.trainId

    invalid_pairs_trainId.append((pair_label_id_0,pair_label_id_1))

print(invalid_pairs_trainId)

