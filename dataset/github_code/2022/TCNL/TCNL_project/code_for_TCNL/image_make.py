import cv2
import os
import numpy as np

head = '/home/wangzhihao/XAI/XCSGCNN/train/visualization/exp_v3/vgg_multi_concept_mammal/train/head/1365/cat/cat_129.jpg'
torso = head.replace('head', 'torso')
leg = head.replace('head', 'leg')

image_head = cv2.imread(head)
image_torso = cv2.imread(torso)
image_leg = cv2.imread(leg)
cv2.imwrite('head.jpg', image_head)
cv2.imwrite('torso.jpg', image_torso)
cv2.imwrite('leg.jpg', image_leg)

# mask_head = np.array(image_head, dtype=bool)
# mask_head = np.array(mask_head, dtype=int)
# mask_torso = np.array(image_torso, dtype=bool)
# mask_torso = np.array(mask_torso, dtype=int)
# mask_leg = np.array(image_leg, dtype=bool)
# mask_leg = np.array(mask_leg, dtype=int)

# # intersection_head_torso = np.intersect1d(mask_head, mask_torso)
# intersection_head_torso = np.bitwise_and(mask_head, mask_torso)
# # intersection_torso_leg = np.intersect1d(mask_torso, mask_leg)
# intersection_torso_leg = np.bitwise_and(mask_torso, mask_leg)

# image_torso[np.where(intersection_head_torso!=0)] = 0
# image_leg[np.where(intersection_torso_leg!=0)] = 0

# image = image_head + image_torso + image_leg
image = cv2.add(image_head, image_torso)
image = cv2.add(image, image_leg)
cv2.imwrite('test.jpg', image)


