import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# src data/new_train_set2/data_aug/image_0_631054.png
# mask data/new_train_set2/data_aug/mask_0_631054.png
from cv2 import THRESH_OTSU
def Otsu(img):
    t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    t2, Otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('thd', thd)
    cv2.imshow('otsu', Otsu)
    cv2.waitKey()
    cv2.destroyAllWindows()

def findSeeds(img,area=5):
    ret, img1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img1,8)

    centroids = centroids.astype(int)
    return  centroids, img1

def getGrayDiff(image,currentPoint,tmpPoint):
    return abs(int(image[currentPoint[0],currentPoint[1]]) - int(image[tmpPoint[0],tmpPoint[1]]))
    # return abs(int(image[tmpPoint[0],tmpPoint[1]]))

def regional_growth (gray,seeds,threshold=5) :
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), \
                        (0, 1), (-1, 1), (-1, 0)]
    threshold = threshold
    height, weight = gray.shape
    seedMark = np.ones(gray.shape)
    seedList = []
    for seed in seeds:
        if(seed[0] < gray.shape[0] and seed[1] < gray.shape[1] and seed[0]  > 0 and seed[1] > 0):
            seedList.append(seed)
    label = 0
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
        seedMark[currentPoint[0],currentPoint[1]] = label
        for i in range(8):
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(gray,currentPoint,(tmpX,tmpY))
            if grayDiff < threshold and seedMark[tmpX,tmpY] == 1:
                seedMark[tmpX,tmpY] = label
                seedList.append((tmpX,tmpY))
    return seedMark*255


def regionGrow(img):
    seed, img1 = findSeeds(img)
    img = regional_growth(img, seed)
    Image.fromarray(img).show()
    print("ok...")
    return img
