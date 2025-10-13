import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

def Data_class_analyze(masks,# a numpy array (batch, H, W, C)
                       class_labels): # A list of classes with correct order. for example class0 should be placed in first place.
    """This function analyzes the images and masks. it counts number and area of samples for each class.
    It is a good representation for imbalance data. It also gives hints to data augmentation and over-sampling"""

    shape = np.shape(masks)
    num_classes = len(class_labels)
    class_samples = np.zeros(num_classes)
    class_areas = np.zeros(num_classes)
    class_distribution = np.zeros((shape[0],num_classes))
    for i in range(shape[0]):
        msk = masks[i]
        if len(msk.shape)>2:
            msk = np.sum(msk, axis=2)# convert to one image
        for j in range(num_classes):
            area = np.sum(msk == j)
            if area > 0:
                # if j == 5:
                #     plt.imshow(msk)
                #     plt.show()
                class_samples[j] += 1
                class_areas[j] += area
                class_distribution[i,j] += 1
    return class_samples, class_areas, class_distribution



def split_train_val(class_samples = 0, class_distribution = 0, val_percent = 0.2):
    a = 0
    class_distribution1 = np.copy(class_distribution)
    shape = np.shape(class_distribution)
    val_samples = int(shape[0]*val_percent)
    train_samples = shape[0] - val_samples
    val_index = np.zeros(val_samples)
    train_index = np.zeros(train_samples)
    val_indexes = 0
    val_samples1 = np.copy(val_samples)
    for i in range(1,shape[1]):
        ind = np.where(class_samples[1:] == np.min(class_samples[1:]))[0]+1
        ind = ind[0]
        val = class_samples[ind]
        class_samples[ind] = np.inf
        random.seed(42)
        val_sample_inds = random.sample(range(int(val)), int(np.ceil(val*val_percent)))
        print(val_sample_inds)
        k = 0
        for m in range(shape[0]):
            if class_distribution[m,int(ind)] == 1:
                if len(np.where(np.array(val_sample_inds) == k)[0]):
                    val_index[val_indexes] = m
                    class_distribution[m] = 0*class_distribution[m]
                    val_indexes += 1
                    val_samples1 -= 1
                    if val_samples1 == 0:
                        break
                    # print(k)
                k += 1
        if val_samples1 ==0:
            break
    val_index = np.where(np.sum(class_distribution,axis = 1) == 0)
    train_index = np.where(np.sum(class_distribution, axis=1) != 0)

    # print(val_index)





    return train_index, val_index


def addsamples(images,mask,sample_th = 0.2, tissue_labels = None):
    a = 2
    angles = [45,90,135,180,225,270,325]
    shape = np.shape(images)
    shape_m = np.shape(mask)
    new_ims = []
    new_masks = []
    class_samples, class_areas, class_distribution = Data_class_analyze(mask, tissue_labels)
    average_class = class_samples / np.shape(mask)[0]
    average_area = class_areas / np.sum(class_areas)
    avg = 100 * average_class * 100 * average_area / np.sum(100 * average_class * 100 * average_area)
    new_ims = []
    new_masks = []
    for i in range(1,shape_m[3]):# start from 1 to skip background
        ind = np.where(avg[1:] == np.min(avg[1:]))[0]+1
        ind = ind[0]
        val = avg[ind]
        avg[ind] = np.inf
        class_inds = np.where(class_distribution[:,ind] == 1)
        if val < sample_th:
            up_num = int(shape[0] * sample_th/(1-sample_th) - 100*average_area[ind])
            for j in range(3):# fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break
                    new_ims.append(cv2.flip(images[inds],j-1))
                    new_masks.append(cv2.flip(mask[inds],j-1))
                    up_num -= 1

            for rot in angles:
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                    imm = cv2.warpAffine(images[inds], M, (shape[1], shape[2]),borderValue=(255,255,255))
                    new_ims.append(imm)
                    imm = cv2.warpAffine(np.array(mask[inds], dtype=np.uint8), M, (shape[1], shape[2]))
                    new_masks.append(imm)


                    up_num -= 1

            for j in range(3):  # fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    image = cv2.flip(images[inds], j-1)
                    msk = cv2.flip(mask[inds], j-1)
                    for rot in angles:
                        if up_num <= 0:
                            break
                        M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                        imm = cv2.warpAffine(image, M, (shape[1], shape[2]),borderValue=(255,255,255))
                        new_ims.append(imm)
                        imm = cv2.warpAffine(np.array(msk, dtype=np.uint8), M, (shape[1], shape[2]))
                        new_masks.append(imm)
                        up_num -= 1


            class_distribution[class_inds] = 0 * class_distribution[class_inds]
    a = 0
    images1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape[3]))
    mask1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape_m[3]))
    images1[0:shape[0]] = images
    mask1[0:shape[0]] = mask

    new_ims1 = np.array(new_ims)
    new_masks1 = np.array(new_masks, dtype=np.uint8)
    images1[shape[0]:] = new_ims1
    mask1[shape[0]:] = new_masks1
    images1 = images1.astype('float32')
    mask1 = mask1.astype('int64')

    return images1, mask1
