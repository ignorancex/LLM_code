from __future__ import print_function

import glob
import os
import random
import shutil
import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

def cleanDir(filepath):
    if not os.path.exists(filepath):
        # ---filepath
        #       |____train
        #       |____validation
        os.mkdir(filepath)
        os.mkdir(filepath + '/img')
        os.mkdir(filepath + '/mask')
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
        os.mkdir(filepath + '/img')
        os.mkdir(filepath + '/mask')

def copy_file(old_Dir,new_Dir,file_num):
    for i in range(file_num.__len__()):
        shutil.copyfile(old_Dir+"/"+str(file_num[i])+'.png',new_Dir+"/"+str(file_num[i])+'.png')

def randomSplitDataset(index, k=5, len=155, img_dir='data/train_binary', mask_dir='data/label_padding', aug_train_dir='data/aug_train',aug_val_dir='data/aug_validation'):
    # choice the i-th fold as the validation set, intotal k-fold
    # keep constant seed for genalize the sequnce
    numList = [[] for i in range(k)]
    for i in range(len // k):
        a = [x for x in range(1 + k * i, k * (i + 1) + 1)]
        random.seed(i)
        random.shuffle(a)
        for j in range(a.__len__()):
            numList[j].append(a[j] - 1)
    # split data into two folders
    cleanDir(aug_train_dir)
    cleanDir(aug_val_dir)
    validationNum = numList[index]
    trainNum = []
    for i in range(k):
        if i != index:
            trainNum += numList[i]
    print(trainNum.__len__())
    print(validationNum.__len__())
    train_img = aug_train_dir+'/img/'
    val_img = aug_val_dir+'/img/'
    train_mask = aug_train_dir+'/mask/'
    val_mask = aug_val_dir+'/mask/'
    copy_file(img_dir,train_img,trainNum)
    copy_file(img_dir, val_img , validationNum)
    copy_file(mask_dir,train_mask ,trainNum)
    copy_file(mask_dir, val_mask , validationNum)
    return aug_train_dir, aug_val_dir, 'img', 'mask'

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255  # normalization
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255 # normalization
        mask = mask /255
        mask[mask > 0.5] = 1    # binary
        mask[mask <= 0.5] = 0
    return (img,mask)

def data_gen(generator,flag_multi_class=False,num_class=2):
    for(img, mask ) in generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        # subset='training',
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    return train_generator
    # for (img,mask) in train_generator:
    #     img,mask = adjustData(img,mask,flag_multi_class,num_class)
    #     yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        # img = io.imread(os.path.join(test_path,"%04d.png"%i),as_gray = as_gray)

        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        print(img.shape)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def testGenerator_1(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    img_Dir = os.listdir(test_path)
    for i in range(num_image):
        # img = io.imread(os.path.join(test_path,"%04d.png"%i),as_gray = as_gray)
        img = io.imread(os.path.join(test_path,img_Dir[i]), as_gray=as_gray)
        print(img.shape)
        # img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

def saveResult(save_path,npyfile,binary=False,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        if binary:
            item = (item >= 0.5).astype(np.int_)
        else:
            item = item.astype(np.float)
        # img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path,"%d.png"%i),img)