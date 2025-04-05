import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


import pickle
import math
import os
import numpy as np
import random
from PIL import Image

DICT_1 = [i for i in range(5)]
DICT_2 = [i for i in range(5)]

def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90: # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def take_class(elem):
    return relabel_class(elem[1])
    
parent_path = os.path.join('..', '..', 'pkl')
by_writer_dir = os.path.join(parent_path, 'images_by_writer')

writers = load_obj(by_writer_dir)

def index_class(digit, sub_l):
    # reture structure is [file_path of images of this class]
    imgs_by_class = []
    for (f, c) in sub_l:
        if relabel_class(c) == digit:
            imgs_by_class.append(f)
    random.shuffle(imgs_by_class)
    return imgs_by_class
        

def show_digits(j):
    plt.figure(figsize=(18,2))
    (w, l) = writers[j]
    sub_l = l
    sub_l.sort(key=take_class)

    i = 0
    my_dict = []
    my_dict.extend(DICT_1)
    my_dict.extend(DICT_2)
    for (f, c) in sub_l:
        if (relabel_class(c) in my_dict) and (len(my_dict) > 0):
            file_path = os.path.join(parent_path, f)
            img = Image.open(file_path)
            rgb = np.array(img).copy()
            plt.subplot(1, 10,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(rgb / 255.0, cmap=plt.cm.binary)
            plt.xlabel(relabel_class(c))
            i += 1
            my_dict.remove(relabel_class(c))
        else:
            pass        

    plt.show()

def ten_digits_in_ord(j):
    (w, l) = writers[j]
    
    digits = [i for i in range(10)]
    i = 0
    for k in digits:
        imgs = index_class(k, l)
        file_path = os.path.join(parent_path, imgs[0])
        img = Image.open(file_path)
        rgb = np.array(img).copy()
        plt.subplot(1, 10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(rgb / 255.0, cmap=plt.cm.binary)
        i += 1
        
#     plt.show()
    
def plot_writers(seqs):    
    for w in seqs:
        plt.figure(figsize=(10, 1))
        gs1 = gridspec.GridSpec(10, 1)
        gs1.update(wspace=0.025, hspace=0.025) 
        ten_digits_in_ord(w)
        plt.savefig("{}.png".format(w), dpi=100)
        plt.show()
