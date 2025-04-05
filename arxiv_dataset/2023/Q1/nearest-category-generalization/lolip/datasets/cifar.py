# https://github.com/keras-team/keras/issues/2653

import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100


def cifar100coarse_leave_one_out(to_left_out):
    if to_left_out == 0:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="coarse")
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
        rest = (x_train[y_train==0], x_test[y_test==0],
                0*np.ones((y_train==0).sum()), 0*np.ones((y_test==0).sum()))
        x_train, x_test = x_train[y_train != 0], x_test[y_test != 0]
        y_train, y_test = y_train[y_train != 0], y_test[y_test != 0]
        y_train[y_train == 19] = 0
        y_test[y_test == 19] = 0
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="coarse")
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
        rest = (x_train[y_train==to_left_out], x_test[y_test==to_left_out],
                to_left_out*np.ones((y_train==to_left_out).sum()), to_left_out*np.ones((y_test==to_left_out).sum()))
        x_train, x_test = x_train[y_train != to_left_out], x_test[y_test != to_left_out]
        y_train, y_test = y_train[y_train != to_left_out], y_test[y_test != to_left_out]
        y_train[y_train > to_left_out] -= 1
        y_test[y_test > to_left_out] -= 1
    return x_train, y_train, x_test, y_test, rest

def cifar10_leave_one_out(to_left_out):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    rest = (x_train[y_train==to_left_out], x_test[y_test==to_left_out],
            to_left_out*np.ones((y_train==to_left_out).sum()), to_left_out*np.ones((y_test==to_left_out).sum()))
    x_train, x_test = x_train[y_train != to_left_out], x_test[y_test != to_left_out]
    y_train, y_test = y_train[y_train != to_left_out], y_test[y_test != to_left_out]
    y_train[y_train > to_left_out] -= 1
    y_test[y_test > to_left_out] -= 1

    return x_train, y_train, x_test, y_test, rest


fine_labels = [
    'apple', # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]

coarse_labels = [
    'aquatic mammals',
    'fish',
    'flowers',
    'food containers',
    'fruit and vegetables',
    'household electrical device',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertebrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'vehicles 1',
    'vehicles 2',
]

superclass_mapping = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}
