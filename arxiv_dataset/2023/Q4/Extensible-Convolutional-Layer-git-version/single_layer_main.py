
import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np

import tqdm


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

import sys
import os
file_path = os.path.abspath(os.getcwd())
os.chdir(file_path)

import Generated_CNN
from Generated_CNN import *
from load_data import *

#parser
import argparse



'''
#load data
dataset = "cifar10"
train_images, train_labels, test_images, test_labels = load_data(dataset, grayscale = False, resize = False)


#initialize parameters
kernel_size_0 = 4
stride = 1
activation = "sigmoid"
image_size = train_images.shape[1]
initial_channel = train_images[0].shape[-1]
class_count = len(np.unique(train_labels))
training_count = 1000 #amount of images to train on
freeze = True
n_count = 50000 #amount of images to train with supervised learning
l0 = None
l1 = None
'''


def get_params(dataset, training_count, freeze, n_count): #called from external file
    
    dataset = dataset
    training_count = training_count
    freeze = freeze
    n_count = n_count
    
    train_images, train_labels, test_images, test_labels = load_data(dataset, grayscale = False, resize = False)
    
    class_count = len(np.unique(train_labels))
    initial_channel = train_images[0].shape[-1]
    image_size = train_images.shape[1]
    kernel_size_0 = 4
    stride = 1
    activation = "sigmoid"
    l0 = None
    l1 = None
    
    return dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count



def visualise_ord_CNN(l0, images_x_0_9, train_images, train_labels):
    figure = plt.figure(figsize=(20, 20)) #handler for controlling the size of the figure

    images_x_0_9 = []
    for i in range(10):
        images_x_0_9.append(train_images[np.argmax(train_labels == i)])
    image_size = train_images[0].shape[0]
    initial_channel = train_images[0].shape[2]

    for i in range(len(images_x_0_9)):
        feature_maps = tf.convert_to_tensor(l0.call(images_x_0_9[i].reshape(1, image_size, image_size, initial_channel).astype(np.float32)))

        plt.subplot((len(images_x_0_9) * 2) // 5, 5, i+1 + (i // 5) * 5)
        plt.imshow(images_x_0_9[i].reshape(image_size,image_size,initial_channel))
        plt.title("label "+ str(i))
        plt.subplot((len(images_x_0_9) * 2) // 5, 5, i+6 + (i // 5) * 5)

        index_map = np.argmax(feature_maps, axis = 3).reshape(feature_maps.shape[1], feature_maps.shape[2], 1)
        plt.imshow(index_map, cmap='gist_ncar')
    plt.show()


def generate_result_name(l0, l1, dataset, training_count, n_count, freeze):
    name = "results/"+dataset + "_"
    name += str(training_count) + "_"
    name += str(n_count) + "_"
    name += str(freeze) + "_"
    if l0 != None:
        name += "l0_"+str(l0.filters) + "_"
    if l1 != None:
        name += "l1_"+str(l1.filters) + "_"
    return name


def execute_exp(dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count):
    #execute experiment based on parameters
    #initialize model
    new_model = extensible_CNN_layer_multi_module_3D(kernel_size = (kernel_size_0,kernel_size_0,initial_channel), stride = stride, activation = activation)


    #initialize visualisation variables
    images_x_0_9_all = []
    for i in range(10):
        images_x_0_9_all.append(train_images[train_labels.flatten() == i])

    #only get 1 image from each class
    images_x_0_9 = []
    for i in range(10):
        images_x_0_9.append(images_x_0_9_all[i][0])

    images_x_0_9 = np.array(images_x_0_9).reshape(10, 1, image_size, image_size, images_x_0_9_all[0].shape[3],1).astype(np.float32)   
    initialize_x_0_9(images_x_0_9)


    #train model
    images_CNN_train = train_images[:training_count].reshape(training_count, 1, image_size, image_size, images_x_0_9_all[0].shape[3], 1).astype(np.float32)
    epochs = 30
    for i in tqdm.tqdm(range(epochs)):
        new_model, max_inactive_ratio = generate_model_on_images(images_CNN_train, new_model, images_x_0_9, n = 7)
        if max_inactive_ratio < 0.1:
            break
        print("max_inactive_ratio: ", max_inactive_ratio)

    #visualisation
    #print("Visualisation")
    #show_all_index_maps_3D(new_model, images_x_0_9)

    #get integrated model
    l0 = new_model.get_aggregated_conv_2D()

    model_integration = keras.Sequential([
        l0,
        #keras.layers.MaxPooling2D((2, 2)),
        #l1,
        #keras.layers.BatchNormalization(),
        #keras.layers.Conv2D(64, (3, 3), strides=1, activation='gelu'),
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (4, 4), strides=1, activation='sigmoid'),
        #dense layers

        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),

        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(class_count, activation='softmax')
    ])



    model_integration.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])


    #freeze layers
    l0.trainable = not freeze
    print("l0 trainable: ", l0.trainable)
    history_integration = model_integration.fit(train_images[0:n_count], train_labels[0:n_count], epochs=40, validation_data=(test_images, test_labels))

    #save the history and the model
    #get history name
    history_name = generate_result_name(l0, l1, dataset, training_count, n_count, freeze)
    #create the folder if it does not exist
    if not os.path.exists(history_name):
        os.makedirs(history_name)

    #save the history
    import pickle
    with open(history_name + "/history_integration", 'wb') as file_pi:
        pickle.dump(history_integration.history, file_pi)
    #save the model
    model_integration.save(history_name + "/model_integration")



    #train a comparison model
    filter_count = len(new_model.filter_list)
    #filter_count = 64
    #n_count = 50000

    l0_ord = tf.keras.layers.Conv2D(filter_count, (kernel_size_0, kernel_size_0), strides=stride, activation=activation, input_shape=(image_size, image_size, initial_channel))

    backend_ord = keras.Sequential([
        l0_ord,
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (4, 4), strides=1, activation='sigmoid'),
        keras.layers.MaxPooling2D((2, 2)),
        #dense layers
        keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(class_count, activation='softmax')
    ])

    backend_ord.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    print("Images for training: ", n_count)
    history_ord = backend_ord.fit(train_images[0:n_count], train_labels[0:n_count], epochs=40, validation_data=(test_images, test_labels))

    #save the history and the model
    with open(history_name + "/history_ord", 'wb') as file_pi:
        pickle.dump(history_ord.history, file_pi)
    #save the model
    backend_ord.save(history_name + "/model_ord")



def execute_exp_fewshot(dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count):
    #execute experiment based on parameters
    #initialize model
    new_model = extensible_CNN_layer_multi_module_3D(kernel_size = (kernel_size_0,kernel_size_0,initial_channel), stride = stride, activation = activation)


    #initialize visualisation variables
    images_x_0_9_all = []
    for i in range(10):
        images_x_0_9_all.append(train_images[train_labels.flatten() == i])

    #only get 1 image from each class
    images_x_0_9 = []
    for i in range(10):
        images_x_0_9.append(images_x_0_9_all[i][0])

    images_x_0_9 = np.array(images_x_0_9).reshape(10, 1, image_size, image_size, images_x_0_9_all[0].shape[3],1).astype(np.float32)   
    initialize_x_0_9(images_x_0_9)


    #train model
    images_CNN_train = train_images[:training_count].reshape(training_count, 1, image_size, image_size, images_x_0_9_all[0].shape[3], 1).astype(np.float32)
    epochs = 30
    for i in tqdm.tqdm(range(epochs)):
        new_model, max_inactive_ratio = generate_model_on_images(images_CNN_train, new_model, images_x_0_9, n = 7)
        if max_inactive_ratio < 0.1:
            break
        print("max_inactive_ratio: ", max_inactive_ratio)

    #get integrated model
    l0 = new_model.get_aggregated_conv_2D()

    model_integration = keras.Sequential([
        l0,
        #keras.layers.MaxPooling2D((2, 2)),
        #l1,
        #keras.layers.BatchNormalization(),
        #keras.layers.Conv2D(64, (3, 3), strides=1, activation='gelu'),
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (4, 4), strides=1, activation='sigmoid'),
        #dense layers

        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),

        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(class_count, activation='softmax')
    ])



    model_integration.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])


    #freeze layers
    l0.trainable = not freeze
    print("l0 trainable: ", l0.trainable)
    
    #select n_count images for each class for training
    images_to_train = []
    images_labels = []
     
    for i in range(class_count):
        images_to_train.append(train_images[train_labels.flatten() == i][:n_count])
        images_labels.append(train_labels[train_labels.flatten() == i][:n_count])
        
    images_to_train = np.array(images_to_train).reshape(class_count*n_count,image_size, image_size, images_x_0_9_all[0].shape[3]).astype(np.float32)
    images_labels = np.array(images_labels).reshape(class_count*n_count, 1).astype(np.float32)
    
    #train model
    history_integration = model_integration.fit(images_to_train, images_labels, epochs=40, validation_data=(test_images, test_labels))
    
    #save the history and the model
    #get history name
    history_name = generate_result_name(l0, l1, dataset, training_count, n_count, freeze)
    #create the folder if it does not exist
    if not os.path.exists(history_name):
        os.makedirs(history_name)

    #save the history
    import pickle
    with open(history_name + "/history_integration", 'wb') as file_pi:
        pickle.dump(history_integration.history, file_pi)
    #save the model
    model_integration.save(history_name + "/model_integration")
    
    
    #ord model
    filter_count = len(new_model.filter_list)
    l0_ord = tf.keras.layers.Conv2D(filter_count, (kernel_size_0, kernel_size_0), strides=stride, activation=activation, input_shape=(image_size, image_size, initial_channel))

    backend_ord = keras.Sequential([
        l0_ord,
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (4, 4), strides=1, activation='sigmoid'),
        keras.layers.MaxPooling2D((2, 2)),
        #dense layers
        keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(class_count, activation='softmax')
    ])

    backend_ord.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    print("Images for training: ", n_count)
    
    history_ord = backend_ord.fit(images_to_train, images_labels, epochs=40, validation_data=(test_images, test_labels))

    #save the history and the model
    with open(history_name + "/history_ord", 'wb') as file_pi:
        pickle.dump(history_ord.history, file_pi)
    #save the model
    backend_ord.save(history_name + "/model_ord")
    
    

#main

def supervised_single_layer_single_dataset():
    dataset = "cifar10"
    training_count = 100
    min_training_count = 100
    max_training_count = 2600
    step_training_count = 500
    n_count = 50000
    freeze = True
    
    
    for training_count in range(min_training_count, max_training_count+1, step_training_count):
        for freeze in [True, False]:
            dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count  = get_params(dataset=dataset, training_count=training_count, n_count=n_count, freeze=freeze)
    
            execute_exp(dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count)


def supervised_single_layer_all_dataset():
    datasets = ["cifar10", "cifar100", "fashion_mnist", "mnist"]
    training_count = 1300
    n_count = 50000
    freeze = True
    
    for dataset in datasets:
        for freeze in [True, False]:
            dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count  = get_params(dataset=dataset, training_count=training_count, n_count=n_count, freeze=freeze)
    
            execute_exp(dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count)


def few_shot():
    datasets = [ "cifar100"] #redo cifar100

    training_count = 1500
    
    #n_count = 1000 - 20000
    n_count_min = 1000 #n images per class
    n_count_max = 50000/20  #2500
    
    n_count_step = 150
    
    freeze = True
    
    for dataset in datasets:
        if dataset == "cifar100":
            n_count_min = 100
            n_count_max = 25000/100 #250
            n_count_step = 30
        else:
            n_count_min = 1000
            n_count_max = 50000/20
            n_count_step = 150
        for n_count in range(n_count_min, int(n_count_max+1), n_count_step):
            dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count  = get_params(dataset=dataset, training_count=training_count, n_count=n_count, freeze=freeze)
    
            execute_exp_fewshot(dataset, train_images, train_labels, test_images, test_labels, class_count, initial_channel, image_size, kernel_size_0, stride, activation, l0, l1, freeze, n_count, training_count)
        


def transfer_learning(pairs_to_test, training_count, expansion_count, freeze, freeze_ord):
    
    #load dataset for src and target dataset
    src_dataset = pairs_to_test[0]
    target_dataset = pairs_to_test[1]
     
    src_dataset, src_train_images, src_train_labels, src_test_images, src_test_labels, src_class_count, src_initial_channel, src_image_size, src_kernel_size_0, src_stride, src_activation, src_l0, src_l1, src_freeze, src_n_count, src_training_count  = get_params(dataset=src_dataset, training_count=training_count, n_count=50000, freeze=freeze)
    
    target_dataset, target_train_images, target_train_labels, target_test_images, target_test_labels, target_class_count, target_initial_channel, target_image_size, target_kernel_size_0, target_stride, target_activation, target_l0, target_l1, target_freeze, target_n_count, target_training_count  = get_params(dataset=target_dataset, training_count=expansion_count, n_count=50000, freeze=freeze)
    
    
    new_model = extensible_CNN_layer_multi_module_3D(kernel_size = (src_kernel_size_0,src_kernel_size_0,src_initial_channel), stride = src_stride, activation = src_activation)

    
    #takes spaces to visualize the results, so not implemented
    #initialize visualisation variables
    image_size = src_image_size
    images_x_0_9_all = []
    for i in range(10):
        images_x_0_9_all.append(src_train_images[src_train_labels.flatten() == i])

    #only get 1 image from each class
    images_x_0_9 = []
    for i in range(10):
        images_x_0_9.append(images_x_0_9_all[i][0])

    images_x_0_9 = np.array(images_x_0_9).reshape(10, 1, image_size, image_size, images_x_0_9_all[0].shape[3],1).astype(np.float32)   
    initialize_x_0_9(images_x_0_9)
    
    
    image_size = src_image_size 

    #train model
    images_CNN_train = src_train_images[:training_count].reshape(training_count, 1, image_size, image_size, images_x_0_9_all[0].shape[3], 1).astype(np.float32)
    epochs = 30
    for i in tqdm.tqdm(range(epochs)):
        new_model, max_inactive_ratio = generate_model_on_images(images_CNN_train, new_model, images_x_0_9, n = 7)
        if max_inactive_ratio < 0.1:
            break
        print("max_inactive_ratio: ", max_inactive_ratio)

    
    class_count = src_class_count
    train_images = src_train_images
    train_labels = src_train_labels
    test_images = src_test_images
    test_labels = src_test_labels
    kernel_size_0 = src_kernel_size_0
    activation = src_activation
    stride = src_stride
    initial_channel = src_initial_channel
    n_count = src_n_count
    
    
    l0 = new_model.get_aggregated_conv_2D()
    model_integration = keras.Sequential([
        l0,
        #keras.layers.MaxPooling2D((2, 2)),
        #l1,
        #keras.layers.BatchNormalization(),
        #keras.layers.Conv2D(64, (3, 3), strides=1, activation='gelu'),
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.BatchNormalization(),
        keras.layers.Flatten(),

        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(class_count, activation='softmax')
    ])



    model_integration.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])


    #freeze layers
    l0.trainable = not freeze
    
    src_history_integration = model_integration.fit(train_images, train_labels, epochs=40, validation_data=(test_images, test_labels))
    
    filter_count = l0.filters
    l0_ord = tf.keras.layers.Conv2D(filter_count, (kernel_size_0, kernel_size_0), strides=stride, activation=activation, input_shape=(image_size, image_size, initial_channel))

    backend_ord = keras.Sequential([
        l0_ord,
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.BatchNormalization(),
        #dense layers
        keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(class_count, activation='softmax')
    ])

    backend_ord.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    print("Images for training: ", n_count)
    
    src_history_ord = backend_ord.fit(train_images, train_labels, epochs=40, validation_data=(test_images, test_labels))
    l1 = None
    dataset = "Transfer_"+src_dataset + "_" + target_dataset
    #save src histories
    import pickle
    history_name = generate_result_name(l0, l1, dataset, training_count, n_count, freeze)
    
    #create the folder if it does not exist
    if not os.path.exists(history_name):
        os.makedirs(history_name)
    
    with open(history_name + "/src_history_ord", 'wb') as file_pi:
        pickle.dump(src_history_ord.history, file_pi)
        
    with open(history_name + "/src_history_integration", 'wb') as file_pi:
        pickle.dump(src_history_integration.history, file_pi)
        
    
    #now train on target dataset, first by expanding the model, with expansion_count
    images_CNN_train = target_train_images[:expansion_count].reshape(expansion_count, 1, image_size, image_size, images_x_0_9_all[0].shape[3], 1).astype(np.float32)
    
    epochs = 20
    for i in tqdm.tqdm(range(epochs)):
        new_model, max_inactive_ratio = generate_model_on_images(images_CNN_train, new_model, images_x_0_9, n = 7)
        if max_inactive_ratio < 0.1:
            break
        print("max_inactive_ratio: ", max_inactive_ratio)
    
    train_images = target_train_images
    train_labels = target_train_labels
    test_images = target_test_images
    test_labels = target_test_labels
    class_count = target_class_count
    
    l0 = new_model.get_aggregated_conv_2D()
    model_integration = keras.Sequential([
        l0,
        #keras.layers.MaxPooling2D((2, 2)),
        #l1,
        #keras.layers.BatchNormalization(),
        #keras.layers.Conv2D(64, (3, 3), strides=1, activation='gelu'),
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.BatchNormalization(),
        keras.layers.Flatten(),

        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(class_count, activation='softmax')
    ])



    model_integration.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])


    #freeze layers
    l0.trainable = not freeze
    
    target_history_integration = model_integration.fit(train_images, train_labels, epochs=40, validation_data=(test_images, test_labels))
    
    filter_count = l0.filters
    
    if not freeze_ord:
        pass #not frozen
    else:
        l0_ord.trainable = False
        
    backend_ord = keras.Sequential([
        l0_ord,
        keras.layers.MaxPooling2D((2, 2)),
        #dense layers
        keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='sigmoid', name = "ord_target_dense"),
        tf.keras.layers.Dense(class_count, activation='softmax', name = "output_ord_target")
    ])

    backend_ord.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    print("Images for training: ", n_count)
    
    target_history_ord = backend_ord.fit(train_images, train_labels, epochs=40, validation_data=(test_images, test_labels))
        
    #save histories
    with open(history_name + "/target_history_ord", 'wb') as file_pi:
        pickle.dump(target_history_ord.history, file_pi)

    with open(history_name + "/target_history_integration", 'wb') as file_pi:
        pickle.dump(target_history_integration.history, file_pi)
    
    
    pass
    


def transfer_learning_exp():
    pairs_to_test = [ #[src, target]
        ["cifar10", "cifar100"]#, #simple layer 0 features are similar, making it not significant for single layer experiments
        #["mnist", "fashion_mnist"]
    ]
    
    training_count = 1500 #1500 for src dataset to train l0
    expansion_count = 1000 #1000 for target dataset to train l0
    
    #usually freezed mode would give better results
    freeze = True
    
    for pair in pairs_to_test:
        transfer_learning(pair, training_count, expansion_count, freeze, freeze_ord = True)
    


if __name__ == "__main__":
    #get the parameters
    #supervised_single_layer_all_dataset()
    #few_shot()
    transfer_learning_exp()