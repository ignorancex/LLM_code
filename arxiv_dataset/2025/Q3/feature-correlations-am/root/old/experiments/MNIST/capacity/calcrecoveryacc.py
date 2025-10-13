import math
from random import randint

import numpy as np
import matplotlib.pyplot as plt
from root.experiments.MNIST.information.hamming import hamming_distance

from root.utils import plot_img
from root.model.hopfield import HopfieldNetwork

#1 Memories
total_memories = 4_000
num_neurons = 16
num_examples = [1,2,4,5,10,20,50,100]

#2 setup
max_polynomial = 20

def get_random_idxs(num_memories):
    """
    @param: num_memories, num_images_per_class
    @returns a list of lists of indexes of the memories
    """
    num_images_per_class = int(num_memories/2)

    num_test_images_per_class = min(10, num_images_per_class)
    
    zero_idxs = np.random.randint(0,num_images_per_class,num_test_images_per_class)
    one_idxs = np.random.randint(num_images_per_class,num_memories,num_test_images_per_class)
    idxs = np.concatenate((zero_idxs,one_idxs))
    return idxs

def get_recall_quality(memories, polydegree, num_neurons, network_at_maxcap = False, is_continous = False, plot_updated_images = False, num_updates = 1, verbose = False, test_idxs = [], corrupt = False, add_noise_bits = 0):
    """
    Takes a minimum of 2 patterns
    """
    recall_quality = 0
    
    isImage = True if math.sqrt(num_neurons) % 2 == 0 else False
    
    #train Hopfield net
    network = HopfieldNetwork(num_neurons, polydegree, max_cap = network_at_maxcap, continous=is_continous)
    network.learn(memories)

    idxs = []
    if len(test_idxs) == 0:
        idxs = get_random_idxs(memories.shape[0])
    else:
        idxs = np.array(test_idxs)

    for idx in idxs:
        memory = memories[idx,:]

        if plot_updated_images:
            if isImage:
                dims = int(math.sqrt(num_neurons))
                image = memory.reshape(dims,dims) #correct image if printed
                plt.imshow(image)
            else:
                print(memory)
            plt.show()

        for i in range(num_updates):
            if i == 0:
                if corrupt:
                    noisy_memory = memory.copy()
                    noise_idxs = np.arange(num_neurons/2,num_neurons, dtype=np.int32)
                    np.put(noisy_memory, noise_idxs, -1.0)
                    network.update(noisy_memory)
                if add_noise_bits != 0:
                    noisy_memory = memory.copy() 
                    
                    #I randomly select bits I want to add noise to
                    #Then at those indexes, I flip the bit (noise)

                    np.random.seed(0) #ensure same noise and deterministic behavior
                    noise_idxs = np.random.choice(num_neurons, add_noise_bits, replace=False)
                    noisy_memory[noise_idxs] = np.where(noisy_memory[noise_idxs] == -1, 1, -1)                    
                    network.update(noisy_memory)
                else:
                    network.update(memory) #should also be correct as flattening works as expected
            else: 
                network.update(network.excitation)

        restored = network.get_state()
        if plot_updated_images:
            if isImage:
                dims = int(math.sqrt(num_neurons))
                image = restored.reshape(dims,dims) #correct image if printed
                plt.imshow(image)
            else:
                print(memory)
            plt.show()

        #MacKay: if restored version (stable state) has 50% of bits flipped (compared to the original image), the recall performance is 0 (not recognizable)
        #scaled inner product of memory & restored memory by num_neurons
        performance = np.dot(memory, restored) / num_neurons 
        if verbose:
            print("restore performance", performance)

        recall_quality = recall_quality + performance

    average_recall_quality = recall_quality/idxs.shape[0]
    
    return average_recall_quality



def get_recall_qualities(memories, polydegrees, num_neurons, network_at_maxcap = False, is_continous = False, plot_updated_images = False, num_updates = 1):
    """
    @param: memories. 2-D numpy float array First dimension: example, Second dimension: features (if image, flattened)
    @param: polydegrees, array, of polydegrees for which the recall quality should be performed
    @param: num_neurons, integer. number of neurons in the Hopfield network used for testing. Equivalent to memory_dimension
    @param: plot_updated_images, Boolean

    @return: recall_qualities, Array of floats. 
    """
    
    recall_qualities = []
    dims = int(math.sqrt(num_neurons))

    for polydegree in polydegrees:
        #train Hopfield net
        network = HopfieldNetwork(num_neurons, polydegree, max_cap = network_at_maxcap, continous=is_continous)
        network.learn(memories)
        #fine

        num_memories = memories.shape[0] #works as expected
        zero_idxs = np.random.randint(0,int(num_memories/2),5)
        one_idxs = np.random.randint(int(num_memories/2),num_memories,5)
        idxs = np.concatenate((zero_idxs,one_idxs))

        avg_recall_quality = 0

        for idx in idxs:

            image = memories[idx, :].reshape(dims,dims) #correct image if printed
            if plot_updated_images:
                plt.imshow(image)
                plt.show()

            for i in range(num_updates):
                if i == 0:
                    network.update(image.flatten()) #should also be correct as flattening works as expected
                else: 
                    network.update(network.excitation)

            restored = network.get_state().reshape(dims,dims)
            if plot_updated_images:
                plt.imshow(restored)
                plt.show()


            #MacKay: if restored version (stable state) has 50% of bits flipped (compared to the original image), the recall performance is 0 (not recognizable)
            #scaled inner product of memory & restored memory by num_neurons
            recall_quality = np.inner(image.flatten(), restored.flatten()) / num_neurons 

            avg_recall_quality = avg_recall_quality + recall_quality
        
        avg_recall_quality = avg_recall_quality/idxs.shape[0]

        recall_qualities.append(avg_recall_quality)

    return recall_qualities

def main():

    for examples_per_class in num_examples:
    #3 create random memories
        num_classes = int(total_memories/examples_per_class)
        random = np.random.randint(0,2,num_classes*num_neurons)
        randomarray = np.where(random == 0, -1, random)

        #extend the random data #num_examples of times
        random_ints = np.tile(randomarray, examples_per_class)

        #store as float, as well
        random_floats = np.array(random_ints, dtype=np.float64)

        #constant memories
        #constantarray = np.random.randint(1,2,num_unique_memories*num_neurons)

        #int_memories = np.reshape(random_ints,(num_examples*num_unique_memories,num_neurons))
        float_memories = np.reshape(random_floats,(total_memories,num_neurons))
        #int_constant_memories = np.reshape(constantarray, (num_unique_memories, num_neurons))
        #float_constant_memories = np.array(int_constant_memories, dtype=np.float64)

        #print("int", int_memories)
        print("float", float_memories)
        #print("const int", int_constant_memories)
        #print("const float", float_constant_memories)

        #for plot
        polydegrees = np.arange(1,max_polynomial) #x

        #ints_recall_qualities = get_recall_qualities(int_memories, polydegrees, num_neurons)
        float_recall_qualities = get_recall_qualities(float_memories, polydegrees, num_neurons)
    #   constant_int_recall_qualities = get_recall_qualities(int_constant_memories, polydegrees, num_neurons)
    #   constant_float_recall_qualities = get_recall_qualities(float_constant_memories, polydegrees, num_neurons)

        plt.plot(polydegrees, float_recall_qualities, label="examples per class: {}".format(examples_per_class))

    plt.legend(loc='best')
    plt.show()

    #plt.plot(polydegrees, constant_int_recall_qualities, constant_float_recall_qualities)

if __name__ == "__main__":
    main()