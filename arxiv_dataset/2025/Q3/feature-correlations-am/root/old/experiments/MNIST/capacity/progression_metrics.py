import matplotlib.pyplot as plt
import numpy as np

from root.infocapacity.estimate_cap import estimate_cap

from root.experiments.MNIST.digits.subsets import get_fifty_random_images, get_first_fifty_images

from root.experiments.MNIST.capacity.definitions import DATASET_SHARE

num_neurons = 784
num_memories = 100

def calc_thresholds(first_set, second_set = None):
    log_thresholds = []

    for share in DATASET_SHARE:
        first_subset = first_set[0:int(share*50), :]
        
        if second_set is not None:
            second_subset = second_set[0:int(share*50), :]
            set = np.concatenate((first_subset, second_subset))
        else:
            set = first_subset
            
        log_threshold, _ = estimate_cap(set)

        log_thresholds.append(log_threshold)

    log_thresholds = np.array(log_thresholds, dtype=np.float64)
    
    return log_thresholds

def calc_prob_density():
    #TODO

    zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)
    
    #binary_set = np.array(np.where(unary_set == -1, 0, unary_set), dtype=np.float64)

    #p_six_eights = np.sum(binary_set, axis=0)
    pass

#test code
def main():
    zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)
    print(calc_thresholds(sixes, eights))

    #random images is a bit of legacy code – it's not deterministic
    zeros, ones, sixes, eigths = get_fifty_random_images(inBinary = True)
    print(calc_thresholds(zeros, ones))

if __name__ == "__main__":
    main()