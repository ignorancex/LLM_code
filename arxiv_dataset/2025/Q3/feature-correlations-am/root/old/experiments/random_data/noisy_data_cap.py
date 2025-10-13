import math
import matplotlib.pyplot as plt
import numpy as np

from root.infocapacity.estimate_cap import estimate_cap

def estimate_n(table):
    """
    Takes: 2D numpy array representing an unlabeled dataset with rows as data points and columns as features.

    Returns: the approximated n for the dataset assuming 0% retrieval error

    Intuition: if the dataset contains less overlapping / complicated memories (orthogonal memories), n should be smaller because the hopfield net doesn't have to remember as many memories 
    but can extract features instead

    More thresholds => more complexity in the set => higher n required
    """
    row_sums = []

    for row in table:
        row_sums.append(np.sum(row))
    row_sums.sort(reverse=True)

    #count thresholds in both
    row_thresholds = 1

    curr_threshold = row_sums[0]
    for num in row_sums:
        #what if we vary the threshold equality to an inequalty?
        #if curr_threshold
        if num != curr_threshold:
            curr_threshold = num
            row_thresholds = row_thresholds + 1

    return math.log(row_thresholds)/math.log(2)


def main():

    random = np.random.randint(0, 2, 1000)
    randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)
    data = np.reshape(randomarray, (10,100))
    
    duplicateddata = np.tile(data, (2,1))

    #10% noise to duplicates
    noise_percentage = 0.2

    noise_1000 = np.array([0.] * int(1000 * (1-noise_percentage)) + [1.] * int(1000*noise_percentage/2) + [-1.] * int(1000*noise_percentage/2)) 

    np.random.shuffle(noise_1000) #shuffle noisy bits
    noisydata = np.where(noise_1000 == 0, randomarray, noise_1000)

    noisy_duplicated = np.reshape(np.concatenate((randomarray,noisydata)), (20,100))

    cap_data, _ = estimate_cap(data)
    cap_duplicated, _ = estimate_cap(duplicateddata)
    cap_noisy, _ = estimate_cap(noisy_duplicated)

    print(cap_data)
    print(cap_duplicated)
    print(cap_noisy)

if __name__ == "__main__":
    main()
