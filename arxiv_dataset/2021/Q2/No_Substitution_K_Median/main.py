import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import Birch
import math


def birch_skm_part1_helper(data, m, k, delta):
    """
    The function receive data and calculates k centers using the birch function in sklearn, and their quantile radius
    :param data: numpy array
    :param m: Size of the data
    :param k: Number of centers.
    :param delta: int
    :return: tuple of two numpy array. (k_medoids, k_distances).
    """
    birch_instance = Birch(n_clusters=k, threshold=0.1)  # birch instance
    birch_instance.fit(data)  # Run birch on the data
    labels = birch_instance.predict(data) # calculate the cluster number for each point
    l_medoids = []
    # since birch does not return centers, I have to calculate them
    for label in range(
            np.unique(labels).size):
        # calculate the center for each cluster
        cluster = data[labels == label]
        kmedoids_instance_for_birch = kmedoids(cluster.tolist(), init_centers(cluster, 1))
        kmedoids_instance_for_birch.process()
        l_medoids.append(cluster[kmedoids_instance_for_birch.get_medoids()][0])
    l_medoids = np.array(l_medoids)
    q = calc_q(m, delta)  # calculate q
    # calculate the distance to the quantile points around each center
    l_distances = calc_quantile_radius_around_centers(data, l_medoids, q, k)
    return l_medoids, l_distances


def kmedoids_skm_part1_helper(data, m, k, delta):
    """
    The function receive data and calculates k centers using the kmedoids library, and their quantile radius
    :param data: numpy array
    :param m: Size of the data
    :param k: Number of centers.
    :param delta: int
    :return: tuple of two numpy array. (k_medoids, k_distances).
    """
    data_lists = data.tolist()  # modify to list of lists, so it
    # Create instance of K-Medoids algorithm.
    kmedoids_instance = kmedoids(data_lists, init_centers(data, k))
    # Run cluster analysis and obtain results.
    kmedoids_instance.process()
    l_medoids = data[kmedoids_instance.get_medoids()]  # calculate the medoids.
    q = calc_q(m, delta)  # calculate q
    # calculate the distance to the quantile points around each center
    l_distances = calc_quantile_radius_around_centers(data, l_medoids, q, k)
    return l_medoids, l_distances


def skm1part1(data, m, k, delta, birch_kmediods_mode=0):
    """
    The  estimation(first) part of SKM1.
    :param birch_kmediods_mode: int. if 0 then black box is birch, otherwise black box ix k_medoids.
    :param data: Numpy array
    :param m: Size of the data
    :param k: Number of centers.
    :param delta: confidence parameter
    :return: Return the mediods calculated by the approximation offline algorithm and the quantile radius
    """
    if birch_kmediods_mode == 0: # run birch
        return birch_skm_part1_helper(data, m, k, delta)
    if birch_kmediods_mode == 1: # run k-medoids
        return kmedoids_skm_part1_helper(data, m, k, delta)



def calc_q(m, delta):
    """
    return the quantile q, given m and delta.
    :param m: Size of the data
    :param delta: int
    :return: q: quantile size
    """
    print(' this is m:%s' % m)
    q = (9 * math.log((2 * m ** 2) / delta)) / m
    if q < 1:
        return q
    else:
        print('q is bigger than !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return 1


def calc_radius_point(data, center, q):
    """
    Return the point that is q-quantile point with regards to the distance from center, i.e., qp_S{center,q}.
    :param data: numpy array
    :param center: A point
    :param q: quantile
    :param m: size of the data
    :return: float
    """
    l_distance = np.linalg.norm(data - np.array([center]),
                                axis=1)  # Distance of every point in data to center. Note unsorted distance list
    l_distance = np.sort(l_distance)  # sort in ascending order
    q_rounding = math.ceil(
        l_distance.shape[0] * q) - 1  # calculate the exact location of the distance to the quantile point
    return l_distance[q_rounding]  # return the quantile point


def calc_quantile_radius_around_centers(data, l_centers, q, k):
    """
    Calculate the radius around each center, i.e., d(c,qp_S(c,q))
    :param data:
    :param l_centers: k points
    :param q: Quantile
    :param m: Size of the data
    :param k: Number of centers.
    :return: quantile points for every center ->list of list of floats.
    """
    l_quantile_points = []
    # Calculate the quantile points for every center.
    for i in range(k):
        l_quantile_points.append(calc_radius_point(data, l_centers[i], q))  # Calculate the quantile radius
    return l_quantile_points


def skm1part2(data, l_centers, l_distances, k):
    """
    The second part of SKM1, The selection part. The first point in the stream, which fall inside the quantile
     ball of a center, is selected in-order to replace this center. We do this k times (for ech center).
    :param data: numpy array
    :param l_centers: numpy array, k centers from skm part 1.
    :param l_distances: numpy array, the quantile radius of each center in l_centers
    :param k: number of clusters
    :return: If algorithm success return k centers(List of k selected points), otherwise return " Failed".
    """
    output = np.ones((k, l_centers.shape[1])) * 255  # initialize the output.
    # init the list which indicates if we already selected a point in the ball of center i
    l_found_point = [False for _ in range(k)]  # This list indicates whether we already choose point for cluster i.
    output_size = 0  # indicator that count the number of selected items,
    # it is important in-order to understand if the algorithm failed in the end of the run.
    for x in data:  # run over the  data sequentially.
        # check if a point is inside one of the quantile balls of one of the centers.
        for i, (center, distance) in enumerate(zip(l_centers, l_distances)):
            if (l_found_point[i] == 0) and (np.linalg.norm(x - center) <= distance):  # check for center[i]
                # select a point
                output[i] = np.array(x)  # replace  center i with the selected medoid
                # print(output)
                l_found_point[i] = True
                output_size += 1
                # print("found center number %s, this is the center: %s" % (output_size, x))
                if output_size == k:  # Found point for every cluster.
                    print(' i am here. Return from skm part 2')
                    return output
    print("Algorithm failed: Did not find k centers")  # Algorithm failed since we did not find k centers.

    return output[np.array(l_found_point)]


def init_centers(data, k):
    """
    Pick k random points from the data
    :param data: numpy array
    :param k: int - number of centers
    :return: List of k integers ->  k random indices from the data.
    """
    init_ids = []
    while len(init_ids) < k:
        sample = np.random.randint(0, data.shape[0])  # draw one point from the data
        if sample not in init_ids:
            init_ids.append(sample)
    return init_ids


def skm1(data, k, delta, birch_kmediods_mode=0):
    """
    SKM. Sequential algorithm choose k medoids without substitution.
    :param data: numpy array
    :param k: int - number of centers
    :param delta: int
    :return: The chosen medoids and the centers returned from the offline algorithm (skmpart1).
    """
    m = data.shape[0]
    phase_size = round(m / 2)  # divide the sequence into half.
    # running an offline algorithm on the first half of the stream in order to compute the centers
    l_centers, l_distances = skm1part1(data[:phase_size], m, k, delta,
                                       birch_kmediods_mode)  # phase 1 - offline algorithm
    # Second part: The selection part. select the k medoids
    output = skm1part2(data[phase_size:], l_centers, l_distances, k)  # phase 2 - selection phase
    return output, l_centers


def calculate_inertia(data, l_centers):
    """
    Calculate the k-mediods risk
    :param data: numpy array
    :param l_centers: numpy array. array of centers.
    :return:
    """
    label = []
    for features in data:
        distances = [np.linalg.norm(features - centroid) for centroid in l_centers]
        label.append(np.argmin(distances))
    return np.mean(np.linalg.norm(data - l_centers[label], axis=1), dtype=np.float64)


def main():
    return


if __name__ == '__main__':
    main()
