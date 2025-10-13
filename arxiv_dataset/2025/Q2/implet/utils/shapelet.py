import math
import numpy as np

AEON_NUMBA_STD_THRESHOLD = 1e-8
def z_norm_process_shapelet(shapelet):
    length = len(shapelet)
    Sum = 0.0
    Sum2 = 0.0
    for i in shapelet:
        Sum += i
        Sum2 += i * i

    mean = Sum / length

    # std = math.sqrt((Sum2 - mean * mean * length) / length)
    std = np.std(shapelet)
    if std > AEON_NUMBA_STD_THRESHOLD:
        shapelet = (shapelet - mean) / std
    else:
        shapelet = np.zeros(length)
    return shapelet
def compute_shapelet_distance(series, shapelet, length, position):
    series = series.flatten()
    if position is None:
        position = 0
    # shapelet = z_norm_process_shapelet(shapelet)

    sabs = np.abs(shapelet)
    sorted_indicies = np.array(
        sorted(range(len(shapelet)), reverse=True, key=lambda j: sabs[j])
    )
    subseq = series[position: position + length]

    Sum = 0.0
    Sum2 = 0.0
    for i in subseq:
        Sum += i
        Sum2 += i * i
    #
    # mean = Sum / length
    #
    # # std = math.sqrt((Sum2 - mean * mean * length) / length)
    # std = np.std(subseq)
    # if std > AEON_NUMBA_STD_THRESHOLD:
    #     subseq = (subseq - mean) / std
    # else:
    #     subseq = np.zeros(length)

    best_dist = 0
    best_pos = position

    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [Sum, Sum]
    sums2 = [Sum2, Sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            # start = series[pos - n]
            # end = series[pos - n + length]

            # sums[n] += mod * end - mod * start
            # sums2[n] += mod * end * end - mod * start * start

            # mean = sums[n] / length

            # std = math.sqrt((sums2[n] - mean * mean * length) / length)
            # std = np.std(series[pos:pos+length])
            dist = 0
            # use_std = std > AEON_NUMBA_STD_THRESHOLD

            for j in range(length):
                val = series[pos + sorted_indicies[j]]
                # val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_pos = pos  # [use_std, series[pos:pos+length], shapelet, pos, mean, std]
                best_dist = dist

        i += 1
    return best_dist if best_dist == 0 else 1 / length * best_dist, best_pos


# def compute_shapelet_distance(instance, shapelet, is_normalize=True, loss_func=np.linalg.norm):
#     def z_norm(array):
#         return (array - np.mean(array)) / np.std(array)
#
#     instance = instance.flatten()
#     n = len(instance)
#     m = len(shapelet)
#
#     shapelet_distance = float('inf')
#     best_location = None
#     # if is_normalize:
#     #     shapelet = z_norm(shapelet)
#
#     for i in range(n - m + 1):
#         subsequence = instance[i:i + m].copy()
#         mean = np.mean(subsequence)
#         std = np.std(subsequence)
#         if is_normalize:
#             subsequence = z_norm(subsequence)
#         distance = loss_func(subsequence - shapelet) / m
#
#         if shapelet_distance > distance:
#             shapelet_distance = distance
#             best_location = [subsequence,shapelet,i,mean,std]
#
#     return shapelet_distance, best_location


def entropy(labels):
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


def information_gain_for_shapelet(shapelet_distances, labels):
    original_entropy = entropy(labels)

    num_instance = len(labels)

    # Sort shapelet_distance and generate potential thresholds (midpoints between sorted shapelet_distance)
    sorted_distances = np.sort(shapelet_distances)
    potential_thresholds = (sorted_distances[:-1] + sorted_distances[1:]) / 2

    # Track the best information gain
    best_info_gain = 0
    best_threshold = None
    # Evaluate each threshold to find the maximum information gain
    for threshold in potential_thresholds:
        below_threshold = labels[shapelet_distances <= threshold]
        above_threshold = labels[shapelet_distances > threshold]

        # Calculate entropies for each subset
        below_entropy = entropy(below_threshold) if len(below_threshold) > 0 else 0
        above_entropy = entropy(above_threshold) if len(above_threshold) > 0 else 0

        # Calculate weighted entropy and information gain
        weighted_entropy = (len(below_threshold) / num_instance) * below_entropy + (
                len(above_threshold) / num_instance) * above_entropy
        info_gain = original_entropy - weighted_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_threshold = threshold

    return best_info_gain, best_threshold


def get_distances_info_gain(dataset, shapelet, length, pos, labels):
    num = dataset.shape[0]
    dataset = dataset.reshape(num, -1)
    shapelet_distances = np.zeros(num)
    for i in range(num):
        shapelet_distances[i], _ = compute_shapelet_distance(dataset[i], shapelet, length=length, position=pos)
    info_gain, best_threshold = information_gain_for_shapelet(shapelet_distances, labels)
    return shapelet_distances, info_gain, best_threshold
