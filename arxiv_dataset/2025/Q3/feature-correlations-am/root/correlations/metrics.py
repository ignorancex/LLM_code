import dcor
import mlx.core as mx
import numpy as np
from root.correlations.metric import Metric


class HammingDistance(Metric):
    @staticmethod
    def calculate(row1, row2):
        """
        Calculate the hamming distance between two rows
        """
        row1 = mx.array(row1)
        row2 = mx.array(row2)
        val = mx.sum(row1 != row2)
        return float(val)


class BinaryMinPairwiseHammingDistance(Metric):
    @staticmethod
    def calculate(data, batch_size=1000):
        """
        Calculate the minimum pairwise binary hamming distance for a dataset using batch processing.
        Values in the dataset should be either -1 or 1.
        """
        data = mx.array(data)
        num_rows = data.shape[0]
        min_distance = float("inf")

        # Iterate over batches to reduce memory usage
        for batch_start in range(0, num_rows, batch_size):
            batch_end = min(batch_start + batch_size, num_rows)
            batch_data = data[batch_start:batch_end]
            for i in range(batch_data.shape[0]):
                # Compute Hamming distance: count of differing elements
                distances = mx.sum(batch_data[i] != data, axis=1)
                relevant_distances = distances[batch_start + i + 1 : batch_end]
                if relevant_distances.size > 0:
                    min_batch_distance = mx.min(relevant_distances)
                    if min_batch_distance < min_distance:
                        min_distance = min_batch_distance

        if min_distance == float("inf"):
            return 0

        return float(min_distance)


class BinaryMeanPairwiseHammingDistance(Metric):
    @staticmethod
    def calculate(data, batch_size=1000):
        """
        Calculate the mean pairwise binary hamming distance for a dataset using batch processing.
        Values in the dataset should be either -1 or 1.
        """
        data = mx.array(data)
        rows = data.shape[0]
        total_distance = 0
        count = 0

        # Iterate over batches to reduce memory usage
        for start in range(0, rows, batch_size):
            end = min(start + batch_size, rows)
            batch_data = data[start:end]
            for i in range(batch_data.shape[0]):
                # Compute Hamming distance: count of differing elements
                distances = mx.sum(batch_data[i] != data, axis=1)
                total_distance += mx.sum(
                    distances[start + i + 1 : end]
                )  # Avoid double-counting pairs
                count += len(distances[start + i + 1 : end])

        if count == 0:
            return 0

        mean_distance = total_distance / count

        return float(mean_distance)


class BinaryMeanPairwiseColHammingDistance(Metric):
    @staticmethod
    def calculate(data, batch_size=1000):
        """
        Calculate the mean pairwise binary hamming distance for a dataset over columns using batch processing.
        Values in the dataset should be either -1 or 1.
        """
        data = mx.array(data)
        cols = data.shape[1]  # Get the number of columns
        total_distance = 0
        count = 0

        # Iterate over column batches to reduce memory usage
        for start in range(0, cols, batch_size):
            end = min(start + batch_size, cols)
            batch_data = data[:, start:end]
            for i in range(batch_data.shape[1]):
                # Compute Hamming distance: count of differing elements for each column pair
                distances = mx.sum(batch_data[:, i : i + 1] != data, axis=0)
                total_distance += mx.sum(
                    distances[start + i + 1 : end]
                )  # Avoid double-counting pairs
                count += len(distances[start + i + 1 : end])

        if count == 0:
            return 0

        mean_distance = total_distance / count

        return float(mean_distance)


class MeanPairwiseDCor(Metric):
    @staticmethod
    def calculate(data, inverse=False):
        """
        Calculate the average correlation for the given data.
        """
        dist_corr_matrix = MeanPairwiseDCor._compute_distance_corr_matrix(data)
        return (
            float(mx.mean(dist_corr_matrix))
            if not inverse
            else float(1 - mx.mean(dist_corr_matrix))
        )

    @staticmethod
    def _compute_distance_corr_matrix(patterns: mx.array):
        data = np.array(patterns, dtype=float)
        n = data.shape[0]  # number of samples (rows)
        dist_corr_matrix = np.zeros((n, n))  # Initialize empty matrix

        for i in range(n):
            for j in range(n):
                if i != j:  # Avoid self-correlation
                    dist_corr_matrix[i, j] = dcor.distance_correlation(
                        data[i, :], data[j, :]
                    )
                dist_corr_matrix[j, i] = dist_corr_matrix[i, j]  # Symmetry

        return mx.array(dist_corr_matrix).astype(mx.float32)


class ContinousMeanPairwiseHammingDistance(Metric):
    @staticmethod
    def calculate(data):
        """
        Calculate the continous mean pairwise hamming distance for a dataset
        """
        data = mx.array(data)
        rows = data.shape[0]
        total_distance = 0
        count = 0

        for i in range(rows):
            for j in range(i + 1, rows):
                x1 = data[i]
                x2 = data[j]
                if x1 == x2:
                    return 0
                separation = abs(x1 - x2)
                # how many bits do I need to represent this?
                bits = mx.ceil(mx.log2(2 / separation))
                total_distance += bits
                count += 1

        if count == 0:
            return 0

        mean_distance = total_distance / count
        return mean_distance
