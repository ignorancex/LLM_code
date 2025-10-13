import mlx.core as mx

from root.correlations.metrics import (
    BinaryMeanPairwiseColHammingDistance,
    BinaryMeanPairwiseHammingDistance,
    BinaryMinPairwiseHammingDistance,
    HammingDistance,
    MeanPairwiseDCor,
)


class Correlations:
    @staticmethod
    def calc_average_hd(patterns: mx.array):
        return BinaryMeanPairwiseHammingDistance.calculate(patterns)

    def calc_average_hd_cols(patterns: mx.array):
        return BinaryMeanPairwiseColHammingDistance.calculate(patterns)

    @staticmethod
    def calc_min_hd(patterns: mx.array):
        return BinaryMinPairwiseHammingDistance.calculate(patterns)

    @staticmethod
    def calc_hd(array1, array2):
        return HammingDistance.calculate(array1, array2)

    @staticmethod
    def calc_average_dcor(patterns: mx.array, inverse=False):
        return MeanPairwiseDCor.calculate(patterns, inverse)

    @staticmethod
    def calc_empirical_covariance(patterns: mx.array) -> mx.array:
        mean = mx.mean(patterns, axis=0, keepdims=True)
        centered = patterns - mean
        C = mx.divide(mx.matmul(centered.T, centered), patterns.shape[0])
        return C

    @staticmethod
    def calc_mean_vector(patterns: mx.array) -> mx.array:
        return mx.mean(patterns, axis=0, keepdims=True)

    @staticmethod
    def calc_mean_vector_avg(patterns: mx.array) -> mx.array:
        # adjust
        mean_vect = mx.mean(patterns, axis=0, keepdims=True)
        return mx.mean(mean_vect, axis=1, keepdims=True)

    @staticmethod
    def calc_eigenvalue_spectra(patterns: mx.array) -> mx.array:
        mx.set_default_device(mx.cpu)
        eigenvalues, eigenvectors = mx.linalg.eigh(
            Correlations.calc_empirical_covariance(patterns)
        )
        # sort eigvals
        indexes = mx.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indexes]
        eigenvectors = eigenvectors[:, indexes]

        return eigenvalues
