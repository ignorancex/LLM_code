import mlx.core as mx
from joblib import Parallel, delayed
from tqdm import tqdm

from root.model.hopfield import HopfieldNetwork


class Eval:
    def __init__(self, datasets, model):
        self.datasets: mx.array = datasets
        self.model: HopfieldNetwork = model

    def get_k_maxes(self) -> mx.array:
        """
        Get k_maxes for all subsets of the dataset
        """
        results = []

        for _, dataset in tqdm(
            enumerate(self.datasets),
            total=self.datasets.shape[0],
            desc="Calculating k_maxes",
        ):
            results.append(self._calc_k_max(dataset))

        return mx.array(results).astype(mx.float32)

    def get_k_maxes_parallel(self, threshold=None, stop_count=None) -> mx.array:
        def debug_calc_k_max(dataset):
            k_max = self._calc_k_max(dataset)
            print(f"Calculated k_max: {k_max} for dataset with shape {dataset.shape}")
            return k_max

        results = []
        threshold_reached_count = 0

        for dataset in tqdm(self.datasets, desc="Calculating k_maxes"):
            k_max = debug_calc_k_max(dataset)
            results.append(k_max)

            if threshold is not None and stop_count is not None:
                if k_max >= threshold:
                    threshold_reached_count += 1
                if threshold_reached_count >= stop_count:
                    print(
                        f"Threshold of {threshold} reached {stop_count} times. Stopping early."
                    )
                    break

        return mx.array(results).astype(mx.float32)

    def get_polydegrees_for_acc(self, goal_acc=1.0) -> mx.array:
        """
        This function creates datapoints for multiple subsets of the dataset.
        The datapoints can then be plotted.
        """
        results = []

        for dataset in tqdm(self.datasets, desc="Processing Correlation"):
            results.append(self._calc_polydegree_for_acc(dataset, goal_acc))

        return mx.array(results).astype(mx.float32)

    def _calc_polydegree_for_acc(self, patterns: mx.array, goal_acc=1.0):
        """
        Calculates n at which the model can store and retrieve all patterns given to it at a certain accuracy.
        """
        polydegree = 2.0
        prev = 0
        curr = 0
        while True:
            self.model.set_polydegree(polydegree)
            prev = curr
            curr = self._calc_restoration_acc(patterns)
            print(f"Accuracy at polydegree {polydegree}: {curr}")
            if curr >= goal_acc:
                break

            if abs(curr - prev) < 0.01 and prev != 0:
                polydegree += 3.0
            else:
                polydegree += 1.0

        return polydegree

    def _calc_k_max(self, patterns: mx.array, goal_acc=1.0, start_at=None):
        """
        Calculates the maximum number of patterns that can be stored and retrieved by the model at a certain accuracy.
        """
        if not start_at:
            start_at = 1

        low = start_at
        high = patterns.shape[0]
        best_k_max = low

        while low <= high:
            mid = (low + high) // 2
            memories = patterns[:mid]
            acc = self._calc_restoration_acc(memories)

            if acc >= goal_acc:
                best_k_max = mid
                low = mid + 1
            else:
                high = mid - 1

        return best_k_max

    def _calc_restoration_acc(self, patterns: mx.array):
        """
        Calculates average restoration accuracy for a given dataset and model.
        """
        # setup
        accuracies = 0
        num_neurons = self.model.get_num_neurons()
        self.model.learn(patterns)

        # exec
        for point in patterns:
            self.model.update(point)
            restored = self.model.get_state()
            accuracies += mx.inner(point, restored) / num_neurons

        # average accuracy over all patterns
        restoration_acc = accuracies / patterns.shape[0]
        return restoration_acc

    def _calc_alpha(self, patterns):
        """
        Calculates alpha for patterns in a model
        """
        K_max = self._calc_k_max(patterns)
        alpha = mx.divide(
            K_max,
            (mx.power(self.model.get_num_neurons(), self.model.get_poly_degree() - 1)),
        )
        return alpha
