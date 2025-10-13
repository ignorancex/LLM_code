import os

import mlx.core as mx
import numpy as np
import torchvision
import random

from root.correlations.correlations import Correlations


class Data:
    @staticmethod
    def load(relative_path) -> mx.array:
        array = []
        data_dir = os.path.dirname(__file__)
        abs_path = os.path.join(data_dir, "arrays/" + relative_path + ".npy")

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File at {relative_path} does not exist.")

        with open(abs_path, "rb") as f:
            array = mx.load(f)

        return mx.array(array).astype(mx.float32)

    @staticmethod
    def save(relative_path, data: mx.array) -> mx.array:
        script_dir = os.path.dirname(__file__)
        abs_path = os.path.join(script_dir, "arrays/" + relative_path + ".npy")

        with open(abs_path, "wb") as f:
            mx.save(f, data)

        print("Successfully saved data to", "arrays/" + relative_path)

        return True

    @staticmethod
    def generate_rademacher_patterns(num_patterns, num_cols) -> mx.array:
        np.random.seed(42)  # Set the random seed for reproducibility
        patterns = np.random.choice(
            [-1, 1], size=(num_patterns, num_cols)
        )  # Generate random binary data
        return mx.array(patterns).astype(mx.float32)

    @staticmethod
    def generate_correlated_patterns(num_patterns, num_cols, eps) -> mx.array:
        patterns = np.random.choice(
            [-1, 1],
            p=[0.5 * (1 - eps), 0.5 * (1 + eps)],
            size=(num_patterns, num_cols),
        )
        return mx.array(patterns).astype(mx.float32)

    def generate_real_world_patterns(num_patterns, num_cols) -> mx.array:
        # generates multi-variate binary patterns
        mx.set_default_device(mx.cpu)

        # Create a random positive-definite covariance matrix
        A = mx.random.normal([num_cols, num_cols])
        cov = mx.matmul(A, A.T)

        # Mean vector for the multivariate normal distribution
        mean = mx.zeros(num_cols)

        # Generate patterns from a multivariate normal distribution
        data = mx.random.multivariate_normal(mean, cov, shape=[num_patterns])

        # Convert continuous values into {-1, 1} by thresholding at zero
        # Values >= 0 become 1, values < 0 become -1
        patterns = mx.where(data >= 0, 1, -1)

        return mx.array(patterns).astype(mx.float32)

    @staticmethod
    def get_mnist_patterns(digit, num_patterns) -> mx.array:
        mnist_train = torchvision.datasets.MNIST("dataset/", train=True, download=True)
        labels = mnist_train.targets.numpy()
        data = mnist_train.data.numpy()

        specific_number_indices = np.where(labels == digit)[0]
        specific_number_data = data[specific_number_indices][:num_patterns]

        binarized_images = np.where(specific_number_data >= 128, 1, -1)
        binarized_images = binarized_images.reshape(num_patterns, -1)

        return mx.array(binarized_images).astype(mx.float32)

    @staticmethod
    def create_subset_of_mnist_with_goal_hd(num_patterns, goal_hd) -> mx.array:
        # make a dataset containing num_patterns of each defined digit
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        base = mx.concatenate([Data.get_mnist_patterns(i, 5000) for i in digits])

        # create a subset with a target mean Hamming distance
        subset = Data.build_subset_target_mean(
            base,
            target_hd=goal_hd,
            goal_num_patterns=num_patterns,
            max_rounds=20000,
            max_no_improve=1000,
            acceptance_margin=5.0,
        )
        subset = mx.array(subset).astype(mx.float32)

        # print the mean HD
        print("size of set", subset.shape)
        print("Mean HD:", Correlations.calc_average_hd(subset))

        return subset

    def build_subset_target_mean(
        base_dataset,
        target_hd,
        goal_num_patterns,
        batch_size=20,
        max_rounds=1000,
        max_no_improve=50,
        acceptance_margin=2.0,
    ):
        """
        Hill-climbing style approach to get a final subset of size 'goal_num_patterns'
        whose *average* Hamming distance is near 'target_hd'.

        Not enforcing min distance or uniform distance—only the final *mean* matters.

        Args:
        base_dataset (np.ndarray): shape (N, D) of binary patterns (0/1).
        target_hd (float): the desired mean Hamming distance.
        goal_num_patterns (int): how many patterns we want in the final subset.
        batch_size (int): number of random candidates we evaluate each round.
        max_rounds (int): max loop iterations.
        max_no_improve (int): stop after this many consecutive no-improvement rounds.
        acceptance_margin (float): how much "worse" (further from target) we allow
                                    in order to keep building the set.
        Returns:
        subset (list of np.ndarray)
        """

        N = base_dataset.shape[0]
        used_indices = set()

        # Start with an empty subset or pick one random to kick things off
        subset = []
        idx = np.random.randint(0, N)
        subset.append(base_dataset[idx])
        used_indices.add(idx)

        old_mean_hd = Correlations.calc_average_hd(subset)
        old_diff = abs(old_mean_hd - target_hd)

        no_improvement_rounds = 0

        for _ in range(max_rounds):
            if len(subset) >= goal_num_patterns:
                break

            # Gather a batch of random candidates
            valid_indices = [i for i in range(N) if i not in used_indices]
            if not valid_indices:
                # no more candidates to pick
                break

            batch_size_eff = min(batch_size, len(valid_indices))
            candidate_indices = random.sample(valid_indices, batch_size_eff)

            best_candidate_info = None
            best_improvement = float("-inf")

            for idx_cand in candidate_indices:
                idx_cand = int(idx_cand)
                candidate = base_dataset[idx_cand]
                # Evaluate the new mean if we add this candidate
                new_mean_hd = Correlations.calc_average_hd(subset + [candidate])
                new_diff = abs(new_mean_hd - target_hd)

                improvement = old_diff - new_diff  # positive => closer to target

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate_info = (idx_cand, candidate, new_mean_hd, new_diff)

            # If we found a best candidate, decide whether to accept it
            if best_candidate_info:
                idx_cand, candidate, new_mean_hd, new_diff = best_candidate_info

                # Accept if it improves or is only slightly worse by <= acceptance_margin
                if (best_improvement > 0) or (best_improvement >= -acceptance_margin):
                    subset.append(candidate)
                    used_indices.add(idx_cand)

                    old_mean_hd = new_mean_hd
                    old_diff = new_diff
                    no_improvement_rounds = 0
                else:
                    # Did not meet acceptance criteria
                    no_improvement_rounds += 1
            else:
                # No candidate in the batch at all (rare) => no improvement
                no_improvement_rounds += 1

            # If we haven't improved for many rounds, we give up
            if no_improvement_rounds >= max_no_improve:
                break

        return subset

    def add_pattern_if_improves(subset, candidate, target_hd):
        """
        Return True if adding 'candidate' gets the subset's mean HD *closer*
        to 'target_hd' than before, False otherwise.
        """
        old_hd = Correlations.calc_average_hd(subset)
        # test new subset
        new_subset = subset + [candidate]
        new_hd = Correlations.calc_average_hd(new_subset)

        old_diff = abs(old_hd - target_hd)
        new_diff = abs(new_hd - target_hd)

        return new_diff < old_diff

    def create_average_hd_subset(base_dataset, target_hd, goal_num_patterns):
        """
        Create a subset from the base dataset with a target mean Hamming distance.
        """

        def calculate_mean_hd(subset):
            n = len(subset)
            if n < 2:
                return 0
            total_hd = sum(
                np.sum(x != y) for i, x in enumerate(subset) for y in subset[i + 1 :]
            )
            return total_hd / (n * (n - 1) / 2)

        subset = []
        added_idxs = set()
        while len(subset) < goal_num_patterns:
            # Randomly sample a candidate
            idx = np.random.randint(0, base_dataset.shape[0])
            if idx in added_idxs:
                continue
            candidate = base_dataset[idx]

            # Check if the candidate aligns with the target HD
            subset.append(candidate)
            current_hd = calculate_mean_hd(subset)
            if abs(current_hd - target_hd) > 1e-3:
                subset.pop()  # Remove if it doesn't align
            else:
                added_idxs.add(idx)

        return np.array(subset)

    # old but might work still
    def create_datasets(subset_length, hd_range) -> mx.array:
        datasets = []

        for hd in hd_range:
            subset = []
            added_idxs = set()

            while len(subset) < subset_length:
                idx1 = mx.random.randint(0, self.base_set.shape[0])

                # goal hamming distance is the mean hamming distance I want.

                if idx1 not in added_idxs:
                    new_memory = self.base_set[idx1]
                    fitting_hd = True
                    subset_idx = 0
                    # if the hamming distance is too low, don't add the memory
                    while fitting_hd and subset_idx < len(subset):
                        curr_hd = HammingDistance.calculate(
                            new_memory, subset[subset_idx]
                        )
                        if curr_hd < hd:
                            fitting_hd = False
                        subset_idx += 1

                    if fitting_hd:
                        subset.append(new_memory)
                        added_idxs.add(idx1)

            datasets.append(subset)

        return mx.array(datasets).astype(mx.float32)
