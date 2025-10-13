from root.data.data import Data
import mlx.core as mx


def calculate_mnist_average_hd(samples_per_digit=3000):
    # Get a smaller sample of digits (0-9)
    digits = list(range(10))
    mnist_data = mx.concatenate(
        [Data.get_mnist_patterns(i, samples_per_digit) for i in digits]
    )

    # Calculate and print average Hamming distance
    from root.correlations.correlations import Correlations

    avg_hd = Correlations.calc_average_hd(mnist_data)

    # Add debug information
    print(f"Dataset shape: {mnist_data.shape}")
    print(f"Data type: {mnist_data.dtype}")
    print(f"Data range: min={mx.min(mnist_data)}, max={mx.max(mnist_data)}")
    print(f"Average Hamming distance across MNIST: {avg_hd}")

    # Verify calculation with a small subset
    if mnist_data.shape[0] > 2:
        sample_hd = Correlations.calc_average_hd(mnist_data[:2])
        print(f"Sample HD between first two patterns: {sample_hd}")

    return avg_hd


if __name__ == "__main__":
    calculate_mnist_average_hd()
