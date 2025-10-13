import math
from scipy.stats import binom

def binomial_tail_probability(n_bits, bit_accuracy):
    """
    Computes the sum from k >= n_bits * bit_accuracy to n_bits of:
    (n_bits choose k) * (1/2)^n_bits
    """
    threshold_k = math.ceil(n_bits * bit_accuracy)
    tail_prob = sum(
        binom.pmf(k, n_bits, 0.5)
        for k in range(threshold_k, n_bits + 1)
    )

    return tail_prob

# Example usage:
n_bits = 16*48
bit_accuracy = 0.950 # for example
result = binomial_tail_probability(n_bits, bit_accuracy)
print(f"Tail probability: {result:.6f}")
print(math.log10(result))
