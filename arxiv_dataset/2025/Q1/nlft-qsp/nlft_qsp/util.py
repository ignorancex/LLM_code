# Utility functions

def next_power_of_two(n):
    """Returns the smallest power of two that is `>= n`."""
    return 1 << (n - 1).bit_length()

def sequence_shift(c, s):
    """Shifts the coefficients in the given list by s to the right,
    so that the returned vector `r` satisfies `r[k + s] = c[k]`."""
    s %= len(c)
    return c[-s:] + c[:-s]