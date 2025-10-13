import random

def unique_seed_generator(base_seed: int):
    """
    A generator function that yields unique random seeds based on a base seed.

    Parameters
    ----------
    base_seed : int
        The initial seed to ensure consistency.

    Yields
    ------
    int
        A unique random seed.
    """
    rng = random.Random(base_seed)
    seeds = set()

    while True:
        seed = rng.randint(0, 2**32 - 1)
        if seed not in seeds:
            seeds.add(seed)
            yield seed

def generate_error_distribution(total_error_rate, num_segments, seed = None):
    """Generate random segments of a total error rate, each segment being a part of the total.

    Args:
        total_error_rate (float): The total error rate to be distributed.
        num_segments (int): The number of random segments to generate.

    Yields:
        float: A part of the total error rate, each call yields a segment.
    """
    if seed is not None:
        random.seed(seed)
    
    random_parts = [random.random() for _ in range(num_segments)]
    sum_of_parts = sum(random_parts)
    
    for part in random_parts:
        yield part / sum_of_parts * total_error_rate
