"""
data generators for exchangeability
"""


import numpy as np


def generate_binary_iid(
        p: float,
        size: int,
        rng: np.random.Generator = None,
):
    """Generate an i.i.d. Bernoulli(p) sequence."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    return rng.binomial(1, p=p, size=size)


def generate_binary_exch_noniid(
        p: float,
        size: int,
        rng: np.random.Generator = None,
):
    """Generate an exchangeable but non-i.i.d. sequence."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    b0 = rng.binomial(1, p=p)
    b = rng.binomial(1, p=p, size=size)
    return np.minimum(b0, b)  # b0 AND bt


def generate_binary_changepoint(
        p: float,
        q: float,
        size: int,
        change_loc: float = 0.5,
        change_len: float = None,
        rng: np.random.Generator = None,
):
    """Generate a Bernoulli sequence with a changepoint in the middle.

    The sequence is Ber(p) for up until time `size*change_loc`, and then 
    it switches to Ber(q) for `change_len`. 
    If `change_len` is None, it does not switch back to Ber(p).
    
    Both `change_loc` and `change_len` are in [0, 1] representing fractions of `size`.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    change_start = int(size * change_loc)
    if change_len is not None:
        change_end = change_start + int(size * change_len)
    else:
        change_end = size
    if change_end >= size:
        return np.concatenate([
            rng.binomial(1, p=p, size=change_start),
            rng.binomial(1, p=q, size=size - change_start),
        ])
    else:
        return np.concatenate([
            rng.binomial(1, p=p, size=change_start),
            rng.binomial(1, p=q, size=change_end - change_start),
            rng.binomial(1, p=p, size=size - change_end),
        ])


def generate_binary_change_twice(
        p: float,
        q: float,
        size: int,
        rng: np.random.Generator = None,
):
    """Generate a Bernoulli sequence with two quick periods of change,
    that is, Ber(p) [0-0.4] -> Ber(q) [0.4-0.5] -> Ber(p) [0.5-0.9] -> Ber(q) [0.9-1.0]."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    assert size >= 10

    s1, s2, s3 = [int(frac * size) for frac in [0.4, 0.1, 0.4]]
    s4 = size - (s1 + s2 + s3)
    return np.concatenate([
        rng.binomial(1, p=p, size=s1),
        rng.binomial(1, p=q, size=s2),
        rng.binomial(1, p=p, size=s3),
        rng.binomial(1, p=q, size=s4),
    ])


def generate_binary_change_repeated(
        p: float,
        q: float,
        size: int,
        change_every: int,
        rng: np.random.Generator = None,
):
    """Generate a Bernoulli sequence with repeated periods of change between two means.
    p -> q -> p -> q -> ..."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    assert size > change_every

    seq = rng.binomial(1, p=p, size=change_every)
    n = change_every
    use_p = False
    while n < size:
        seq = np.concatenate([
            seq,
            rng.binomial(1, p=(p if use_p else q), size=change_every)
        ])
        n += change_every
        use_p = not use_p
    
    return seq[:size]


def generate_binary_markov(
        p_10: float,
        p_11: float,
        size: int,
        p_init: float = 0.5,
        rng: np.random.Generator = None,
):
    """Generate a binary sequence from a first-order Markov process.

    `p_kj` is the transition probability from state `j` to state `k`.
    The resulting sequence is NOT exchangeable."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    probs = np.array([p_10, p_11])

    seq = np.zeros(size, dtype=int)
    seq[0] = rng.binomial(1, p=p_init)
    for t in range(1, size):
        seq[t] = rng.binomial(1, p=probs[seq[t - 1]])
    return seq


def generate_ar(order: int, size: int, rng: np.random.Generator = None):
    """Generate a sequence from an AR(order) process.

    This is an example for a _continuous_ process.
    The resulting sequence is NOT exchangeable."""
    raise NotImplementedError


def generate_urn(n, m, rng: np.random.Generator = None):
    """Generate a sequence of marbles from an urn.

    This is an example for a _non-binary but discrete_ process.
    The resulting sequence is exchangeable (but not i.i.d.).
    """
    raise NotImplementedError
