import mlx.core as mx


def calc_alphas(x, k_maxes, N, n):
    alphas = []
    for x in range(len(x)):
        alpha = mx.divide(
            k_maxes[x],
            (mx.power(N, n - 1)),
        )
        alphas.append(float(alpha))
    return alphas
