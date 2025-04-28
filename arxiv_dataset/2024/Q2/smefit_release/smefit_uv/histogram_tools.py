# -*- coding: utf-8 -*-
# import arviz as az
import numpy as np


def find_xrange(samples, threshold_x):
    x_low = []
    x_high = []

    for sample in samples:
        n, bins = np.histogram(
            sample,
            bins="fd",
            density=True,
        )

        max_index = np.argmax(n)
        threshold = threshold_x * n[max_index]
        indices = np.where(n >= threshold)[0]

        low, up = np.percentile(sample, 2.5), np.percentile(
            97.5
        )  # az.hdi(sample, hdi_prob=.95)

        index_max = np.max(indices)
        index_min = np.min(indices)

        x_high.append(bins[index_max])
        x_low.append(bins[index_min])

        # if low < 0:
        #     x_low.append(bins[index_min])
        # else:
        #     x_low.append(- 0.1 * bins[index_max])

    low, up = min(x_low), max(x_high)

    return low - 0.05 * (up - low), up + 0.1 * (up - low)
