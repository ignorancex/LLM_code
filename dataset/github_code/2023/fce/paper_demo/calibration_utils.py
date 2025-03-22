import numpy as np
import skfuzzy

def expected_calibration_error(y_true, y_pred, num_bins):
    ece_vals = []
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))
        if sum(correct[mask]) > 0:
            ece_vals.append(round(o / sum(correct[mask]), 3))
        else:
            ece_vals.append(0)

    return ece_vals, round(o / y_pred.shape[0], 3)


def expected_calibration_error_plot(y_true, y_pred, num_bins):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    x = []
    y = []
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            x.append(np.sum(prob_y[mask]) / y_pred.shape[0])
            y.append(np.sum(correct[mask]) / y_pred.shape[0])

    return x, y


def intervals(parts, duration):
    part_duration = duration / parts
    return [i * part_duration for i in range(parts)]


def fuzzy_binning(x, bins):
    ticks = intervals(bins, 1)
    ticks.append(1.0)

    x = np.array([x])
    y = {}

    for i in range(len(ticks) - 1):
        t0, t1 = ticks[i], ticks[i + 1]

        mid = (t0 + t1) / 2
        b = (t0 + mid) / 2
        c = (mid + t1) / 2
        a = (2 * t0) - b
        d = (2 * t1) - c

        y[i] = round(float(skfuzzy.trapmf(x, np.array([a, b, c, d]))), 3)

    return y


def fuzzy_conf(g, correct, prob_y):
    acc_sum = []
    conf_sum = []
    for i in range(len(correct)):
        if correct[i] == 1:
            acc_sum.append(g[i])
            conf_sum.append(g[i] * prob_y[i])

    return (sum(acc_sum), sum(conf_sum))


def fuzzy_calibration_error(y_true, y_pred, n_bins):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    mem = []
    for p in prob_y:
        mem.append(fuzzy_binning(p, bins=n_bins))

    bins = n_bins

    g_bin = {}
    total_mem_g_bin = {}
    acc_sum_g_bin = {}
    conf_sum_g_bin = {}
    acc_g_bin = {}
    conf_g_bin = {}

    fce_num = 0
    fce_den = 0

    fce_vals = []

    for bin_ in range(bins):
        g_bin[bin_] = [x[bin_] for x in mem]
        total_mem_g_bin[bin_] = sum(g_bin[bin_])
        acc_sum_g_bin[bin_], conf_sum_g_bin[bin_] = fuzzy_conf(g_bin[bin_], correct, prob_y)

        if total_mem_g_bin[bin_] != 0:
            acc_g_bin[bin_] = acc_sum_g_bin[bin_] / total_mem_g_bin[bin_]
            conf_g_bin[bin_] = conf_sum_g_bin[bin_] / total_mem_g_bin[bin_]
        else:
            acc_g_bin[bin_] = 0
            conf_g_bin[bin_] = 0

        fce_vals.append(abs(acc_g_bin[bin_] - conf_g_bin[bin_]))

        fce_num += total_mem_g_bin[bin_] * abs(acc_g_bin[bin_] - conf_g_bin[bin_])
        fce_den += total_mem_g_bin[bin_]

    fce = round(float(fce_num / fce_den), 3)

    return fce_vals, fce


