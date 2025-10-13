import numpy as np


def lambert(x):
    x = np.where(x < 0, np.nan, x)  # 将小于0的值替换为nan
    y = x.copy()
    x = np.log(x)
    while True:
        z = np.log(y) + y
        tmp = z - x
        mask = np.abs(tmp) >= 0.00001
        y = np.where(tmp < 0, y * 1.02, y * 0.98)
        y = np.maximum(y, 1e0)  # 将y的值截断为一个很小的正数，避免出现负无穷大的情况
        if not mask.any():
            break
    y = np.round(y, 4)
    return y