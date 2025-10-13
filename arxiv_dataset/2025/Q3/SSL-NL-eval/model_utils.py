import numpy as np

depths_base = [round(d,2) for d in np.arange(0.083, 1, 0.083)]

model_depth_map = {
        'CNN': -0.08,
        'embeds': 0
    } | {f'T{i+1}': depths_base[i] for i in range(len(depths_base))}