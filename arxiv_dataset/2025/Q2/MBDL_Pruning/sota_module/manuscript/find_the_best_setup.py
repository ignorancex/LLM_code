import os
import glob
import json
from collections import defaultdict
import pandas as pd
import tqdm


file_paths = glob.glob('/opt/experiment/20230212_deq_cal*/*/*/result.json')

ret = defaultdict(list)

for file in tqdm.tqdm(file_paths):
    with open(file, 'r') as f:
        tmp = json.load(f)

        if tmp['config']['dataset']['pmri_modl']['acceleration_rate'] == 4:

            ret['trial_id'].append(tmp['trial_id'])
            ret['tst_psnr'].append(tmp['tst_psnr'])
            ret['config'].append(tmp['config'])

ret = pd.DataFrame(ret)
ret = ret[ret['tst_psnr'] == ret['tst_psnr'].max()]

print(ret.to_dict())

# print(pd.DataFrame(ret)['tst_psnr'].max())