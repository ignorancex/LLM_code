from glob import glob
import os
import json

ROOT = './mmdetection/work_dirs'
list_models = os.listdir(ROOT)
with open('best_epoch.txt', 'w') as f:
    for model in list_models:
        json_path = list(sorted(glob(ROOT + '/' + model + '/*.log')))[-1]
        best_val = 0.0
        best_epoch = -1
        with open(json_path, 'r') as json_file:
            for line in json_file:
                # print(line)
                if 'Epoch(val)' in line:
                    epoch = int(line.split('Epoch(val) [')
                                [1].split('][1657]')[0])
                    mAP = float(line.split('segm_mAP: ')[
                        1].split(', segm_mAP_50')[0])
                    if mAP > best_val:
                        best_val = mAP
                        best_epoch = epoch
        json_file.close()

        # break
        s = ' '.join([model, str(best_val), str(best_epoch), '\n'])
        f.write(s)
f.close()
