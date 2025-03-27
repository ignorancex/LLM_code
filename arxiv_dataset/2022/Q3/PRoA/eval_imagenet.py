import timm
import os 
import robustness
from robustness.attacker import AttackerModel
from verification import verify
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
import time
ds = robustness.datasets.DATASETS['imagenet']('/home/tianle/datasets/ImageNet2012')
_, test_loader = ds.make_loaders(workers=8,
                                    batch_size=1,
                                    test_subset = True,
                                    test_subset_size = 100)

# for model_name in ['vit_base_patch16_224']:
for model_name in ['resnetv2_50', 'vit_base_patch16_224', 'mobilenetv2_100', 'efficientnet_b0']:

    print(model_name)

    m = timm.create_model(model_name, pretrained=True)
    model = AttackerModel(m, dataset=ds)
    model = model.cuda()
    model.eval()
    n_test = len(test_loader)

    result_ac = []
    time_ac = []

    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="adaptive", tau=0.05, delta=1e-10, sample_limit=50000, bs=1000, kwargs={
            'rot': 30,
            'trans': 0, 
            'scale': 0,
            'hue': 0,
            'satu': 0,
            'bright': 0,
            'cont': 0,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'spatial',
            'attack_type': 'random',
            'do_tqdm': False
            })

        end = time.time()
        time_ac.append(end-start)
        total_num += num

        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
            
    print(model_name+"_PRoA:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
