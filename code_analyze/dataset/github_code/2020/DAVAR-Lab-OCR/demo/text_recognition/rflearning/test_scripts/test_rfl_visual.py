"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_rfl_visual.py
# Abstract       :    RF-Learning visual counting Model evaluation config

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""

import os

_base_ = [
    '../../__base__/test_base_setting.py'
]

ckpts = list()

# model name setting
out_name = 'davar_test_rfl_visual'


# model parameter dictionary
tmp_dict = dict()

# experiment Name
tmp_dict['Name'] = 'rfl_visual'


# ===================== model .pth file path ========================
tmp_dict['ModelPath'] = '/data1/workdir/davar_opensource/rflearning_visual/RFL_visual_pretrained-2654bc6b.pth'
out_name += '/' + tmp_dict['ModelPath'].split('/')[-2].split('.')[0]

# ===================== model config file path ========================
tmp_dict['ConfigPath'] = '/data1/open-source/demo/text_recognition/rflearning/configs/rfl_res32_visual.py'

# ===================== model test mode ========================
tmp_dict['Epochs'] = None
ckpts.append(tmp_dict)

# save result of the test experiment
out_path = os.path.join('/data1/output_dir/sota_exp', out_name + '/')

force_test = False
force_eval = False

do_test = 1  # 1 for test
do_eval = 1

test_path = out_path + 'res/'
eval_path = out_path + 'eval/'
