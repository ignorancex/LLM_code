# preprocessing configs
in_res = (128, 128, 128)
out_res = (16, 16, 16)
win_min, win_max = -1000, 400
input_keys = ["airway", "artery", "vein", "lobe"]
resample_configs = {
    "image": {"size": in_res, "interp": "linear"},
    "airway": {"size": in_res, "interp": "nearest"},
    "artery": {"size": in_res, "interp": "nearest"},
    "vein": {"size": in_res, "interp": "nearest"},
    "lobe": {"size": in_res, "interp": "nearest"}
}

# model configs
enc_cfgs = {
    "in_channels": 9
}
dec_cfgs = {
    "num_channels": [963, 64],
    "num_classes": 19,
    "num_layers": 1,
    "drop_prob": 0.3
}

# training configs
w_ce = 0.5
w_dice = 1
batch_size = 8
num_workers = 4
max_lr = 1e-3
min_lr = 1e-6
epochs = 50

# evaluation configs
eval_freq = 5
eval_res = (128, 128, 128)
eval_batch_size = 4

# inference configs
weights_path = "/media/dntech/_mnt_storage/kaiming/data/lung_segment/logs/lbav/model_44.pth"
infer_batch_size = 1
window_size = (96, 96, 96)
