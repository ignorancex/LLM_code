# preprocessing configs
in_res = (128, 128, 128)
out_res = in_res
win_min, win_max = -1000, 400
input_keys = ["grids"]
resample_cfgs = {
    "image": {"size": out_res, "interp": "linear"},
    "lungsegment": {"size": out_res, "interp": "nearest"},
}

# model configs
enc_cfgs = {
    "in_channels": 3
}
dec_cfgs = {
    "num_channels": [512, 256, 128, 64],
    "num_layers": 1,
    "num_classes": 19
}

# training configs
batch_size = 2
num_workers = 4
max_lr = 1e-3
min_lr = 1e-6
epochs = 50
w_ce = 0.5
w_dice = 1

# evaluation configs
eval_freq = 5
eval_res = (256, 256, 256)
eval_batch_size = 1
window_size = (128, 128, 128)
weights_path = "/media/dntech/_mnt_storage/kaiming/data/lung_segment/logs/coord/model_49.pth"
