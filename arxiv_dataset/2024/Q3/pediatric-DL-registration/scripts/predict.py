import sys
from datetime import datetime

from deepreg.predict import predict
from timeit import default_timer as timer
from datetime import timedelta

date = str(sys.argv[1]) # Date of training is e.g. (for example) 20210406-111300
ckpt = str(sys.argv[2]) # Integer for epoch number of a certain checkpoint e.g. 25,50, 75, 100, 125 or 150 if nbr_epochs=150 and saving_interval=25
version = str(sys.argv[3]) # Version used e.g. RigidReg, RigidAffineReg or NoReg followed by f0, f1 to f4
# run = str(sys.argv[4]) # Run used e.g. run1, run2 or run3
res = str(sys.argv[4]) # Resolution of input images : 2.0, 1.5 or 1.0 (in mm)
# mode = str(sys.argv[6]) # Mode to be used: predictions on validation or test set, e.g. valid or test

log_root = f"/media/andjela/SeagatePor"
log_dir = f"logs_predict_{version[:-3]}_{res}/" + datetime.now().strftime("%Y%m%d-%H%M%S")+f'{version[-3:]}'
ckpt_path = f"{log_root}/logs_train_{version[:-3]}_{res}/{date}/save/ckpt-{ckpt}" #To change depending on where checkpoints are saved
config_path = [f"{log_root}/{version}_{res}.yaml"] 

#Use mode=valid to predict if network works well, then last step->calculate perfomance on mode=test set
start = timer()
predict(
    gpu="0",
    gpu_allow_growth=True,
    ckpt_path=ckpt_path,
    mode="test", #f"{mode}"
    sample_label="all",
    batch_size=1,
    log_root=log_root,
    log_dir=log_dir,
    save_nifti=True,
    save_png=True,
    config_path=config_path,
)
end = timer()
print('Time Taken:', timedelta(seconds=end-start))
# log_root = f"demos/{name}"
# log_dir = "logs_predict/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# ckpt_path = f"{log_root}/dataset/pretrained/ckpt-4000"
# config_path = [f"{log_root}/{name}.yaml"]
# if args.test:
#     config_path.append("config/test/demo_unpaired_grouped.yaml")

# predict(
#     gpu="0",
#     gpu_allow_growth=True,
#     ckpt_path=ckpt_path,
#     mode="test",
#     batch_size=1,
#     log_root=log_root,
#     log_dir=log_dir,
#     sample_label="all",
#     config_path=config_path,
# )

# print(
#     "\n\n\n\n\n"
#     "=========================================================\n"
#     "The prediction can also be launched using the following command.\n"
#     "deepreg_predict --gpu '' "
#     f"--config_path demos/{name}/{name}.yaml "
#     f"--ckpt_path demos/{name}/dataset/pretrained/ckpt-4000 "
#     f"--log_root demos/{name} "
#     "--log_dir logs_predict "
#     "--save_png --mode test\n"
#     "=========================================================\n"
#     "\n\n\n\n\n"
# )