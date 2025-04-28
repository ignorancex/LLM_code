import os, sys
from datetime import datetime

from deepreg.train import train

version = str(sys.argv[1]) # Run used e.g. RigidReg_run1, RigidAffineReg_run1 or NoReg_run1 where run can be run1, run2 or run3 OR f0, f1, f2, f3, f4
res = str(sys.argv[2]) # Resolution of input images : 2.0, 1.5 or 1.0 (in mm)

log_root = f"/media/andjela/SeagatePor"
log_dir = log_root+f"/logs_train_{version[:-3]}_{res}/" + datetime.now().strftime("%Y%m%d-%H%M%S") #previously version[:-5]
config_path = [f"{log_root}/{version}_{res}.yaml"]


train(
    gpu="0",
    config_path=config_path,
    gpu_allow_growth=True,
    ckpt_path="",
    log_root=log_root,
    log_dir=log_dir,
)

# print(
#     "\n\n\n\n\n"
#     "=======================================================\n"
#     "The training can also be launched using the following command.\n"
#     "deepreg_train --gpu '0' "
#     f"--config_path demos/{name}/{name}.yaml "
#     f"--log_root demos/{name} "
#     "--log_dir logs_train\n"
#     "=======================================================\n"
#     "\n\n\n\n\n"
# )