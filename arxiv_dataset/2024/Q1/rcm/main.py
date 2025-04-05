




import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("scp", type=str)
parser.add_argument("--device", type=int, default=0)
opts = parser.parse_args()


seeds = [0, 1, 2, 3, 4]
strides = ["1222", "1122"]


command_type = "CUDA_VISIBLE_DEVICES={device} python {scp} resnet18 cifar10 --strides={stride} --seed={seed} -m={seed}"


for stride in strides:
    for seed in seeds:
        command = command_type.format(
            device=opts.device, scp=opts.scp,
            stride=stride, seed=seed
        )
        print(f"====={command}======")
        os.system(command)

