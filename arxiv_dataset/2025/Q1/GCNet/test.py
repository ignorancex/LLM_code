import os

# Single gpu
os.system("CUDA_VISIBLE_DEVICES=0 python ./mmsegmentation/tools/test.py ./gcnet-s_4xb3-120k_cityscapes-1024x1024.py ./weight/seg/gcnet-s_weight.pth")