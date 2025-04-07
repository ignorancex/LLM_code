import os

# Multiple gpus
os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./mmsegmentation/tools/dist_train.sh gcnet-s_4xb3-120k_cityscapes-1024x1024.py 4 --work-dir ./weight/seg")

# Single gpu
# os.system("CUDA_VISIBLE_DEVICES=0 bash ./mmsegmentation/tools/dist_train.sh gcnet-s_4xb3-120k_cityscapes-1024x1024.py --work-dir ./weight/seg")
