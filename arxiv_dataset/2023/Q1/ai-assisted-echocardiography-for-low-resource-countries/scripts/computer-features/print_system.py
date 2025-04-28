# flake8: noqa
### USAGE:
#   cd $HOME/repositories/echocardiography/scripts/computer-features
#   conda activate rt-ai-echo-VE
#   python print_system.py

import platform
import sys

import cv2
import numpy
import torch

print(f'Python:    {sys.version}')
print(f'Platform:  {platform.platform()}')
print(f'NumPy:     {numpy.__version__}')
print(f'opencv:    {cv2.__version__}')
print(f'PyTorch:   {torch.__version__}')
print(f'cuda_is_available: {torch.cuda.is_available()}')
print(f'cuda version: {torch.version.cuda}')
print(f'cuda.device_count  {torch.cuda.device_count()}')
