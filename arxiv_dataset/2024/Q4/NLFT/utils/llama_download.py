import torch
from modelscope import snapshot_download, AutoModel
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B', revision='master', cache_dir='/root/autodl-tmp/cache_dir')
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', revision='master', cache_dir='/root/autodl-tmp/cache_dir')
