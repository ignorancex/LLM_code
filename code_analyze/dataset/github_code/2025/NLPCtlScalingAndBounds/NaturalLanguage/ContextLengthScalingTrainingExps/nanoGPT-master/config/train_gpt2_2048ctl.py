# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'NLP_contextScaling_testFull_2048ctl_fp32'
wandb_run_name='gpt2-124M_01'

# these make the total batch size be ~0.5M
# 6 batch size * 2048 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 6
block_size = 2048
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

train_file_name = 'train_P99.99_SFrom0.0.bin'