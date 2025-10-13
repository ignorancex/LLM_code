# Dataset settings
dataset = dict(
    type="VariableFeatureDataset",
    data_path="./data/lrs3_debug/train.tsv",
    root_path="./data/lrs3_debug",
    max_sample_size = 600,
    min_sample_size = 5,
    audio_feat_name = "mel_tacotron",
    audio_max = 1.756106,
    audio_min = -11.512925,
    audio_stack = 2,
    content_name = "mhubert_unit",
    speaker_name = "speaker_embedding",
    pitch_name = "f0_code_dddmvc",
    shuffle=True,
)

# grad_checkpoint = True
num_workers = 8

max_tokens = 600 * 16
batch_size = 64

# Acceleration settings
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="V2SFlowDecoder-S/2",
    enable_flash_attn=False, #True,
    enable_layernorm_kernel=False, #True,
    content_vocab_size=1000,
    pitch_vocab_size=20,
    speaker_embed_dim=256,
)
scheduler = dict(
    type="rflow_audio_x0",
    sample_method="logit-normal",
    loss_type="l1",
)

# Log settings
seed = 42
outputs = "./save/train"
wandb = False
epochs = 200
log_every = 100
ckpt_every = 50000

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15

warmup_steps = 10000
