# Dataset settings
dataset = dict(
    type="VariableFeatureDataset",
    data_path="./data/lrs3_debug/test.tsv",
    root_path="./data/lrs3_debug",
    video_feat_name = "avhubert_video_feat",
    audio_max = 1.756106,
    audio_min = -11.512925,
    audio_stack = 2,
)

num_workers = 8

batch_size = 1

# Acceleration settings
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="V2SFlowDecoder-S/2",
    enable_flash_attn=False, # True,
    enable_layernorm_kernel=False, # True,
    content_vocab_size=1000,
    pitch_vocab_size=20,
    speaker_embed_dim=256,
)
scheduler = dict(
    type="rflow_audio_x0",
    sample_method="logit-normal",
    num_sampling_steps=30,
    cfg_scale=2.0,
)

# Log settings
seed = 42
verbose = 1

save_dir = "./save/inference"
# load = "./checkpoints/v2sflow_decoder.pt"
load = "new_model.pt"

encoder = "./checkpoints/v2sflow_encoder.pt"
encoder_cfg = dict(
    type="V2SFlowEncoder-S",
    content_vocab_size=1000,
    pitch_vocab_size=20,
    speaker_embed_dim=256,
)
vocoder = "./checkpoints/hifigan_vocoder.pt"
vocoder_cfg = "./checkpoints/hifigan_vocoder_config.json"
