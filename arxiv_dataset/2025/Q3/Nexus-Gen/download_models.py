from modelscope import snapshot_download

snapshot_download('DiffSynth-Studio/Nexus-GenV2', local_dir='models/Nexus-GenV2')
flux_path = snapshot_download('black-forest-labs/FLUX.1-dev', 
    allow_file_pattern=[
    "text_encoder/model.safetensors",
    "text_encoder_2/*",
    "ae.safetensors",
    ],
    local_dir='models/FLUX/FLUX.1-dev')
