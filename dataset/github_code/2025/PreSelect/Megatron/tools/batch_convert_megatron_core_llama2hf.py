import subprocess
import os
import time
import argparse
import yaml
from tqdm import tqdm


def convert_megatron2hf(megatron_dir, hf_dir, hf_tokenizer_dir, use_bf16_converter, dry_run=False):
    if use_bf16_converter:
        saver = "llama2_hf_bf"
        print("Using BF16 converter")
    else:
        saver = "llama2_hf"
    cmd = f"""
    mkdir -p {hf_dir}
    python tools/checkpoint/util.py \
    --model-type GPT \
    --loader megatron_core \
    --saver {saver} \
    --load-dir {megatron_dir} \
    --save-dir {hf_dir} \
    --hf-tokenizer-path {hf_tokenizer_dir} \
    --megatron-path /workspace/datapool/data1/storage/xiwen/kashun/Pretrain-Data-Selection/Megatron-LM-NEO/megatron
    """
    print(cmd)
    start_time = time.time()

    if dry_run:
        success = True
        duration = 0
        return success, duration
    
    try:
        subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
        success = True
    except subprocess.CalledProcessError:
        success = False
        raise Exception(f"Error converting {megatron_dir} to {hf_dir}")
    
    end_time = time.time()
    duration = end_time - start_time

    return success, duration


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MEGATRON checkpoint to Huggingface checkpoint")
    parser.add_argument("--megatron-dir", type=str, default=None, help="Path to MEGATRON checkpoint directory")
    parser.add_argument("--hf-dir", type=str, default=None, help="Path to Huggingface checkpoint directory")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size")
    parser.add_argument("--config", type=str, default=None, help=f"Path to YAML config file, use --generate-config to generate a template")
    parser.add_argument("--generate-config", action="store_true", help="Generate a template YAML config file")
    parser.add_argument("--rename-hf-by-billions", action="store_true", help="Rename Huggingface checkpoint directories by billions of tokens")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing Huggingface checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (Do not execute commands)")
    return parser.parse_args()

YAML_TEMPLATE = """VOCAB_SIZE: 64000
TOKENIZER_DIR: /workspace/megatron/neo/tokenizer.model
MEGATRON_CHECKPOINT_PATH: /workspace/checkpoints
HF_CHECKPOINT_PATH: /workspace/hf_ckpt
SEQ_LEN: 8192
GB: 256
CKPTS_TO_KEEP_BY_BILLIONS_OF_TOKENS:
  - 10
  - 20
  - 30
  - 40
  - 50
  - 60
  - 70
  - 80
  - 90
  - 100
    """

def generate_config():
    global YAML_TEMPLATE
    with open("convert_megatron2hf.template.yaml", "w") as f:
        f.write(YAML_TEMPLATE)
    print("Template YAML config file generated as convert_megatron2hf.template.yaml")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def find_closest_checkpoint(dir_list_by_iter, target_tokens, seq_len=4096, gb=256):
    """
    Find the closest checkpoint to the target number of tokens
    dir_list_by_iter: list of dirs, looke like ["/workspace/checkpoints/NEO_7b_nl_tp1_pp1_mb1_gb256_gas1/exp1.1/checkpoint/iter_0002000", "/workspace/checkpoints/NEO_7b_nl_tp1_pp1_mb1_gb256_gas1/exp1.1/checkpoint/iter_0004000", ...]
    target_tokens: target number of tokens
    """
    tokens_per_iter = seq_len * gb
    closest = None
    closest_tokens = None
    min_diff = float("inf")
    for d in dir_list_by_iter:
        assert "iter_" in d, f"""Invalid directory name: {d}, expected ["/workspace/checkpoints/NEO_7b_nl_tp1_pp1_mb1_gb256_gas1/exp1.1/checkpoint/iter_0002000", "/workspace/checkpoints/NEO_7b_nl_tp1_pp1_mb1_gb256_gas1/exp1.1/checkpoint/iter_0004000", ...]"""
        iter_str = d.split("_")[-1]
        if iter_str.endswith("/"):
            iter_str = iter_str[:-1]
        try:
            iter_num = int(iter_str)
        except ValueError:
            raise ValueError(f"Invalid iter_str: {iter_str}")
        tokens = iter_num * tokens_per_iter
        diff = abs(tokens - target_tokens)
        if diff < min_diff:
            min_diff = diff
            closest = d
            closest_tokens = tokens
            closest_tokens_by_billions_str = f"{closest_tokens/1e9:.2f}B"
    
    print(f"Closest checkpoint to {target_tokens/1e9:.2f}B tokens: {closest}, corresponding tokens: {closest_tokens_by_billions_str}, diff: {min_diff/1e9:.2f}B")
    return closest, closest_tokens_by_billions_str


def batch_convert_megatron2hf(cfg, args):
    MEGATRON_CHECKPOINT_PATH = cfg["MEGATRON_CHECKPOINT_PATH"]
    HF_CHECKPOINT_PATH = cfg["HF_CHECKPOINT_PATH"]
    # VOCAB_SIZE = cfg["VOCAB_SIZE"]
    HF_TOKENIZER_DIR = cfg["TOKENIZER_DIR"]
    CKPTS_TO_KEEP_BY_BILLIONS_OF_TOKENS = cfg["CKPTS_TO_KEEP_BY_BILLIONS_OF_TOKENS"]
    dir_list_by_iter = [os.path.join(MEGATRON_CHECKPOINT_PATH, d) for d in os.listdir(MEGATRON_CHECKPOINT_PATH) if os.path.isdir(os.path.join(MEGATRON_CHECKPOINT_PATH, d))]
    BF16 = cfg.get("BF16", False)

    for ckp in tqdm(CKPTS_TO_KEEP_BY_BILLIONS_OF_TOKENS):
        megatron_dir, tokens_str = find_closest_checkpoint(dir_list_by_iter, ckp*1e9, cfg["SEQ_LEN"], cfg["GB"])
        if args.rename_hf_by_billions:
            hf_dir = os.path.join(HF_CHECKPOINT_PATH, tokens_str)
        else:
            # the last folder, starts with "iter_"
            if megatron_dir.endswith("/"):
                megatron_dir = megatron_dir[:-1]
            iter_subdir = os.path.basename(megatron_dir)
            assert "iter_" in iter_subdir, f"Invalid iter_subdir: {iter_subdir}"
            hf_dir = os.path.join(HF_CHECKPOINT_PATH, iter_subdir)
        if args.skip_existing and os.path.exists(hf_dir):
            print(f"Skip existing enabled. Skipping existing Huggingface checkpoint: {hf_dir}")
            continue
        success, duration = convert_megatron2hf(megatron_dir, hf_dir, HF_TOKENIZER_DIR, BF16, args.dry_run)
        print(f"Conversion successful: {success}, duration: {duration:.2f} seconds")


def main():
    args = parse_args()
    # generate template config file
    if args.generate_config:
        generate_config()
        return
    
    # convert using config file
    if args.config is not None:
        assert args.megatron_dir is None and args.hf_dir is None and args.vocab_size is None, "Please provide either --config or --megatron-dir, --hf-dir, and --vocab-size"

        cfg = load_config(args.config)
        batch_convert_megatron2hf(cfg, args)

    else:
        # convert single checkpoint
        assert args.megatron_dir is not None and args.hf_dir is not None and args.vocab_size is not None, "Please provide either --config or --megatron-dir, --hf-dir, and --vocab-size"

        # success, duration = convert_megatron2hf(args.megatron_dir, args.hf_dir, args.vocab_size)
        # print(f"Conversion successful: {success}, duration: {duration:.2f} seconds")
        raise NotImplementedError("Please provide a YAML config file using --config")

if __name__ == "__main__":
    main()