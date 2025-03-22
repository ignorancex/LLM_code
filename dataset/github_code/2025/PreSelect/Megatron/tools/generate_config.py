import argparse
import yaml


parser = argparse.ArgumentParser("Generate yaml config for convert checkpoint")
parser.add_argument("--name",type=str)
parser.add_argument("--megatron_ckpt_path",type=str)
parser.add_argument("--hf_ckpt_path",type=str)
parser.add_argument("--seq_len",type=int)
parser.add_argument("--global_bz",type=int)
parser.add_argument("--model_size",type=str)


if __name__ == "__main__":
    args = parser.parse_args()

    YAML_TEMPLATE = f"""VOCAB_SIZE: 64005
TOKENIZER_DIR: neo/scripts/tokenizer.model
MEGATRON_CHECKPOINT_PATH: {args.megatron_ckpt_path}
HF_CHECKPOINT_PATH: {args.hf_ckpt_path}
SEQ_LEN: {args.seq_len}
GB: {args.global_bz}
BF16: true

CKPTS_TO_KEEP_BY_BILLIONS_OF_TOKENS:
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 9
    """

    YAML_TEMPLATE_1B = f"""VOCAB_SIZE: 64005
TOKENIZER_DIR: neo/scripts/tokenizer.model
MEGATRON_CHECKPOINT_PATH: {args.megatron_ckpt_path}
HF_CHECKPOINT_PATH: {args.hf_ckpt_path}
SEQ_LEN: {args.seq_len}
GB: {args.global_bz}
BF16: true

CKPTS_TO_KEEP_BY_BILLIONS_OF_TOKENS:
  - 3
  - 6
  - 9
  - 12
  - 15
  - 18
  - 21
  - 24
  - 27
  - 30
  - 33
  - 36
  - 39
  - 42
  - 45
  - 48
  - 51
  - 54
  - 57
  - 60
  - 63
  - 66
  - 69
  - 72
  - 75
  - 78
  - 81
  - 84
  - 87 
  - 90
  - 93
  - 96
  - 99
  - 102
  - 105
  - 108
  - 111
  - 114
  - 117
  - 120
  - 123
  - 126
  - 129
  - 132
  - 135
  - 138
  - 141
  - 144
  - 147
  - 150
  - 153
    """

    YAML_TEMPLATE_3B = f"""VOCAB_SIZE: 64005
TOKENIZER_DIR: neo/scripts/tokenizer.model
MEGATRON_CHECKPOINT_PATH: {args.megatron_ckpt_path}
HF_CHECKPOINT_PATH: {args.hf_ckpt_path}
SEQ_LEN: {args.seq_len}
GB: {args.global_bz}
BF16: true

CKPTS_TO_KEEP_BY_BILLIONS_OF_TOKENS:
    - 5
    - 10
    - 15
    - 20
    - 25
    - 30
    - 35
    - 40
    - 45
    - 50
    - 55
    - 60
    - 65
    - 70
    - 75
    - 80
    - 85
    - 90
    - 95
    - 100
    - 105
    """
    if "1B" in args.model_size:
        with open(f"./neo/configs/{args.model_size}-{args.name}_convert_config.yaml", "w") as f:
            f.write(YAML_TEMPLATE_1B)
    elif "3B" in args.model_size:
        with open(f"./neo/configs/{args.model_size}-{args.name}_convert_config.yaml", "w") as f:
            f.write(YAML_TEMPLATE_3B)
    else:
        with open(f"./neo/configs/{args.model_size}-{args.name}_convert_config.yaml", "w") as f:
            f.write(YAML_TEMPLATE)     
    print(f"Template YAML config file generated as {args.model_size}-{args.name}_convert_config.yaml")