import os
import argparse
from setproctitle import setproctitle

from utils import model_arg_dict, chat_template_dict

setproctitle("mmmm")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='qwen2.5-32B')
parser.add_argument('--framework', default='vllm', choices=['vllm', 'sglang'])
args = parser.parse_args()

model = model_arg_dict[args.model_name]

if __name__ == '__main__':
    if args.framework == 'vllm':
        chat_template = chat_template_dict[args.model_name]

        os.system(f"""
        CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve {model} \
        --port 8123 \
        --api-key needlechain \
        --dtype auto \
        --max_model_len 32768 \
        --tensor-parallel-size 4 \
        --max_num_seqs 1 \
        --chat-template {chat_template}""")
    #     --max_model_len 32768 \  # gemma에서는 없이
    elif args.framework == 'sglang':
        from sglang.test.test_utils import is_in_ci

        # if is_in_ci():
        #     from patch import launch_server_cmd
        # else:
        from sglang.utils import launch_server_cmd

        from sglang.utils import wait_for_server, print_highlight, terminate_process

        server_process, port = launch_server_cmd(f"""
        python3 -m sglang.launch_server \
        --model-path {model} \
        --tp 4 \
        --trust-remote-code \
        --host 0.0.0.0 --port 8123 --mem-fraction-static 0.9"""
        )

        wait_for_server(f"http://localhost:{port}")
        print(f"Server started on http://localhost:{port}")

# enable_prefix_caching=True

# nohup python model_serve.py --model_name qwen2.5-3B > logs/model &
# nohup python model_serve.py --model_name qwen2.5-7B-int4 > logs/model &
# nohup python model_serve.py --model_name qwen2.5-32B > logs/model &
# nohup python model_serve.py --model_name qwen3-32B > logs/model &
# nohup python model_serve.py --model_name QwQ > logs/model &
# nohup python model_serve.py --model_name llama3.1-70B > logs/model &
# nohup python model_serve.py --model_name qwen_long > logs/model &
# nohup python model_serve.py --model_name gemma3 > logs/model &
# nohup python model_serve.py --model_name phi4 > logs/model &

# nohup python model_serve.py --model_name llama3.1-DS > logs/model &
# nohup python model_serve.py --model_name qwen2.5-DS > logs/model &

# export VLLM_ATTENTION_BACKEND="XFORMERS"
# export VLLM_ATTENTION_BACKEND="FLASHMLA"
# export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
# export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
# export VLLM_ATTENTION_BACKEND="FLASHINFER"


