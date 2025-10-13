'''
    This code is partly adopted from https://github.com/deepseek-ai/DeepSeek-Prover-V1.5
'''
import os
import time

import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer


from prover.utils import AttrDict, MODEL_FORMAT


class GeneratorProcess(mp.Process):
    def __init__(self, local_rank, node_rank, model_path, task_queue, request_statuses, lock, args, max_model_len = None):
        super().__init__()
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.model_path = model_path
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.max_model_len = max_model_len
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            n=1,
        )
        self.prompt_func = MODEL_FORMAT[args.mode]['prompt']
        self.output_func = MODEL_FORMAT[args.mode]['output']

    def run(self):
        seed = int(time.time()) % 1000 + (self.node_rank * 8 + self.local_rank) * 1000
        os.environ['LOCAL_RANK'] = str(self.local_rank)

        ### FIX MULTI GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.local_rank)
        torch.cuda.set_device(f'cuda:0')  # Explicitly set the GPU for this process
        ###
        if 'lora' in self.model_path:
            self.lora_path = self.model_path
            self.model_path = 'deepseek-ai/DeepSeek-Prover-V1.5-RL'
            llm = LLM(model=self.model_path, max_num_batched_tokens=8192, seed=seed, trust_remote_code=True, enable_lora=True, max_lora_rank=64) # REFORMAT THE CODE
        else:
            self.lora_path = ''
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            llm = LLM(model=self.model_path, trust_remote_code=True, max_model_len=self.max_model_len)

        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            model_inputs = [
                    ''.join([
                        item.get('_extra_header', str()),
                        self.prompt_func(item),
                        item.get('_extra_prompt', str()),
                    ]) for _, _, item in inputs
                ]
            if 'Kimina' in self.model_path:
                model_inputs = [
                    ''.join([
                        item.get('_extra_header', str()),
                        self.prompt_func(item),
                        item.get('_extra_prompt', str()),
                    ]) for _, _, item in inputs
                ]
                messages = []
                for model_input in model_inputs:
                    m = [
                            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
                            {"role": "user", "content": model_input}
                        ]
                    messages.append(m)

                prompts = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompts = model_inputs
            if self.lora_path == '':
                model_outputs = llm.generate(
                    prompts,
                    self.sampling_params,
                    use_tqdm=False,                )
            else:
                model_outputs = llm.generate(
                prompts,
                self.sampling_params,
                use_tqdm=False,
                lora_request=LoRARequest("lora_adapter", 1, self.lora_path) # CHANGE TO SUPPORT ALL NOT JUST LORA
                )
            outputs = [self.output_func(_output.outputs[0].text) for _output in model_outputs]
            with self.lock:
                for (_, request_id, _), output in zip(inputs, outputs):
                    self.request_statuses[request_id] = output
