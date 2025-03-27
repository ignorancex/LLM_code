import os
import torch
import platformdirs
from queue import Queue, Empty
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union

from .base import EngineLM, CachedEngine


class ChatLlama3_1(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = """
Cutting Knowledge Date: December 2023
Today Date: 22 September 2024

You are a helpful assistant"""

    def __init__(
            self,
            model_string: str = "/u/hjin3/.llama/checkpoints/Meta-Llama3.1-8B",
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            batch_size: int = 1,
    ):
        """
        :param model_string:
        :param system_prompt:
        """
        root = platformdirs.user_cache_dir("textgrad")
        model_name = model_string.split("/")[-1]
        cache_path = os.path.join(root, f"cache_{model_name}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_string, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_string, torch_dtype=torch.bfloat16, device_map="auto")

        self.model_string = model_string

        # Batch processing
        self.batch_size = batch_size
        self.queue = Queue()
        self._start_batch_thread()

    def _start_batch_thread(self):
        def process_batch():
            while True:
                inputs = []
                while len(inputs) < self.batch_size:
                    try:
                        if len(inputs) == 0:
                            item = self.queue.get()
                        else:
                            item = self.queue.get(timeout=10)
                        inputs.append(item)
                    except Empty:
                        print("Timeout waiting for batch items.")
                        break
                    except Exception as e:
                        print(f"Error in batch processing: {e}")

                if inputs:
                    prompts, system_prompts, result_queues = zip(*inputs)

                    # Generate for the batch
                    text_inputs = [
                        [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": prompt}]
                        for prompt, sys_prompt in zip(prompts, system_prompts)
                    ]

                    text_inputs = self.tokenizer.apply_chat_template(text_inputs,
                                                                     add_generation_prompt=True,
                                                                     tokenize=False)

                    inputs = self.tokenizer(text_inputs, padding="longest", return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    input_length = inputs["input_ids"].shape[1]

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=2000,
                            temperature=1e-6,
                            top_p=0.99,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    for i in range(outputs.size(0)):
                        generated_tokens = outputs[i, input_length:]
                        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        result_queues[i].put(generated_text)

        batch_thread = Thread(target=process_batch, daemon=True)
        batch_thread.start()

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str = None, **kwargs):
        cache_or_none = self._check_cache(system_prompt + content)
        if cache_or_none is not None:
            return cache_or_none

        result_queue = Queue()
        self.queue.put((content, system_prompt, result_queue))
        response = result_queue.get()
        self._save_cache(system_prompt + content, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

