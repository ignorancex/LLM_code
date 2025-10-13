import json
import os
import gc
import torch
from abc import ABC, abstractmethod
from collections import defaultdict
import vllm
import litellm
from transformers import AutoTokenizer
from vllm.distributed.parallel_state import destroy_model_parallel


class PromptModel(ABC):
    def __init__(
        self,
        provider="vllm",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_tokens=1024,
        max_model_len=None,
        temperature=0.0,
        top_p=1.0,
        tensor_parallel_size=1,
        base_url="https://cmu.litellm.ai",
        api_key=None,
        parent_prompt_model=None,
        gpu_memory_utilization=0.9,
        logprobs=None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs

        if self.provider == "vllm":
            self.llm = vllm.LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.sampling_params = vllm.SamplingParams(
                use_beam_search=False,
                best_of=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=logprobs,
                stop=[
                    "<|im_end|>",
                    "<|eot_id|>",
                    "<|END_OF_TURN_TOKEN|>",
                ],
            )
            if tensor_parallel_size > 1:
                import ctypes

                ctypes.CDLL(None).uselocale(-1)
        elif self.provider == "litellm":
            self.base_url = base_url
            self.api_key = api_key or os.environ.get("LITELLM_API_KEY")
            # assert self.api_key is not None, "api_key must be provided for litellm"
        elif self.provider == "parent_prompt_model":
            assert (
                parent_prompt_model is not None
            ), "parent_prompt_model must be provided for parent_prompt_model"
            self.parent_prompt_model = parent_prompt_model
        else:
            raise ValueError(f"Invalid provider: {provider}")

    @abstractmethod
    def prepare_chat(self, *args, **kwargs):
        pass

    def cleanup_model(self):
        if self.provider == "vllm":
            destroy_model_parallel()
            del self.llm.llm_engine.model_executor
            del self.llm
        gc.collect()
        torch.cuda.empty_cache()

    def prepare_prompt(self, chat: list[dict[str, str]]) -> str:
        if self.provider == "parent_prompt_model":
            return self.parent_prompt_model.prepare_prompt(chat)

        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        if self.tokenizer.bos_token is not None:
            prompt = prompt.replace(self.tokenizer.bos_token, "")
        return prompt

    def generate(self, chats, unique_only=True):
        if self.provider == "parent_prompt_model":
            return self.parent_prompt_model.generate(chats)

        # Create a mapping of unique prompts to their indices
        # this is avoid duplicate computation when generating questions
        # TODO: is this the right abstraction layer to do this?
        def _hash_chat(chat: list[dict[str, str]]):
            return json.dumps(chat, sort_keys=True)

        if unique_only:
            chat_to_indices = defaultdict(list)
            unique_chats = []
            for i, chat in enumerate(chats):
                if _hash_chat(chat) not in chat_to_indices:
                    unique_chats.append(chat)
                chat_to_indices[_hash_chat(chat)].append(i)
        else:
            unique_chats = chats

        if self.provider == "vllm":
            prompts = [self.prepare_prompt(chat) for chat in unique_chats]
            responses = self.llm.generate(prompts, self.sampling_params)
            unique_outputs = [response.outputs[0].text for response in responses]
        elif self.provider == "litellm":
            responses = litellm.batch_completion(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                messages=unique_chats,
                max_retries=10,
            )

            unique_outputs = [
                response.choices[0].message.content for response in responses
            ]

        assert len(unique_chats) == len(unique_outputs)

        if unique_only:
            # map the unique outputs back to the original indices
            outputs = [None] * len(chats)
            for unique_chat, unique_output in zip(unique_chats, unique_outputs):
                for original_index in chat_to_indices[_hash_chat(unique_chat)]:
                    outputs[original_index] = unique_output
        else:
            outputs = unique_outputs

        # confirm no None values
        assert all(out is not None for out in outputs), "None values in outputs"

        return outputs
