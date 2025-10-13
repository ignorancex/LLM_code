from dataclasses import dataclass

import tiktoken
from openai.types import CompletionUsage

together_json_support_models = [
    # TODO doc says it work but it seems currently very slow
    # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "togethercomputer/CodeLlama-34b-Instruct",
]


@dataclass
class LLMModelSpec:
    """
    Info about an endpoint model containing cost, name and provider
    """

    name: str
    provider: str
    cost_prompt: float  # cost for 1M token
    cost_completion: float  # cost for 1M token

    def support_json_output(self) -> bool:
        # JSON is only supported for a few models in Together: https://docs.together.ai/docs/json-mode
        return self.provider != "together" or self.name in together_json_support_models

    def cost(self, usage: CompletionUsage) -> float:
        """
        :param usage:
        :return: dollar cost of the query made
        """
        return (
            usage.prompt_tokens * self.cost_prompt
            + usage.completion_tokens * self.cost_completion
        ) / 1e6


class llms:
    vllm_llama3_8B = "meta-llama/Meta-Llama-3-8B-instruct"
    vllm_llama3_70B = "meta-llama/Meta-Llama-3-70B-instruct"
    vllm_llama31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    vllm_llama31_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    vllm_llama31_70B_FP8 = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"
    together_llama_31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    together_llama_31_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    together_llama_33_70B = "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo"
    together_llama_3_8B = "meta-llama/Llama-3-8b-chat-hf"
    together_llama_3_70B = "meta-llama/Llama-3-70b-chat-hf"
    together_mistral_8x7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    smollm_2B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    qwen_0_5B = "Qwen/Qwen2.5-0.5B-Instruct"
    qwen_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
    qwen_3B = "Qwen/Qwen2.5-3B-Instruct"
    qwen_7B = "Qwen/Qwen2.5-7B-Instruct"
    qwen_14B = "Qwen/Qwen2.5-14B-Instruct"
    qwen_32B = "Qwen/Qwen2.5-32B-Instruct"
    qwen_32B_8B = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
    qwen_72B = "Qwen/Qwen2.5-72B-Instruct"
    qwen_72B_INT8 = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
    gemma_2B = "google/gemma-2-2b-it"
    gemma_9B = "google/gemma-2-9b-it"
    gemma_27B = "google/gemma-2-27b-it"
    judgelm_7B = "BAAI/JudgeLM-7B-v1.0"
    pandalm_7B = "WeOpenML/PandaLM-7B-v1"


model_specs = [
    LLMModelSpec(
        name="gpt-4-1106-preview",
        provider="openai",
        cost_prompt=10.0,
        cost_completion=30.0,
    ),
    LLMModelSpec(
        name="gpt-4o-2024-05-13",
        provider="openai",
        cost_prompt=5.0,
        cost_completion=15.0,
    ),
    LLMModelSpec(
        name="gpt-4o-mini-2024-07-18",
        provider="openai",
        cost_prompt=0.150,
        cost_completion=0.600,
    ),
    LLMModelSpec(
        name="gpt-3.5-turbo-0125",
        provider="openai",
        cost_prompt=0.5,
        cost_completion=1.5,
    ),
    LLMModelSpec(
        name=llms.together_llama_3_70B,
        provider="togetherai",
        cost_prompt=0.9,
        cost_completion=0.9,
    ),
    LLMModelSpec(
        name=llms.together_llama_3_8B,
        provider="togetherai",
        cost_prompt=0.2,
        cost_completion=0.2,
    ),
    LLMModelSpec(
        name=llms.together_llama_33_70B,
        provider="togetherai",
        cost_prompt=0.88,
        cost_completion=0.88,
    ),
    LLMModelSpec(
        name=llms.together_llama_31_70B,
        provider="togetherai",
        cost_prompt=0.88,
        cost_completion=0.88,
    ),
    LLMModelSpec(
        name=llms.together_llama_31_8B,
        provider="togetherai",
        cost_prompt=0.2,
        cost_completion=0.2,
    ),
    LLMModelSpec(
        name=llms.together_mistral_8x7B,
        provider="togetherai",
        cost_prompt=1.2,
        cost_completion=1.2,
    ),
    # use prices of together available there https://www.together.ai/pricing
    LLMModelSpec(
        name=llms.vllm_llama3_8B,
        provider="vllm",
        cost_prompt=0.2,
        cost_completion=0.2,
    ),
    LLMModelSpec(
        name=llms.vllm_llama3_70B,
        provider="vllm",
        cost_prompt=0.9,
        cost_completion=0.9,
    ),
    LLMModelSpec(
        # computed by summing tokens and dividing by runtime
        name=llms.vllm_llama31_8B,
        provider="vllm",
        cost_prompt=0.11,
        cost_completion=0.11,
    ),
    LLMModelSpec(
        name=llms.vllm_llama31_70B,
        provider="vllm",
        cost_prompt=0.9,
        cost_completion=0.9,
    ),
    LLMModelSpec(
        # computed by summing tokens and dividing by runtime
        name=llms.vllm_llama31_70B_FP8,
        provider="vllm",
        cost_prompt=0.35,
        cost_completion=0.35,
    ),
    LLMModelSpec(
        name=llms.smollm_2B,
        provider="vllm",
        cost_completion=0.06,  # price for 3B
        cost_prompt=0.06,
    ),
    LLMModelSpec(
        name=llms.qwen_0_5B,
        provider="vllm",
        cost_completion=0.06,  # price for 3B
        cost_prompt=0.06,
    ),
    LLMModelSpec(
        name=llms.qwen_1_5B,
        provider="vllm",
        cost_completion=0.06,  # price for 3B
        cost_prompt=0.06,
    ),
    LLMModelSpec(
        name=llms.qwen_3B,
        provider="vllm",
        cost_completion=0.1,  # all other up to 4B
        cost_prompt=0.1,
    ),
    LLMModelSpec(
        # computed by summing tokens and dividing by runtime
        name=llms.qwen_7B,
        provider="vllm",
        cost_completion=0.12,
        cost_prompt=0.12,
    ),
    LLMModelSpec(
        name=llms.qwen_14B,
        provider="vllm",
        cost_completion=0.3,  # all others between 8.1B-21B
        cost_prompt=0.3,
    ),
    LLMModelSpec(
        name=llms.qwen_32B,
        provider="vllm",
        cost_completion=0.8,  # all others between 21.1B-41B
        cost_prompt=0.8,
    ),
    LLMModelSpec(
        # computed by summing tokens and dividing by runtime
        name=llms.qwen_32B_8B,
        provider="vllm",
        cost_completion=0.36,  # all others between 21.1B-41B
        cost_prompt=0.36,
    ),
    LLMModelSpec(
        name=llms.qwen_72B,
        provider="vllm",
        cost_completion=1.2,
        cost_prompt=1.2,
    ),
    LLMModelSpec(
        # computed by summing tokens and dividing by runtime
        name=llms.qwen_72B_INT8,
        provider="vllm",
        cost_completion=0.58,
        cost_prompt=0.58,
    ),
    LLMModelSpec(
        name=llms.gemma_2B,
        provider="vllm",
        cost_completion=0.06,  # price for 3B
        cost_prompt=0.06,
    ),
    LLMModelSpec(
        # computed by summing tokens and dividing by runtime
        name=llms.gemma_9B,
        provider="vllm",
        cost_completion=0.14,  # all others between 8.1B-21B
        cost_prompt=0.14,
    ),
    LLMModelSpec(
        # computed by summing tokens and dividing by runtime
        name=llms.gemma_27B,
        provider="vllm",
        cost_completion=0.3,  # all others between 21.1B-41B
        cost_prompt=0.3,
    ),
    LLMModelSpec(
        name=llms.judgelm_7B,
        provider="vllm",
        cost_completion=0.2,  # all others between 4B-8B
        cost_prompt=0.2,
    ),
    LLMModelSpec(
        name=llms.pandalm_7B,
        provider="vllm",
        cost_completion=0.2,  # all others between 4B-8B
        cost_prompt=0.2,
    ),
]
name_to_llm_spec: dict[str, LLMModelSpec] = {model.name: model for model in model_specs}
name_to_llm_spec["gpt-4o"] = name_to_llm_spec["gpt-4o-2024-05-13"]
name_to_llm_spec["gpt-4o-mini"] = name_to_llm_spec["gpt-4o-mini-2024-07-18"]


def num_tokens(message: str, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(message, allowed_special={"<|endoftext|>"}))


def cost_together_completion(
    prompt: str, completion: str, model="gpt-4o-mini-2024-07-18"
):
    spec = name_to_llm_spec[model]
    if not isinstance(prompt, str):
        prompt = ""
    if not isinstance(completion, str):
        completion = ""

    cost = (
        num_tokens(prompt) * spec.cost_prompt
        + num_tokens(completion) * spec.cost_completion
    )
    return cost * 1e-6


if __name__ == "__main__":
    print(("Hello whats up"))
    print(
        cost_together_completion(
            prompt="Hello whats up",
            completion="I am fine",
            model=llms.together_llama_3_70B,
        )
    )
