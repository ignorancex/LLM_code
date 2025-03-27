import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewards.rewards import RewardAggregater
from rewards.prm_utils import CODEPRM_PROMPT
from rewards.examples import rewards_test_examples


def test():
    model = AutoModelForCausalLM.from_pretrained(
        'path/to/PRM',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('path/to/PRM')
    aggregator = RewardAggregater(
        model=model,
        tokenizer=tokenizer
    )

    for test_example in rewards_test_examples:
        question, reasoning_steps, test_cases = test_example['question'], test_example['reasoning_steps'], test_example['test_cases']
        prompt = CODEPRM_PROMPT.format(question=question)
        test_reward = aggregator.update_reward(
            prompt,
            reasoning_steps,
            test_cases,
            1,
        )
        print(test_reward)

if __name__ == '__main__':
    test()