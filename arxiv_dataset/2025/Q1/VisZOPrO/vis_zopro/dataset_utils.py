from typing import List, Dict, Any
import random

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator
from trl.trainer.judges import DEFAULT_PAIRWISE_SYSTEM_PROMPT
from trl.trainer.utils import first_true_indices
from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from extract_preference_pairs import extract_dialogue


def generate_both_pairs(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[List[Dict[str, str]]],
    batch_size: int = 2,
    generation_length: int = 53,  # input_ids were generated with max_length=53
    filter_percentage: float = 0.6,  # filter out the worst 60% of completions
):
    # filter out the worst 60% of completions i.e. the ones with the highest scores
    prompts = prompts[:int(len(prompts) * filter_percentage)]
    prompt_ids = [tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True) for prompt in prompts]
    # generate second completions
    data_loader = DataLoader(Dataset.from_list([{"input_ids": prompt_id} for prompt_id in prompt_ids]), batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    policy.eval()
    answers_1 = []
    answers_2 = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(policy.device)
            # have same generation config as PPOTrainer # FIXME ?
            generations = policy.generate(
                inputs=input_ids,
                max_new_tokens=generation_length,
                num_return_sequences=2,
                temperature=0.7 + 1e-7,
                do_sample=True,
                top_k=0.0,
                top_p=1.0
            )
            prompt_len = len(input_ids[0])
            for i, gen in enumerate(generations):
                if i % 2 == 0:
                    answers_2.append(gen.tolist()[prompt_len:])
                else:
                    answers_1.append(gen.tolist()[prompt_len:])
    answers_2 = tokenizer.batch_decode(answers_2, skip_special_tokens=False)
    answers_1 = tokenizer.batch_decode(answers_1, skip_special_tokens=False)

    # remove user template and system template from prompts
    final_dataset = [{"prompt": prompt, "answer_1": gen, "answer_2": ans} for prompt, gen, ans in zip(prompts, answers_1, answers_2)]

    return final_dataset


# given some prompts and a policy, generate completions.
# These are to be paired with input_ids (which were generated earlier)
# to choose the better of the two completions
def generate_pairs(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts_ids: List[List[int]],
    answers_ids: List[List[int]],
    scores: List[float],
    batch_size: int = 2,
    generation_length: int = 53,  # input_ids were generated with max_length=53
    filter_percentage: float = 0.6,  # filter out the worst 60% of completions
):
    # sort scores and input_ids by how close score is to zero
    scores, answer_1_ids, prompt_ids = zip(*sorted(zip(scores, answers_ids, prompts_ids), key=lambda x: abs(x[0])))
    # remove duplicates from answer_1_ids and prompt_ids, if any, keep order
    unique_answer_1_ids = []
    if answer_1_ids:  # Check if the list is not empty
        unique_answer_1_ids.append(answer_1_ids[0])
        for ans_id in answer_1_ids[1:]:
            if ans_id != unique_answer_1_ids[-1]:
                unique_answer_1_ids.append(ans_id)

    unique_prompt_ids = []
    if prompt_ids:  # Check if the list is not empty
        unique_prompt_ids.append(prompt_ids[0])
        for prompt_id in prompt_ids[1:]:
            if prompt_id != unique_prompt_ids[-1]:
                unique_prompt_ids.append(prompt_id)

    # filter out the worst 60% of completions i.e. the ones with the highest scores
    answer_1_ids = list(unique_answer_1_ids[:int(len(unique_answer_1_ids) * filter_percentage)])
    prompt_ids = list(unique_prompt_ids[:int(len(unique_prompt_ids) * filter_percentage)])
    # remove all 0s and pad tokens from prompt_ids
    for i in range(len(prompt_ids)):
        # find first non-zero, non-pad token
        start_index_non_zero = first_true_indices(torch.tensor(prompt_ids[i]) != 0).item()
        start_index_non_pad = first_true_indices(torch.tensor(prompt_ids[i]) != tokenizer.pad_token_id).item()
        start_index = max(start_index_non_zero, start_index_non_pad)
        prompt_ids[i] = prompt_ids[i][start_index:]

    answers_1 = tokenizer.batch_decode(answer_1_ids, skip_special_tokens=False)
    prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)
    # generate second completions
    data_loader = DataLoader(Dataset.from_list([{"input_ids": prompt_id} for prompt_id in prompt_ids]), batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    policy.eval()
    answers_2 = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(policy.device)
            # have same generation config as PPOTrainer # FIXME ?
            generations = policy.generate(
                inputs=input_ids,
                max_new_tokens=generation_length,
                num_return_sequences=1,
                temperature=0.7 + 1e-7,
                do_sample=True,
                top_k=0.0,
                top_p=1.0
            )
            prompt_len = len(input_ids[0])
            answers_2.extend([gen.tolist()[prompt_len:] for gen in generations])
    answers_2 = tokenizer.batch_decode(answers_2, skip_special_tokens=False)

    # remove user template and system template from prompts
    final_dataset = [{"prompt": prompt, "answer_1": gen, "answer_2": ans} for prompt, gen, ans in zip(prompts, answers_1, answers_2)]

    return final_dataset


# judge the generated samples with an external model, locally loaded
def judge_samples(model_name, samples: List[Dict], batch_size: int = 2, system_prompt: str = None, shuffle_order: bool = True):
    # the following setup assumes the model can fit on a single GPU
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    accelerator = Accelerator()
    model.to(accelerator.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=accelerator.device)
    system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT

    prompts = [sample["prompt"] for sample in samples]
    completions = [[sample["answer_1"], sample["answer_2"]] for sample in samples]

    # Shuffle the order of the completions to avoid positional bias
    if shuffle_order:
        flip_mask = np.random.choice([True, False], size=len(prompts))
        completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

    contents = [[{"role": "user", "content": system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])}] for prompt, candidates in zip(prompts, completions)]

    preferences = pipe(contents, max_new_tokens=1, batch_size=batch_size)

    annotated_dataset = []
    for i, (prompt, candidates, preference) in enumerate(tqdm(zip(prompts, completions, preferences))):
        preference = preference[0]["generated_text"][-1]["content"].strip()
        if preference in ["0", "1"]:
            if not prompt or not candidates[0] or not candidates[1]:
                continue
            annotated_dataset.append({"prompt": prompt, "chosen": candidates[int(preference)], "rejected": candidates[1 - int(preference)]})

    error_rate = 1 - len(annotated_dataset) / len(samples)
    print("Finished judging samples with error rate (formatting issues or bad generations): {:.2f}".format(error_rate))
    return annotated_dataset


def process4pl(examples: Dict[str, Any], example_name, tokenizer, max_length, task="chat") -> Dict[str, Any]:
    # processes the dataset for policy training
    examples = extract_dialogue(examples, example_name, task)
    return_dict = {
        "input_ids": tokenizer.apply_chat_template(examples['prompt'], tokenize=True, max_length=max_length, add_generation_prompt=True),
        "prompt": examples['prompt'],  # this is used by some methods (e.g. online DPO), other methods should drop this column
    }
    return return_dict


class DatasetYielder:
    def __init__(self, train_data: Dataset, eval_data: Dataset, num_epochs: int, intraepoch_refinement: bool, iteration_percentage: float = 0.05):
        # if  intraepoch_refinement is True, yields the same whole datasets for each iteration
        # if False, yields 5% of the data for each iteration and repeats the process for num_epochs
        # TODO: add a warmup param so that an bigger chunk is yielded in the beginning of intraepoch_refinement
        self.train_data = train_data
        self.eval_data = eval_data
        self.num_epochs = num_epochs
        self.intraepoch_refinement = intraepoch_refinement
        self.epoch_counter = 0
        self.train_indices = []
        self.eval_indices = []

        if self.intraepoch_refinement:
            self.train_len = len(self.train_data)
            self.eval_len = len(self.eval_data)
            self.train_chunk_size = int(self.train_len * iteration_percentage)
            self.eval_chunk_size = int(self.eval_len * iteration_percentage)
            self.train_indices_all_epochs = []
            self.eval_indices_all_epochs = []

            for _ in range(self.num_epochs):
                # Create indices for each 5% chunk, ensuring no overlap within an epoch
                train_indices_epoch = []
                eval_indices_epoch = []

                temp_train_indices = list(range(self.train_len))
                random.shuffle(temp_train_indices)

                temp_eval_indices = list(range(self.eval_len))
                random.shuffle(temp_eval_indices)

                for i in range(0, self.train_len, self.train_chunk_size):
                    train_indices_chunk = temp_train_indices[i:i+self.train_chunk_size]
                    if len(train_indices_chunk) < self.train_chunk_size and len(train_indices_chunk) > 0:
                        train_indices_chunk = temp_train_indices[i:]
                    elif len(train_indices_chunk) == 0:
                        continue

                    train_indices_epoch.extend(train_indices_chunk)

                for i in range(0, self.eval_len, self.eval_chunk_size):

                    eval_indices_chunk = temp_eval_indices[i:i+self.eval_chunk_size]

                    if len(eval_indices_chunk) < self.eval_chunk_size and len(eval_indices_chunk) > 0:
                        eval_indices_chunk = temp_eval_indices[i:]
                    elif len(eval_indices_chunk) == 0:
                        continue

                    eval_indices_epoch.extend(eval_indices_chunk)

                self.train_indices_all_epochs.append(train_indices_epoch)
                self.eval_indices_all_epochs.append(eval_indices_epoch)

    def __iter__(self):
        self.epoch_counter = 0
        self.train_index = 0
        self.eval_index = 0
        return self

    def __next__(self):
        if not self.intraepoch_refinement:
            if self.epoch_counter < self.num_epochs:
                self.epoch_counter += 1
                return self.train_data, self.eval_data
            else:
                raise StopIteration
        else:
            if self.epoch_counter >= self.num_epochs:
                raise StopIteration

            train_indices_epoch = self.train_indices_all_epochs[self.epoch_counter]
            eval_indices_epoch = self.eval_indices_all_epochs[self.epoch_counter]

            train_indices = train_indices_epoch[self.train_index:self.train_index+self.train_chunk_size]
            eval_indices = eval_indices_epoch[self.eval_index:self.eval_index + self.eval_chunk_size]

            if not train_indices or not eval_indices:
                self.epoch_counter += 1
                self.train_index = 0
                self.eval_index = 0
                if self.epoch_counter >= self.num_epochs:
                    raise StopIteration
                else:
                    return self.__next__()

            train_chunk = self.train_data.select(train_indices)
            eval_chunk = self.eval_data.select(eval_indices)

            self.train_index += self.train_chunk_size
            self.eval_index += self.eval_chunk_size

            if self.train_index >= len(train_indices_epoch) and self.eval_index >= len(eval_indices_epoch):
                self.epoch_counter += 1
                self.train_index = 0
                self.eval_index = 0

            return train_chunk, eval_chunk
