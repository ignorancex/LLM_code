import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import json
import sys
import logging
from typing import Union, Dict
from tqdm import tqdm
import re, random


class PandaLMBatchInferenceProviderModified(object):
    """
    Evaluate batch responses with PandaLM
    """

    def __init__(
        self,
        model_path,
        temperature=0,
        top_p=1,
        top_k=1,
        num_beams=4,
        max_new_tokens=512,
        repetition_penalty=1.2,
        device_map="auto",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.device_map = device_map
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            early_stopping=True,
            repetition_penalty=self.repetition_penalty,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        )
        if tokenizer.pad_token is None:
            self.smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
        )
        self.tokenizer = tokenizer

        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        self.model = model
        self.prepared = []
        self.pattern = re.compile(
            r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
        )

    def balanced_preprocess_input(
        self, instruction, input, resp1, resp2, max_len_prompt=1024, skip_flag=False
    ):
        if input:
            input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
        else:
            input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n"
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp1 = f"### Response 1:\n{resp1}\n\n"
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        resp2 = f"### Response 2:\n{resp2}\n\n"
        eval_sequence = "### Evaluation:\n"

        # make sure to remove bos token from the responses and evaluation
        input_tokens = self.tokenizer(
            input_sequence, return_tensors="pt", padding=True
        )["input_ids"]
        reps1_tokens = self.tokenizer(
            resp1,
            return_tensors="pt",
            padding=True,
        )["input_ids"]
        reps2_tokens = self.tokenizer(
            resp2,
            return_tensors="pt",
            padding=True,
        )["input_ids"]
        eval_tokens = self.tokenizer(
            eval_sequence,
            return_tensors="pt",
            padding=True,
        )["input_ids"]

        total_input_size = (
            input_tokens.shape[1]
            + eval_tokens.shape[1]
            + reps1_tokens.shape[1]
            + reps2_tokens.shape[1]
        )

        if (input_tokens.shape[1] + eval_tokens.shape[1]) < max_len_prompt:
            per_resp_len = (
                max_len_prompt - (input_tokens.shape[1] + eval_tokens.shape[1])
            ) // 2
            reps1_tokens, reps2_tokens = (
                reps1_tokens[:, :per_resp_len],
                reps2_tokens[:, :per_resp_len],
            )
        else:
            per_resp_len = (max_len_prompt - (eval_tokens.shape[1])) // 3
            input_tokens, reps1_tokens, reps2_tokens = (
                input_tokens[:, :per_resp_len],
                reps1_tokens[:, :per_resp_len],
                reps2_tokens[:, :per_resp_len],
            )
        reps1_tokens[:, :2], reps2_tokens[:, :2], eval_tokens[:, :2] = (
            torch.Tensor([[2277, 29937]]),
            torch.Tensor([[2277, 29937]]),
            torch.Tensor([[2277, 29937]]),
        )
        input_ids = torch.cat(
            [input_tokens, reps1_tokens, reps2_tokens, eval_tokens], dim=1
        ).to(self.model.device)
        if skip_flag:
            is_truncated = bool(total_input_size > max_len_prompt)
            return input_ids, is_truncated
        return input_ids

    def build_pandalm_prompt(
        self, instruction, input, resp1, resp2, result=None, explain=None, ref=None
    ):
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"
        if input:
            input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n{rsp}\n\n### Evaluation:\n"
        else:
            input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n{rsp}\n\n### Evaluation:\n"
        if result:
            output_sequence = (
                f"{result}\n\n### Reason: {explain}\n\n### Reference: {ref}\n"
            )
            return input_sequence, output_sequence
        else:
            return input_sequence

    def parse_pandalm_response(self, text):
        sp = text.strip().split("\n")
        if sp[0] in ["1", "2"]:
            return int(sp[0])
        elif sp[0].lower() == "tie":
            return 0
        else:
            return 0

    def smart_tokenizer_and_embedding_resize(
        self,
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def preprocess_input(self, instruction, input, response1, response2):
        prompt = self.build_pandalm_prompt(instruction, input, response1, response2)
        prompt_encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = prompt_encoded["input_ids"].to(self.model.device)
        return input_ids

    def generate_response(self, input_ids):
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.max_new_tokens,
            )

        if self.num_beams > 1 or (
            self.temperature == 0 and self.top_p == 1 and self.top_k == 1
        ):
            sequence = generation_output.sequences[0]
        else:
            sequences = generation_output.sequences
            sequence_scores = generation_output.scores

            # Compute the total log probability for each sequence
            total_scores = []
            for seq_idx, sequence in enumerate(sequences):
                # Accumulate log probabilities for the tokens in the sequence
                log_probs = torch.log_softmax(sequence_scores[seq_idx], dim=-1)
                total_log_prob = (
                    log_probs.sum()
                )  # Total log probability of the sequence
                total_scores.append((total_log_prob, sequence))

            # Sort sequences by total log probability (most probable first)
            sequence = sorted(total_scores, key=lambda x: x[0], reverse=True)[0][1]

        sequence_cpu = sequence.to("cpu")
        del (generation_output, sequence)
        torch.cuda.empty_cache()
        return sequence_cpu

    def text_from_sequence(self, sequence):
        output = self.tokenizer.decode(sequence)
        resp = self.postprocess_output(output)
        resp = self.filter_special_token(resp)
        return resp

    def postprocess_output(self, text):
        text = text.strip().split("### Evaluation:")[1].strip()
        self.pattern.sub("", text.strip()).strip()
        return text

    def filter_special_token(self, text):
        return self.pattern.sub("", text.strip()).strip()

    def infer_request(self, instruction, input, response1, response2):
        prompt = self.build_pandalm_prompt(instruction, input, response1, response2)
        prompt_encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        if prompt_encoded["input_ids"].shape[1] > 8192:
            return "Input too long"
        input_ids = prompt_encoded["input_ids"].to(self.model.device)
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.max_new_tokens,
            )

        if self.num_beams > 1 or (
            self.temperature == 0 and self.top_p == 1 and self.top_k == 1
        ):
            sequence = generation_output.sequences[0]
        else:
            sequences = generation_output.sequences
            sequence_scores = generation_output.scores

            # Compute the total log probability for each sequence
            total_scores = []
            for seq_idx, sequence in enumerate(sequences):
                # Accumulate log probabilities for the tokens in the sequence
                log_probs = torch.log_softmax(sequence_scores[seq_idx], dim=-1)
                total_log_prob = (
                    log_probs.sum()
                )  # Total log probability of the sequence
                total_scores.append((total_log_prob, sequence))

            # Sort sequences by total log probability (most probable first)
            sequence = sorted(total_scores, key=lambda x: x[0], reverse=True)[0][1]

        sequence_cpu = sequence.to("cpu")
        output = self.tokenizer.decode(sequence_cpu)
        generated = self.postprocess_output(output)
        resp = self.filter_special_token(generated)

        del (input_ids, generation_output, sequence)
        torch.cuda.empty_cache()

        return resp
