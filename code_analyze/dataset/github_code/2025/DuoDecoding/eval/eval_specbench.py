import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import ipdb
import random
from src.utils import seed_everything, parse_arguments
from src.engine import Decoding
from collections import Counter
from fastchat.model import get_conversation_template


class EvalHumaneval(Decoding):
    def __init__(self, args):
        super().__init__(args)

        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()

        self.draft_time = []
        self.target_time = []
        self.acc_num = []

    def load_data(self):
        # load evaluation data
        self.color_print(f"Loading SpecBench/{self.args.sub_domain} data...", 3)
        data = []
        with open(
            os.path.join(self.args.data_path, f"{self.args.sub_domain}.jsonl")
        ) as f:
            for line in f.readlines():
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["turns"][0])
                encode_special_token_flag = not (
                    "Llama-3.1" in self.args.draft_model
                    and "Llama-3.1" in self.args.target_model
                )
                # print(datum["input_text"])
                input_ids = self.tokenizer.encode(
                    datum["input_text"], add_special_tokens=encode_special_token_flag
                )
                datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                data.append(datum)
        self.data = data

    def preprocess(self, input_text):
        if "vicuna" in self.args.target_model:
            text = input_text.strip()
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
        else:
            if self.args.sub_domain == "summarization":

                text = input_text.strip()
                prompt = text + " TL;DR: "

            elif self.args.sub_domain == "translation":
                prompt = (
                    "Translate German to English. German: "
                    + input_text[len("Translate German to English: ") :]
                    + " English: "
                )
            elif self.args.sub_domain == "math_reasoning":
                prompt = f"{input_text} Let's think step by step.\nStep 1:"
            elif self.args.sub_domain == "rag":
                prompt = input_text
            else:
                text = input_text.strip()
                conv = get_conversation_template("vicuna")
                conv.append_message(conv.roles[0], text)
                conv.append_message(conv.roles[1], None)
                conv.stop_str = "</s>"
                prompt = conv.get_prompt()
        return prompt

    def postprocess(self, input_text, output_text):
        if output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[
                len(input_text) + len(self.tokenizer.bos_token) + 1 :
            ]  # tokenizer will add a '<s> ' at the beginning of the text.
        else:
            generation = output_text[len(input_text) :]
        stop_words = [
            "\nclass",
            "\ndef",
            "\n#",
            "\n@",
            "\nprint",
            "\nif",
            "\n```",
            self.tokenizer.eos_token,
        ]
        for stop_word in stop_words:
            if stop_word in generation:
                next_line = generation.index(stop_word)
                generation = generation[:next_line].strip()
        output_text = input_text + "\n    " + generation
        output_text = output_text.replace("\t", "    ")

        return output_text

    @torch.no_grad()
    def eval(self):
        if self.args.eval_mode == "small" or self.args.eval_mode == "large":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "sd":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        elif self.args.eval_mode == "duodec":
            decoding = self.duodecoding
        elif self.args.eval_mode == "pld":
            decoding = self.pld_forward
        elif self.args.eval_mode == "lade":
            decoding = self.lookahead_forward
        elif self.args.eval_mode == "rest":
            decoding = self.rest_forward
        else:
            raise NotImplementedError

        out_path = os.path.join(
            self.args.exp_name,
            f"{self.args.eval_mode}_specbench_{self.args.sub_domain}.jsonl",
        )
        out_f = open(out_path, "a")
        wall_times = {"time": [], "num_tokens": [], "ttft": []}
        for _ in range(self.args.num_samples_per_task):
            # set random seed. Ensure each experiment runs with a unique random seed.
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)

            # warm up
            n = 10
            print(f"Start warm up...")
            for datum in tqdm.tqdm(
                self.data,
                total=len(self.data),
                disable=not self.accelerator.is_main_process,
                ncols=50,
            ):
                input_ids = datum["input_ids"]
                generate_ids = decoding(input_ids)
                n = n - 1
                if n == 0:
                    break

            self.draft_forward_times = 0
            self.target_forward_times = 0
            self.num_acc_tokens = []
            self.prob_with_flag = []  # draft每个token的prob与他是否被接收

            for datum in tqdm.tqdm(
                self.data,
                total=len(self.data),
                disable=not self.accelerator.is_main_process,
                ncols=50,
            ):
                input_ids = datum["input_ids"]
                torch.cuda.synchronize()
                start_time = time.time()
                generate_ids = decoding(input_ids)

                torch.cuda.synchronize()
                end_time = time.time()
                if self.accelerator.is_main_process:

                    wall_times["time"].append(end_time - start_time)
                    wall_times["num_tokens"].append(
                        generate_ids.shape[1] - input_ids.shape[1]
                    )
                    output = self.postprocess(
                        datum["input_text"], self.tokenizer.decode(generate_ids[0, :])
                    )
                    out_f.write(
                        json.dumps(
                            {
                                "question_id": datum["question_id"],
                                "time": end_time - start_time,
                                "new_tokens": generate_ids.shape[1]
                                - input_ids.shape[1],
                                "completion": output,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                out_f.flush()
        out_f.close()

        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)

        self.accelerator.wait_for_everyone()

        if (
            self.accelerator.num_processes == 1 and self.accelerator.is_main_process
        ) or (
            self.accelerator.num_processes == 2 and not self.accelerator.is_main_process
        ):
            print(
                f"\033[92mtarget model forward times: {self.target_forward_times}\033[0m"
            )

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
            speed_std = (
                (
                    torch.tensor(wall_times["num_tokens"])
                    / torch.tensor(wall_times["time"])
                )
                .std()
                .item()
            )
            self.color_print(
                f"generate speed (tokens / second):  {speed:.2f} with std {speed_std}",
                2,
            )

        if self.accelerator.is_main_process:
            try:
                self.color_print(
                    f"Mean accepted tokens: {sum(self.num_acc_tokens) / len(self.num_acc_tokens)}"
                )
            except:
                pass



if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalHumaneval(args)
    alg.eval()
