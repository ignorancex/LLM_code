
from dataclasses import dataclass
import json
import os
import sqlite3
from string import ascii_uppercase
from typing import Literal

from clize import run
from scipy.stats import entropy
import numpy as np

from task._generation import *
from utils import *


@dataclass
class McqaChoices:
    label: str
    text: str


def norm_mcqa_label(l: str):
    LABEL_MAP = {
        str(n): l for n, l in
        zip(range(1, len(ascii_uppercase) + 1), ascii_uppercase)
    }
    return LABEL_MAP.get(l, l)


def zip_to_mcqa_choices(data: dict) -> list[McqaChoices]:
    return [
        McqaChoices(
            norm_mcqa_label(label), text,
        ) for label, text in zip(
            data["label"],
            data["text"]
        )
    ]


class MultipleChoiceSampleData:

    question: str
    choices: list[McqaChoices]
    answer: str

    def __init__(self, sample: dict, dataset_path: str) -> None:

        self.sample = sample

        if "mmlu" in dataset_path:
            self.question = sample["question"]
            self.choices = [
                McqaChoices(label, text)
                for label, text in zip("ABCD", sample["choices"])
            ]
            self.answer = "ABCD"[int(sample["answer"])]
        elif any(
            (name in dataset_path)
            for name in ("ai2_arc", "commonsense_qa")
        ):
            self.question = sample["question"]
            self.choices = zip_to_mcqa_choices(sample["choices"])
            self.answer = norm_mcqa_label(sample["answerKey"])

    @property
    def choices_str(self):
        return '\n'.join(
            f'{c.label}. {c.text}' for c in self.choices
        )

    @property
    def msg_for_directly_answer(self):
        labels_text = ', '.join(x.label for x in self.choices)
        content = "The following is a multiple-choice question. " + \
            "Please choose the most suitable one among " + \
            labels_text + " as the answer to this question.\n Question: " + \
            f"{self.question}\n{self.choices_str}"
        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        return msg

    @property
    def prompt_r1(self):
        labels_text = ', '.join(x.label for x in self.choices)
        mcq_prompt = "The following is a multiple-choice question. " + \
            "Please choose the most suitable one among " + \
            labels_text + " as the answer to this question.\n Q: "
        return mcq_prompt

    @property
    def msg_for_directly_answer_r1(self):
        content = "Answer this question.\n Question: " + \
            f"{self.question}\n{self.choices_str}"
        msg = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "<--think-->\n\n<--/think-->"}
        ]
        return msg

    def msg_for_thought(
        self,
        method: Literal["sbs", "mrpp", "ltm"],
    ):
        if method == "sbs":
            cot = "Think about it step by step. Tell me your thoughts, but don't tell me the answers now.\n"
            q = "Question:\n" + f"{self.question}\n{self.choices_str}"
        elif method == "mrpp":
            cot = "You need to perform multi-step reasoning, with each step carrying out as many basic operations as possible. Tell me your reasoning, but don't tell me the answers now.\n"
            q = "Question:\n" + f"{self.question}\n{self.choices_str}"
        elif method == "ltm":
            cot = "What subproblems must be solved before answering the inquiry? Tell me the subproblems and the answers to the subproblems, but don't tell me the answers to the inquiry now.\n"
            q = "Inquiry:\n" + f"{self.question}\n{self.choices_str}"

        msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": cot + q},
        ]
        return msg

    def msg_for_thought_r1(
        self, method: str = "sbs"
    ):
        if method == "sbs":
            cot = "Think about the question step by step and answer it. \n"
        elif method == "mrpp":
            cot = "Think about the question with each step carrying out as many basic operations as possible and answer it. \n"
        elif method == "ltm":
            cot = "Think about the question with subproblems must be solved before answering and answer it. \n"

        q = "Question:\n" + f"{self.question}\n{self.choices_str}"

        msg = [
            {"role": "user", "content": cot + q},
        ]
        return msg

    def content_for_answer_with_thought_r1(self,):
        cot = "Think about the question step by step and answer it. \n"
        q = "Question:\n" + f"{self.question}\n{self.choices_str}"
        return cot + q

    def msg_for_answer_r1(self, thoght_msg: list[dict[str, str]], thought: str):
        msg = [
            *thoght_msg,
            {"role": "assistant", "content": f"<--think-->\n{thought}\n<--/think-->"},
        ]
        return msg

    def msg_for_answer(
        self,
        thoght_msg: list[dict[str, str]],
        thought: str,
        method: Literal["sbs", "mrpp", "ltm"],
    ):
        if method == "ltm":
            q = "inquiry"
        else:
            q = "question"
        labels_text = ', '.join(x.label for x in self.choices)
        content = "Now, Please choose the most suitable one among " + \
            labels_text + f" as the answer of the {q}."
        msg = [
            *thoght_msg,
            {"role": "assistant", "content": thought},
            {"role": "user", "content": content},
        ]
        return msg

    @torch.no_grad()
    def evaluate(
        self,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        generated: dict,
        evalued: list[dict, dict, dict, dict] | None = None
    ):
        METHODS = ["base", "sbs", "mrpp", "ltm"]
        changed = False
        if evalued is None:
            evalued = [{}, {}, {}, {}]

        options = [c.label for c in self.choices]
        option_ids: torch.Tensor = tokenizer.encode(
            "".join(f" {l}" for l in options),
            return_tensors="pt",
            add_special_tokens=False
        ).to(llm.device).squeeze(0)

        if option_ids.size(-1) != len(self.choices):
            print(
                "error: len(option_ids) != len(self.choices)",
                len(option_ids.size(-1)),
                len(self.choices),
            )
            print("options:", options)
            print("option_ids:", option_ids)
            return None, False

        for method in list(generated.keys()) + ["base"]:
            if method not in METHODS:
                continue
            eval_id = METHODS.index(method)
            if evalued[eval_id]:
                continue
            changed = True

            if "r1" in llm.config.name_or_path.lower():
                if method == "base":
                    msg = self.msg_for_directly_answer_r1
                else:
                    thought = generated[method]
                    msg = self.msg_for_thought_r1(method)
            else:
                if method == "base":
                    msg = self.msg_for_directly_answer
                else:
                    thought = generated[method]
                    msg = self.msg_for_answer(
                        self.msg_for_thought(method),
                        thought, method,
                    )

            msg.append(
                {"role": "assistant", "content": "The correct answer is"})
            msg_txt = tokenizer.apply_chat_template(
                msg, tokenize=False, continue_final_message=True
            )
            inputs = tokenizer(msg_txt, return_tensors="pt",).to(llm.device)
            outputs = llm.forward(**inputs)
            all_tokens_prob = torch.softmax(outputs.logits, dim=-1)[-1, -1]
            option_probs = all_tokens_prob.gather(-1, option_ids,)
            option_probs = option_probs.cpu().numpy()
            normed_option_probs: np.ndarray = option_probs / option_probs.sum()
            prediction_option_index = option_probs.argmax().item()
            prediction = options[prediction_option_index]
            truth_option_index = options.index(self.answer)

            if self.answer == "":
                correct = -1
                true_prob = .0
                pred_prob = .0
            elif self.answer == prediction:
                correct = 1
                true_prob = normed_option_probs[truth_option_index].item()
                pred_prob = normed_option_probs[prediction_option_index].item()
            else:
                correct = 0
                true_prob = normed_option_probs[truth_option_index].item()
                pred_prob = normed_option_probs[prediction_option_index].item()

            evalued[eval_id] = dict(
                origin_probs=option_probs.tolist(),
                normed_probs=normed_option_probs.tolist(),
                prediction=prediction,
                correct=correct,
                true_prob=true_prob,
                pred_prob=pred_prob,
                entropy=entropy(normed_option_probs).item(),
            )
            torch.cuda.empty_cache()

        return evalued, changed


def get_evaled_data_by_id(id: int, db: sqlite3.Connection):
    cur = db.cursor()
    data = cur.execute(
        "SELECT * FROM evaled WHERE id = ?",
        (id,)
    ).fetchone()
    if data is None:
        return None
    else:
        return [json.loads(d) for d in data[1:]]


def insert_or_update_evaled_data_by_id(id: int, data: tuple[str], db: sqlite3.Connection):
    cur = db.cursor()
    cur.execute(
        "INSERT INTO evaled (id, base, sbs, mrpp, ltm) VALUES (?, ?, ?, ?, ?)",
        (id, *data)
    )
    db.commit()


def init_database(db_path):
    db = sqlite3.connect(db_path)

    db.execute("""drop table if exists evaled""")
    db.execute(
        f"""
    create table if not exists evaled (
        id integer primary key,
        base text,
        ltm text,
        mrpp text,
        sbs text )"""
    )

    db.commit()
    return db


def main(
    *,
    model_name_or_path: 'm' = "meta-llama/Llama-3.1-8B-Instruct",  # type: ignore
    data_name_or_path: 'd' = "",  # type: ignore
    data_start_index: 'S' = 0,  # type: ignore
    data_read_length: 'L' = 0,  # type: ignore
    data_batch_size: 'b' = 128,  # type: ignore
    gpu_mem_util: 'r' = 0.5,  # type: ignore
):
    """Main script to execute the Pilot Experiment.

    :param model_name_or_path: Path to the pre-trained model or its identifier. Examples include "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", or any valid local directory path.
    :param data_name_or_path: Path to the dataset or its identifier. Examples include "cais/mmlu", "allenai/ai2_arc", "tau/commonsense_qa", or any valid local directory path.
    :param data_start_index: Specifies the starting index in the dataset from which to begin reading.
    :param data_read_length: Indicates how many entries to read from the dataset beginning at `data_start_index`. A value of 0 indicates that all subsequent data following `data_start_index` should be used.
    :param data_batch_size: Defines the batch size used during the processing of the dataset.
    :param gpu_id: Identifies which GPU to utilize for computations. This value will be assigned to the `CUDA_VISIBLE_DEVICES` environment variable.
    :param gpu_mem_util: Sets the desired GPU memory utilization rate for the model execution. This is a parameter specific to vllm.
    """

    db = init_database(
        f"{model_name_or_path.split('/')[-1].split('-')[0]}." +
        f"{data_name_or_path.split('/')[-1]}." +
        f"{data_start_index}." +
        "sqlite3"
    )
    print(torch.cuda.device_count())
    llm = LLM(
        model=model_name_or_path,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    dataset = load_hf_dataset(
        data_name_or_path,
        data_start_index, data_read_length,
    )

    for i, batch in enumerate(dataset.iter(data_batch_size)):
        # generation
        print(
            f"## Batch {i} / Samples {i*data_batch_size} / Total {len(dataset)}"
        )
        batch = [dict(zip(batch.keys(), t)) for t in zip(*batch.values())]
        samples = [
            MultipleChoiceSampleData(
                d, data_name_or_path
            ) for d in batch
        ]
        max_tokens = 512

        ltm_thoughts = generate_thought(
            [sample.msg_for_thought("ltm", ) for sample in samples],
            llm, tokenizer, max_tokens
        )
        mrpp_thoughts = generate_thought(
            [sample.msg_for_thought("mrpp",) for sample in samples],
            llm, tokenizer, max_tokens
        )
        sbs_thoughts = generate_thought(
            [sample.msg_for_thought("sbs", ) for sample in samples],
            llm, tokenizer, max_tokens
        )

        # evaluation
        for j, sample in enumerate(samples):
            _id = i*data_batch_size + j + data_start_index
            thoughts = {
                "ltm": post_norm_text(ltm_thoughts[j], model_name_or_path),
                "mrpp": post_norm_text(mrpp_thoughts[j], model_name_or_path),
                "sbs": post_norm_text(sbs_thoughts[j], model_name_or_path),
            }
            evaled = get_evaled_data_by_id(_id, db)
            evaled, changed = sample.evaluate(
                model, tokenizer,
                thoughts, evalued=evaled
            )
            if evaled is not None and changed:
                insert_or_update_evaled_data_by_id(
                    _id, [json.dumps(e) for e in evaled], db
                )

    db.close()


if __name__ == "__main__":
    run(main)
