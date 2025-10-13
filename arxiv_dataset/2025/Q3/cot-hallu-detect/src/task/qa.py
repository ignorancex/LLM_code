
import os
import json
import sqlite3

import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    RobertaModel, RobertaTokenizer
)
from clize import run

from utils import (
    load_model_and_tokenizer,
    load_nli_model_and_tokenizer,
    load_embd_model_and_tokenizer,
)
from task._generation import *
from task._qa_utils import (
    metric_max_over_ground_truths,
    calcu_exact_match_score,
    calcu_max_rouge_l_over_ground_truths,
    calcu_sentences_sims,
)
from task._detection import detect


class QaSample:

    def __init__(self, sample: dict, generated: dict, dataset_path: str) -> None:
        self.question = get_sample_question(sample, dataset_path)
        self.answers = get_sample_answers(
            sample, dataset_path
        )
        self.data = generated
        self.prediction: str = ""

    @time_performance_decorator(enable=True)
    def evaluate(
        self,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        method: str,
        nli_model: DebertaV2ForSequenceClassification,
        nli_tokenizer: DebertaV2Tokenizer,
        embd_model: RobertaModel,
        embd_tokenizer: RobertaTokenizer,
        eigen_score_layers: list[int],
        hidden_score_layers: list[int],
        attn_score_layers: list[int],
        sharpness_layers: list[int],
        sharpness_alpha=1,
        score: dict | None = None
    ):

        if score is None:
            score = {}

        if "match" not in score:
            score["match"] = metric_max_over_ground_truths(
                calcu_exact_match_score, self.data["answer"], self.answers
            )
        if "rouge_l_f1" not in score:
            rouge_l = calcu_max_rouge_l_over_ground_truths(
                self.data["answer"], self.answers, tokenizer
            )
            score["rouge_l_f1"] = rouge_l["f1"]
        if "similarity" not in score:
            score["similarity"] = calcu_sentences_sims(
                self.data["answer"], self.answers, self.question,
                embd_model, embd_tokenizer
            )

        score = detect(
            self.data["question"],
            self.data["thought"],
            self.data["answer"],
            self.data["eigen_score"],
            self.data["self_check"],
            llm, tokenizer,
            method,
            nli_model, nli_tokenizer,
            eigen_score_layers,
            hidden_score_layers, attn_score_layers,
            sharpness_layers,
            build_msg_for_directly_answer,
            build_msg_for_thought,
            build_msg_for_answer,
            sharpness_alpha,
            score,
        )

        torch.cuda.empty_cache()
        return score


def insert_one_evaled(record: tuple, db: sqlite3.Connection, table: str):
    cur = db.cursor()
    cur.execute(
        f"insert into {table} (id, data) values (?, ?)", record
    )
    db.commit()


def update_one_evaled(record: tuple, db: sqlite3.Connection, table: str):
    cur = db.cursor()
    cur.execute(
        f"update {table} set data = ? where id = ?", record
    )
    db.commit()


def get_evaled_data_by_id(id: int, db: sqlite3.Connection, table: str):
    cur = db.cursor()
    cur.execute(
        f"select * from {table} where id = ?", (id,)
    )
    one = cur.fetchone()
    return one[-1] if one else None


def init_database(db_path):
    db = sqlite3.connect(db_path)

    db.execute("""drop table if exists evaled""")
    db.execute(f"""
    create table if not exists evaled (
        id integer primary key,
        data text
    )
    """)

    db.commit()
    return db


def main(
    *,
    model_name_or_path: 'm' = "",  # type: ignore
    nli_name_or_path: 'nli' = "",  # type: ignore
    embd_name_or_path: 'embd' = "",  # type: ignore
    data_name_or_path: 'd' = "",  # type: ignore
    data_start_index: 'S' = 0,  # type: ignore
    data_read_length: 'L' = 0,  # type: ignore
    data_batch_size: 'b' = 128,  # type: ignore
    cot_method: "method" = "base",  # type: ignore
    gpu_mem_util: 'r' = 0.5,  # type: ignore
):

    table = "evaled"

    db = init_database(
        f"{model_name_or_path.split('/')[-1].split('-')[0]}." +
        f"{data_name_or_path.split('/')[-1]}." +
        f"{cot_method}." +
        f"{data_start_index}." +
        "sqlite3"
    )

    llm = LLM(
        model=model_name_or_path,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    nli_model, nli_tokenizer = load_nli_model_and_tokenizer(nli_name_or_path)
    embd_model, embd_tokenizer = load_embd_model_and_tokenizer(
        embd_name_or_path
    )
    dataset = load_hf_dataset(
        data_name_or_path,
        data_start_index, data_read_length,
    )

    # detection hyper-parameters
    sharpness_alpha = 1
    eigen_score_layer = 17  # num_model_layers // 2 + 1
    hidden_score_layer = 15
    attn_score_layer = 23
    shapness_layer = 26

    for i, batch in enumerate(dataset.iter(data_batch_size)):
        # generation
        print(
            f"## Batch {i} / Samples {i*data_batch_size} / Total {len(dataset)}"
        )
        batch = [dict(zip(batch.keys(), t)) for t in zip(*batch.values())]

        thoughts, msgs_for_a, answers = generate_samples(
            batch, cot_method, llm, tokenizer, data_name_or_path
        )
        if "r1" in llm.llm_engine.model_config.served_model_name.lower():
            eigen_score_samples, self_check_samples = stochastically_generate_samples_r1(
                msgs_for_a, llm, tokenizer
            )
        else:
            eigen_score_samples, self_check_samples = stochastically_generate_samples(
                msgs_for_a, llm, tokenizer
            )
        # evaluation
        for j, sample in enumerate(batch):
            _id = i*data_batch_size + j + data_start_index
            sample = QaSample(
                sample,
                {
                    "question": get_sample_question(sample, data_name_or_path),
                    "thought": thoughts[j] if thoughts else "",
                    "answer": answers[j],
                    "eigen_score": eigen_score_samples[j],
                    "self_check": self_check_samples[j],
                },
                data_name_or_path
            )
            evaled = get_evaled_data_by_id(_id, db, table)
            evaled = json.loads(evaled) if evaled else None
            score = sample.evaluate(
                model, tokenizer,
                cot_method,
                nli_model, nli_tokenizer,
                embd_model, embd_tokenizer,
                [eigen_score_layer],
                [hidden_score_layer],
                [attn_score_layer],
                [shapness_layer],
                sharpness_alpha,
                evaled
            )
            

            score["eigen_score"] = score["eigen_score"][0] if isinstance(
                score["eigen_score"], list) else score["eigen_score"]
            score["entropies_perplexity"] = score["entropies_perplexity"][0] if isinstance(
                score["entropies_perplexity"], list) else score["entropies_perplexity"]
            score["attn_score"] = score["attn_score"][0] if isinstance(
                score["attn_score"], list) else score["attn_score"]
            score["hidden_score"] = score["hidden_score"][0] if isinstance(
                score["hidden_score"], list) else score["hidden_score"]

            if evaled is None:
                insert_one_evaled((_id, json.dumps(score)), db, table)
            else:
                update_one_evaled((json.dumps(score), _id), db, table)

    db.close()


if __name__ == "__main__":
    run(main)
