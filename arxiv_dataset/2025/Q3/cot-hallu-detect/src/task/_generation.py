

from typing import Literal, Iterable

from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizerBase
import datasets

from utils import *


def replace_r1_think_tag_keeper(text: str):
    return text.replace(
        "<think>", "<--think-->"
    ).replace(
        "</think>", "<--/think-->"
    )


def build_msg_for_directly_answer(
    question: str,
):
    msg = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer the question:\n{question}"},
    ]
    return msg


def build_msg_for_directly_answer_r1(
    question: str,
):
    msg = [
        {"role": "user", "content": f"Answer the question:\n{question}"},
    ]
    return msg


@time_performance_decorator(enable=True)
def generate_directly_answer(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    param = SamplingParams(n=1, temperature=0.5, max_tokens=256)
    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    responses = llm.generate(
        prompts=prompts, sampling_params=param, use_tqdm=False
    )
    return [res.outputs[0].text for res in responses]


def generate_directly_answer_r1(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    param = SamplingParams(n=1, temperature=0.5, max_tokens=256)
    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    prompts = [
        p + "<think>\n\n</think>\n" for p in prompts
    ]
    responses = llm.generate(
        prompts=prompts, sampling_params=param, use_tqdm=False
    )
    return [res.outputs[0].text for res in responses]


def build_msg_for_thought(
    question: str,
    method: Literal["sbs", "mrpp", "ltm"],
):
    if method == "sbs":
        cot = "Think about it step by step. Tell me your thoughts, but don't tell me the answers now.\n"
        q = "Question:\n" + question
    elif method == "mrpp":
        cot = "You need to perform multi-step reasoning, with each step carrying out as many basic operations as possible. Tell me your reasoning, but don't tell me the answers now.\n"
        q = "Question:\n" + question
    elif method == "ltm":
        cot = "What subproblems must be solved before answering the inquiry? Tell me the subproblems and the answers to the subproblems, but don't tell me the answers to the inquiry now.\n"
        q = "Inquiry:\n" + question

    msg = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": cot + q},
    ]
    return msg


def build_msg_for_thought_r1(
    question: str,
    method: Literal["sbs", "mrpp", "ltm"],
):
    if method == "sbs":
        cot = "Think about it step by step, then answer the question: "
    elif method == "mrpp":
        cot = "Think about the question with subproblems must be solved before answering and answer it directly: "
    elif method == "ltm":
        cot = "Think about the question with each step carrying out as many basic operations as possible and answer it directly: "
    return [
        {"role": "user", "content": cot + question},
    ]


@time_performance_decorator(enable=True)
def generate_thought(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens=256,
):
    param = SamplingParams(n=1, temperature=0.5, max_tokens=max_tokens)
    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    responses = llm.generate(
        prompts=prompts, sampling_params=param, use_tqdm=False
    )
    return [res.outputs[0].text for res in responses]


def generate_thought_r1(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens=256,
):
    param = SamplingParams(n=1, temperature=0.5, max_tokens=max_tokens)
    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    prompts = [
        replace_r1_think_tag_keeper(x)
        for x in prompts
    ]
    responses = llm.generate(
        prompts=prompts, sampling_params=param, use_tqdm=False
    )
    return [res.outputs[0].text for res in responses]


def build_msg_for_answer(
    thoght_msg: list[dict[str, str]],
    thought: str,
    method: Literal["sbs", "mrpp", "ltm"],
):
    if method == "ltm":
        q = "inquiry"
    else:
        q = "question"
    msg = [
        *thoght_msg,
        {"role": "assistant", "content": thought},
        {"role": "user", "content": f"Now, answer the {q}."},
    ]
    return msg


def build_msg_for_answer_r1(
    thoght_msg: list[dict[str, str]],
    thought: str,
):
    msg = [
        *thoght_msg,
        {"role": "assistant", "content": "<--think-->\n" + thought + "\n<--/think-->"},
    ]
    return msg


@time_performance_decorator(enable=True)
def generate_answer(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
):
    param = SamplingParams(n=1, temperature=0.5, max_tokens=128)
    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    responses = llm.generate(
        prompts=prompts, sampling_params=param, use_tqdm=False)
    return [res.outputs[0].text for res in responses]


def generate_answer_r1(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
):
    param = SamplingParams(n=1, temperature=0.5, max_tokens=128)
    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    prompts = [
        replace_r1_think_tag_keeper(p)
        for p in prompts
    ]
    responses = llm.generate(
        prompts=prompts, sampling_params=param, use_tqdm=False
    )
    return [res.outputs[0].text for res in responses]


def post_norm_text(
    text: str,
    llm: LLM | str,
):
    llm_name = llm if isinstance(
        llm, str
    ) else llm.llm_engine.model_config.served_model_name

    if "r1" in llm_name.lower():
        text = replace_r1_think_tag_keeper(text)
    if "llama" in llm_name.lower():
        text = text.replace(
            "<|start_header_id|>assistant<|end_header_id|>", ""
        )
    return text.strip()


def generate_samples(
    batch: datasets.Dataset,
    method: str,
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    data_path: str,
):
    if "r1" in llm.llm_engine.model_config.served_model_name.lower():
        if method == "base":
            msgs_for_a = [
                build_msg_for_directly_answer_r1(
                    get_sample_question(sample, data_path)
                ) for sample in batch
            ]
            answers = [
                post_norm_text(text, llm) for text in
                generate_directly_answer_r1(msgs_for_a, llm, tokenizer)
            ]
            return [], msgs_for_a, answers
        else:
            msgs_for_t = [
                build_msg_for_thought_r1(
                    get_sample_question(sample, data_path), method
                ) for sample in batch
            ]
            thoughts = [
                post_norm_text(text, llm) for text in
                generate_thought_r1(msgs_for_t, llm, tokenizer)
            ]
            msgs_for_a = [
                build_msg_for_answer_r1(
                    msg, thought
                ) for msg, thought in zip(msgs_for_t, thoughts)
            ]
            answers = [
                post_norm_text(text, llm) for text in
                generate_answer_r1(msgs_for_a, llm, tokenizer)
            ]
            return thoughts, msgs_for_a, answers

    if method == "base":
        msgs_for_a = [
            build_msg_for_directly_answer(
                get_sample_question(sample, data_path)
            ) for sample in batch
        ]
        answers = [
            post_norm_text(text, llm) for text in
            generate_directly_answer(msgs_for_a, llm, tokenizer)
        ]
        thoughts = []
    else:
        msgs_for_t = [
            build_msg_for_thought(
                get_sample_question(sample, data_path), method
            ) for sample in batch
        ]
        thoughts = [
            post_norm_text(text, llm) for text in
            generate_thought(msgs_for_t, llm, tokenizer)
        ]
        msgs_for_a = [
            build_msg_for_answer(
                msg, thought, method
            ) for msg, thought in zip(msgs_for_t, thoughts)
        ]
        answers = [
            post_norm_text(text, llm) for text in
            generate_answer(msgs_for_a, llm, tokenizer)
        ]
    return thoughts, msgs_for_a, answers


@time_performance_decorator(enable=True)
def stochastically_generate_samples(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[list[str]], list[list[str]]]:

    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    es_param = SamplingParams(
        n=15,
        temperature=0.5,
        top_p=0.99,
        top_k=5,
        max_tokens=128
    )
    sc_param = SamplingParams(
        n=20,
        temperature=1,
        max_tokens=128
    )

    responses = llm.generate(
        prompts=prompts, sampling_params=es_param, use_tqdm=False
    )
    es_s = [
        [post_norm_text(o.text, llm) for o in response.outputs]
        for response in responses
    ]

    responses = llm.generate(
        prompts=prompts, sampling_params=sc_param, use_tqdm=False
    )
    sc_s = [
        [post_norm_text(o.text, llm) for o in response.outputs]
        for response in responses
    ]

    return es_s, sc_s


def stochastically_generate_samples_r1(
    msg_list: list[list[dict[str, str]]],
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[list[str]], list[list[str]]]:
    prompts = tokenizer.apply_chat_template(msg_list, tokenize=False)
    prompts = [
        replace_r1_think_tag_keeper(p)
        for p in prompts
    ]

    es_param = SamplingParams(
        n=15,
        temperature=0.5,
        top_p=0.99,
        top_k=5,
        max_tokens=128
    )
    sc_param = SamplingParams(
        n=20,
        temperature=1,
        max_tokens=128
    )

    responses = llm.generate(
        prompts=prompts, sampling_params=es_param, use_tqdm=False
    )
    es_s = [
        [post_norm_text(o.text, llm) for o in response.outputs]
        for response in responses
    ]

    responses = llm.generate(
        prompts=prompts, sampling_params=sc_param, use_tqdm=False
    )
    sc_s = [
        [post_norm_text(o.text, llm) for o in response.outputs]
        for response in responses
    ]

    return es_s, sc_s


def load_hf_dataset(
    data_path: str,
    data_range: Iterable[int] | None = None,
    start_index=0,
    data_cnt=0,
) -> datasets.Dataset:

    data_kwargs = {
        "path": data_path,
        "name": "default",
        "split": "validation",
    }
    if "truthful_qa" in data_path:
        data_kwargs["name"] = "generation"
    elif "cnn_dailymail" in data_path:
        data_kwargs["name"] = "3.0.0"
        data_kwargs["split"] = "test"
    elif "trivia_qa" in data_path:
        data_kwargs["name"] = "rc.wikipedia.nocontext"
    elif "PopQA" in data_path:
        data_kwargs["split"] = "test"
    elif "HaluEval" in data_path:
        data_kwargs["name"] = "qa"
        data_kwargs["split"] = "data"
    elif "mmlu" in data_path:
        data_kwargs["split"] = "test"
        data_kwargs["name"] = "all"
    elif "ai2_arc" in data_path:
        data_kwargs["name"] = "ARC-Challenge"
        data_kwargs["split"] = "test"
    elif "commonsense_qa" in data_path:
        pass
    else:
        raise ValueError("Unknown dataset path")

    data = datasets.load_dataset(
        **data_kwargs,
        trust_remote_code=True,
    )

    end_index = min(
        start_index + data_cnt, len(data)
    ) if data_cnt else len(data)
    data_range = range(start_index, end_index)

    print("confirmed data_range:")
    print(data_range)
    data = data.select(data_range)

    return data


def get_sample_question(sample: dict, data_path: str) -> str:
    if any(
        (ds in data_path) for ds in
        """truthful_qa trivia_qa PopQA HaluEval""".split()
    ):
        return sample["question"]
    else:
        raise ValueError("Unknown dataset path")


def get_sample_answers(sample: dict, dataset_path: str) -> list[str]:
    if "truthful_qa" in dataset_path:
        answers = sample["correct_answers"]
    elif "trivia_qa" in dataset_path:
        answers = sample["answer"]["aliases"]
    elif "HaluEval" in dataset_path:
        answers = [sample["right_answer"]]
    elif "PopQA" in dataset_path:
        answers = eval(sample["possible_answers"])
    else:
        raise ValueError("Unknown dataset path")

    return list(set(answers))
