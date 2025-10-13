
from typing import Literal
import gc

import torch
from transformers import PreTrainedTokenizerBase
from transformers import PreTrainedModel
from transformers import DebertaV2ForSequenceClassification
from transformers import DebertaV2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from detect_methods import *
from utils import time_performance_decorator


def split_prompt_to_ids(
    msg: list[dict[str, str]],
    llm: PreTrainedModel | str,
    tokenizer: PreTrainedTokenizerBase,
):
    prompt_ids = tokenizer.apply_chat_template(msg)

    if "r1" in llm.config.name_or_path.lower():  # must before llama
        I = 128014
        prompt_str = tokenizer.apply_chat_template(msg, tokenize=False)
        prompt_str = (
            prompt_str
            .replace("<--think-->", "<think>")
            .replace("<--/think-->", "</think>")
        )
        prompt_ids = tokenizer.encode(prompt_str)
    elif "llama" in llm.config.name_or_path.lower():
        I = 128007
    elif "mistral" in llm.config.name_or_path.lower():
        I = 4

    i = next((
        i for i in range(-1, -len(prompt_ids)-1, -1) if prompt_ids[i] == I
    ), -len(prompt_ids)-1)

    # p_ids = torch.tensor(prompt_ids[:i+1], dtype=torch.int64)
    a_ids = torch.tensor(prompt_ids[i+1:], dtype=torch.int64)
    return torch.tensor(prompt_ids, dtype=torch.int64), a_ids


class LlmConditionalOutput(CausalLMOutputWithPast):

    answer_token_ids: torch.Tensor

    def set_answer_token_ids(self, answer_token_ids):
        self.answer_token_ids = answer_token_ids.to(self.logits.device)

    @property
    def answer_token_ids_list(self):
        return [
            int(token.item()) for token in
            self.answer_token_ids.squeeze(0)
        ]

    @property
    @torch.no_grad()
    def answer_start_index(self):
        return -self.answer_token_ids.shape[1] - 1

    @property
    @torch.no_grad()
    def answer_logits(self):
        return self.logits[
            :, self.answer_start_index:-1, :
        ]

    @property
    def question_with_thought_logits(self):
        return self.logits[
            :, :self.answer_start_index, :
        ]

    @property
    def question_with_thought_hidden_states(self):
        return tuple([
            states[:, :self.answer_start_index, :]
            for states in self.hidden_states
        ])

    @property
    def answer_hidden_states(self):
        return tuple([
            states[:, self.answer_start_index:-1, :]
            for states in self.hidden_states
        ])

    @property
    def answer_attentions(self):
        return tuple([
            attention[:, self.answer_start_index:-1, :]
            for attention in self.attentions
        ])

    @property
    @torch.no_grad()
    def answer_token_probs(self):
        a_probs = torch.softmax(self.logits[
            :, self.answer_start_index:-1, :
        ], dim=-1)  # scores for each vocabulary token before SoftMax
        a_token_probs = a_probs.gather(
            dim=2,
            index=self.answer_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        return a_token_probs

    @property
    @torch.no_grad()
    def answer_liklihood(self):
        a_log_probs = torch.log_softmax(self.logits[
            :, self.answer_start_index:-1, :
        ], dim=-1)  # scores for each vocabulary token before SoftMax
        a_token_log_probs = a_log_probs.gather(
            dim=2,
            index=self.answer_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        return a_token_log_probs.sum().item()


@time_performance_decorator(enable=False)
@torch.no_grad()
def get_tokens_llm_state(
    all_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    llm: PreTrainedModel,
) -> LlmConditionalOutput:
    all_ids = all_ids.to(llm.device).unsqueeze(0)
    answer_ids = answer_ids.to(llm.device).unsqueeze(0)
    output: LlmConditionalOutput = llm.forward(
        all_ids,
        return_dict=True,
        output_hidden_states=True,
        output_attentions=True,
    )
    output.__class__ = LlmConditionalOutput
    output.set_answer_token_ids(answer_ids)

    return output


def build_msg_fot_base_detect(
    question: str, answer: str, build_msg_for_directly_answer
):
    msg = build_msg_for_directly_answer(question)
    msg.append(
        {"role": "assistant", "content": answer}
    )
    return msg


def build_msg_for_r1_detect(
    question: str, thought: str, answer: str,
):
    prompt = "Think about it step by step, then answer the question: {}"
    msg = [
        {
            "role": "user",
            "content": prompt.format(question),
        },
        {
            "role": "assistant",
            "content": "<--think-->\n" + thought + "\n<--/think-->\n\n" + answer
        }
    ]
    return msg


def build_msg_for_r1_detect_ltm(
    question: str, thought: str, answer: str,
):
    prompt = "Think about the question with subproblems must be solved before answering and answer it directly: {}"
    msg = [
        {
            "role": "user",
            "content": prompt.format(question),
        },
        {
            "role": "assistant",
            "content": "<--think-->\n" + thought + "\n<--/think-->\n\n" + answer
        }
    ]
    return msg


def build_msg_for_r1_detect_mrpp(
    question: str, thought: str, answer: str,
):
    prompt = "Think about the question with each step carrying out as many basic operations as possible and answer it directly: {}"
    msg = [
        {
            "role": "user",
            "content": prompt.format(question),
        },
        {
            "role": "assistant",
            "content": "<--think-->\n" + thought + "\n<--/think-->\n\n" + answer
        }
    ]
    return msg


def build_msg_for_cot_detect(
    question: str, thought: str, answer: str,
    method: Literal["sbs", "mrpp", "ltm"],
    build_msg_for_thought, build_msg_for_answer
):
    msg = build_msg_for_thought(
        question, method
    )
    msg = build_msg_for_answer(
        msg, thought, method
    )
    msg.append(
        {"role": "assistant", "content": answer}
    )
    return msg


def detect(
    question: str,
    thought: str,
    answer: str,
    eigen_score_answers: list[str],
    self_check_answers: list[str],
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    method: str,
    nli_model: DebertaV2ForSequenceClassification,
    nli_tokenizer: DebertaV2Tokenizer,
    eigen_score_layers: list[int],
    hidden_score_layers: list[int],
    attn_score_layers: list[int],
    sharpness_layers: list[int],
    build_msg_for_directly_answer,
    build_msg_for_thought,
    build_msg_for_answer,
    sharpness_alpha=1,
    score: dict | None = None
):

    if "verbalized" not in score:
        score["verbalized"] = calcu_verbalized(
            llm, tokenizer, question, answer,
        )
    if "eigen_score" not in score:
        score["eigen_score"] = calcu_layers_eigen_scores(
            llm, tokenizer, eigen_score_answers, eigen_score_layers,
        )

    if "self_check_nli" not in score:
        score["self_check_nli"], _ = calcu_self_check_nli_and_prompt(
            llm, tokenizer,
            nli_model, nli_tokenizer,
            self_check_answers,
            answer,
            "self_check_nli",
        )
    if "self_check_prompt" not in score:
        _, score["self_check_prompt"] = calcu_self_check_nli_and_prompt(
            llm, tokenizer,
            nli_model, nli_tokenizer,
            self_check_answers,
            answer,
            "self_check_prompt",
        )

    need_stat_keys = [
        "perplexity", "likelihood",
        "entropies_avg", "entropies_perplexity",
        "attn_score", "hidden_score",
    ]
    state: LlmConditionalOutput | None = None

    if any((key not in score) for key in need_stat_keys):

        if  method == "base" and "r1" in llm.config.name_or_path.lower():
            all_ids, a_ids = split_prompt_to_ids(
                build_msg_for_r1_detect(question, thought, answer),
                llm, tokenizer
            )
        elif method == "ltm" and "r1" in llm.config.name_or_path.lower():
            all_ids, a_ids = split_prompt_to_ids(
                build_msg_for_r1_detect_ltm(question, thought, answer),
                llm, tokenizer
            )
        elif method == "mrpp" and "r1" in llm.config.name_or_path.lower():
            all_ids, a_ids = split_prompt_to_ids(
                build_msg_for_r1_detect_mrpp(question, thought, answer),
                llm, tokenizer
            )
        elif method == "base":
            all_ids, a_ids = split_prompt_to_ids(
                build_msg_fot_base_detect(
                    question, answer, build_msg_for_directly_answer
                ),
                llm, tokenizer,
            )
        else:  # sbs, mrpp, ltm
            all_ids, a_ids = split_prompt_to_ids(
                build_msg_for_cot_detect(
                    question, thought, answer, method,
                    build_msg_for_thought,
                    build_msg_for_answer,
                ),
                llm, tokenizer,
            )
        state = get_tokens_llm_state(all_ids, a_ids, llm)

    if "perplexity" not in score:
        score["perplexity"] = calcu_perplexity(state.answer_token_probs)
    if "likelihood" not in score:
        score["likelihood"] = state.answer_liklihood
    if any(
        (key not in score) for key in
        ("entropies_avg", "entropies_perplexity")
    ):
        layers = [
            state.question_with_thought_hidden_states[i]
            for i in sharpness_layers
        ]
        entropies_avg_pplx = [
            (layer_output.entropy_avg, layer_output.perplexity)
            for layer_output in calcu_sharpnesses_for_layers(
                layers,
                state.answer_token_ids,
                state.answer_logits,
                llm.lm_head,
                sharpness_alpha,
            )
        ]
        score["entropies_avg"] = [ap[0] for ap in entropies_avg_pplx]
        score["entropies_perplexity"] = [
            ap[1]for ap in entropies_avg_pplx
        ]
    if "attn_score" not in score:
        score["attn_score"] = calcu_layers_attn_scores(
            state.answer_attentions,
            attn_score_layers,
        )
    if "hidden_score" not in score:
        score["hidden_score"] = calcu_layers_hidden_scores(
            state.answer_hidden_states,
            hidden_score_layers,
        )

    if 'state' in locals():
        del state
        torch.cuda.empty_cache()
        gc.collect()
    return score
