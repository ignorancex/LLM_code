
from time import perf_counter
from functools import wraps
from typing import Callable
import json

import debugpy
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    RobertaTokenizer,
    RobertaModel,
)


NLI_TOKENIZER: DebertaV2Tokenizer|None = None
NLI_MODEL: DebertaV2ForSequenceClassification = None


def load_nli_model_and_tokenizer(
    nli_model_path="potsawee/deberta-v3-large-mnli",
    device="auto",
) -> tuple[DebertaV2ForSequenceClassification, DebertaV2Tokenizer]:
    global NLI_TOKENIZER, NLI_MODEL
    if NLI_TOKENIZER is not None and NLI_MODEL is not None:
        return NLI_TOKENIZER, NLI_MODEL
    NLI_TOKENIZER = DebertaV2Tokenizer.from_pretrained(
        nli_model_path, device="cuda")
    NLI_MODEL = DebertaV2ForSequenceClassification.from_pretrained(
        nli_model_path, device_map=device
    )
    NLI_MODEL.eval()
    return NLI_MODEL, NLI_TOKENIZER


EMBD_TOKENIZER: RobertaTokenizer|None = None
EMBD_MODEL: RobertaModel = None


def load_embd_model_and_tokenizer(
    embd_model_path="sentence-transformers/nli-roberta-large",
    device="auto",
) -> tuple[RobertaModel, RobertaTokenizer]:
    global EMBD_TOKENIZER, EMBD_MODEL
    if EMBD_TOKENIZER is not None and EMBD_MODEL is not None:
        return EMBD_TOKENIZER, EMBD_MODEL
    EMBD_TOKENIZER = RobertaTokenizer.from_pretrained(
        embd_model_path, device="cuda")
    EMBD_MODEL = RobertaModel.from_pretrained(embd_model_path, device_map=device)
    EMBD_MODEL.eval()
    return EMBD_MODEL, EMBD_TOKENIZER


def run_debug_service(
    debug: bool,
    port=9501,
):
    if not debug:
        return

    np.seterr(all='raise')
    # 5678 is the default attach port in the VS Code debug configurations.
    # Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", port))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()


def time_performance_decorator(msg="", enable=False):

    def time_performance(func: Callable):

        @wraps(func)
        def warpped(*args, **kwargs):

            if not enable:
                return func(*args, **kwargs)

            print(f"{func.__name__}: Started.")
            start = perf_counter()
            result = func(*args, **kwargs)
            end = perf_counter()
            _ = print(f"{func.__name__}: {msg}") if msg else None
            print(f"{func.__name__}: time consumed {end - start:.1f}s.")
            return result

        return warpped

    return time_performance


class TimePerformanceContext:

    def __init__(self, msg="TimePerformanceContext", enable=False) -> None:
        self.enable = enable
        self.msg = msg

    def __enter__(self):
        if not self.enable:
            return self
        print(f"{self.msg}:")
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enable:
            return
        self.end = perf_counter()
        print(f"time consumed: {self.end - self.start:.1f}s.")

    def inspect(self, **kwargs):
        if not self.enable:
            return
        print(json.dumps(
            kwargs, ensure_ascii=False, indent=1
        ))


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_path,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    torch.set_default_device(model.device)
    torch.set_default_dtype(model.dtype)

    model.eval()

    return model, tokenizer
