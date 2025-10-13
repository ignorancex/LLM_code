
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from utils import time_performance_decorator

# see: https://github.com/akkeshav/confidence_probability_alignment

VERBALIZED = """A language model was asked: {} 
The model's answer was: {}. 

a. Very Certain 
b. Fairly Certain 
c. Moderately Certain 
d. Somewhat Certain 
e. Not Certain 
f. Very Uncertain

Analyse its answer, How certain are you of the model's answer:"""

VERBALIZED_R1 = """<｜begin▁of▁sentence｜><｜User｜>A language model was asked: {}
The model's answer was: {}

a. Very Certain 
b. Fairly Certain 
c. Moderately Certain 
d. Somewhat Certain 
e. Not Certain 
f. Very Uncertain

Analyse its answer, how certain are you of the model's answer?<｜Assistant｜><think>

</think>
Answer:"""

OPTIONS = " a b c d e f"


@time_performance_decorator(enable=False)
@torch.no_grad()
def calcu_verbalized(
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
):
    if "deepseek" in llm.config.name_or_path.lower():
        prompt = VERBALIZED.format(question, answer)
    else:
        prompt = VERBALIZED_R1.format(question, answer)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(llm.device)

    # batch_size=1, seq_len=1, vocab_size
    logits = llm.forward(inputs).logits[-1][-1]
    option_ids = tokenizer.encode(OPTIONS, add_special_tokens=False)
    option_logits = logits[option_ids]
    max_prob_index = torch.argmax(option_logits).item()

    return 1 - max_prob_index / 5
