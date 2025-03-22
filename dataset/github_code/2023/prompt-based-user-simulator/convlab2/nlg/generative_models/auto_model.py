import torch
# from parallelformers import parallelize
from transformers import AutoTokenizer, AutoModelForCausalLM
from convlab2.nlg.generative_models.user_simulator_generative_model import (
    UserSimulatorGenerativeModel)


class AutoModel(UserSimulatorGenerativeModel):
    def __init__(self, model_id='nomic-ai/gpt4all-j'):
        super(AutoModel, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        # parallelize(self.model, num_gpus=2, fp16=True, verbose='detail')
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
