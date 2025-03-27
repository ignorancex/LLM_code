import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from convlab2.nlg.generative_models.user_simulator_generative_model import (
    UserSimulatorGenerativeModel)


class FLANT5Model(UserSimulatorGenerativeModel):
    def __init__(self, model_id='google/flan-t5-xl'):
        super(FLANT5Model, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
