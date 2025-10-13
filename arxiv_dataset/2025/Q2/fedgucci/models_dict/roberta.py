from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch import nn


class RobertaBase(nn.Module):
    def __init__(self, model_path: str, num_labels: int, lora_r: int, lora_alpha: float, lora_target_modules: list = ["query", "value"]):
        super(RobertaBase, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, return_dict=True, num_labels=num_labels)
        task_type = "SEQ_CLS"
        config = LoraConfig(
            task_type=task_type ,inference_mode=False, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="all", target_modules=lora_target_modules
        )
        self.model = get_peft_model(self.model, config)
        # print(self.model)
        self.model.print_trainable_parameters()
        self.classifier = self.model.classifier
    
    def forward(self, inputs, return_hidden_state: bool = False):
        out = self.model(input_ids=inputs[0], attention_mask=inputs[1], output_hidden_states=return_hidden_state)
        if return_hidden_state:
            return None, out.logits, out.hidden_states[0]
        else:
            return out.logits