from transformers import PreTrainedModel, AutoModel, PretrainedConfig, RobertaModel, XLMRobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch
import torch.nn.functional as F
import torch.nn as nn
from .loss import comboLoss  

class QATCConfig(PretrainedConfig):
    model_type = "qatc"

    def __init__(
        self, 
        model_name=None,
        freeze_text_encoder=False,
        alpha=1.0,
        beta=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.freeze_text_encoder = freeze_text_encoder
        self.alpha = alpha  
        self.beta = beta    

class Rational_Tagging(nn.Module):
    def __init__(self, hidden_size):
        super(Rational_Tagging, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        h_1 = self.W1(h_t)
        h_1 = F.relu(h_1)
        p = self.w2(h_1)
        p = torch.sigmoid(p)  
        return p

class QATCForQuestionAnswering(PreTrainedModel):
    config_class = QATCConfig
    base_model_prefix = "qa_model"

    def __init__(self, config):
        super(QATCForQuestionAnswering, self).__init__(config)
        self.config = config
        if "deberta" in self.config.model_name:
            self.model = AutoModel.from_pretrained(self.config.model_name) 
        elif "info" in self.config.model_name:
            self.model = XLMRobertaModel.from_pretrained(self.config.model_name)
        else:
            self.model = RobertaModel.from_pretrained(self.config.model_name)

        if getattr(self.config, "freeze_text_encoder", False):
            print("Freezing text encoder weights")
            for param in self.model.parameters():
                param.requires_grad = False

        self.qa_outputs = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.model.config.hidden_size, 2),
        )

        self.tagging = Rational_Tagging(self.model.config.hidden_size)
        self.loss_fn = comboLoss(config)
        self.init_weights()
        self.model.pooler = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        tagging_labels=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        sequence_output = outputs[0]  

        rational_tag_logits = self.tagging(sequence_output)  

        logits = self.qa_outputs(sequence_output)  
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss_inputs = {
            "attention_mask": attention_mask,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "Tagging": tagging_labels,
            "pt": rational_tag_logits
        }

        if start_logits == None and end_logits == None and tagging_labels == None:
            total_loss = self.loss_fn(loss_inputs)
        else:
            total_loss = None

        if not return_dict:
            return (total_loss, start_logits, end_logits, rational_tag_logits) + outputs[2:]

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
