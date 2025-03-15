from transformers import PreTrainedModel, AutoModel, PretrainedConfig
import torch
import torch.nn as nn
from .loss import FocalLoss

class ClaimModelConfig(PretrainedConfig):
    model_type = "claim_verification"

    def __init__(
        self,
        model_name=None,
        num_labels=3,
        dropout=0.3,
        loss_type='ce',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout = dropout
        self.loss_type = loss_type

class ClaimModelForClassification(PreTrainedModel):
    config_class = ClaimModelConfig
    base_model_prefix = "claim_verification"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, config.num_labels)

        if self.config.loss_type == 'focal':
            self.loss_fn = FocalLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def forward(self, 
                input_ids, 
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None, 
                labels=None,
                **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        cls_output = outputs[1]  
        x = self.dropout(cls_output)
        logits = self.fc(x)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}  
        
        return {"logits": logits}
