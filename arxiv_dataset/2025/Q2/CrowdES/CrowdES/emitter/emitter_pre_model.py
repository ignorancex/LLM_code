from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from transformers.models.segformer.modeling_segformer import SegformerPreTrainedModel, SegformerModel, SegformerDecodeHead
from transformers.utils import ModelOutput
from dataclasses import dataclass

from CrowdES.emitter.emitter_pre_config import CrowdESEmitterPreConfig


@dataclass
class CrowdESEmitterPreModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_unit: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class CrowdESEmitterPreModel(SegformerPreTrainedModel):

    config_class = CrowdESEmitterPreConfig
    base_model_prefix = 'crowdes_emitter_pre'
    main_input_name = 'pixel_values'

    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_unit_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        labels_unit: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CrowdESEmitterPreModelOutput]:  # SemanticSegmenterOutput
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> logits_unit = outputs.logits_unit  # shape (batch_size, num_unit_labels)
        >>> list(logits.shape)
        [1, 150, 128, 128]
        >>> list(logits_unit.shape)
        [1, 64]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f'Number of labels should be >=0: {self.config.num_labels}')

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # Segmentation Head
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        logits = self.decode_head(encoder_hidden_states)

        # Classification Head
        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]
        
        sequence_output = sequence_output.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])
        sequence_output = sequence_output.mean(dim=1)
        logits_unit = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
            )
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(upsampled_logits, labels.float())
            
        if labels_unit is not None:
            loss_fct = BCEWithLogitsLoss()
            loss_unit = loss_fct(logits_unit, labels_unit)
            loss = loss + loss_unit if loss is not None else loss_unit

        if not return_dict:
            if output_hidden_states:
                output = (logits, logits_unit) + outputs[1:]
            else:
                output = (logits, logits_unit) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrowdESEmitterPreModelOutput(
            loss=loss,
            logits=logits,
            logits_unit=logits_unit,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
