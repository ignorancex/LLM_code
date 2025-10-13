from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from ..omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM
import pdb

class OmniSpeechConfig(Qwen2Config):
    model_type = "omni_speech_qwen"

class OmniSpeechQwen2Model(OmniSpeechMetaModel, Qwen2Model):
    config_class = OmniSpeechConfig

    def __init__(self, config: Qwen2Config):
        super(OmniSpeechQwen2Model, self).__init__(config)
        
class OmniSpeechQwen2ForCausalLM(Qwen2ForCausalLM, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechConfig

    def __init__(self, config: Qwen2Config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = OmniSpeechQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )

        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # pdb.set_trace()
        return result

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        result = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        # pdb.set_trace()
        return result

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs
    def _post_decode(self, output, temperature=1.0, top_k=0, top_p=0.0):
        """
        Decoding function, based on the posterior probability output, 
        uses top_k, top_p, and temperature parameters for sampling.

        Parameters:
        - output: torch.Tensor, shaped as (1, 1, D), represents the posterior probability output by the model.
        - top_k: int, indicates selecting the top k tokens with the highest probability for sampling.
                      If 0, no top_k filtering is performed.
        - top_p: float, indicates selecting tokens with cumulative probability not exceeding p for sampling.
                        If 0.0, no top_p filtering is performed.
        - temperature: float, represents the sampling temperature parameter. 
                              The higher the value, the more random the sampling; 
                            the lower the value, the more deterministic the sampling.

        Returns:
        - Selected token index.
        """
        output = output.squeeze(0).squeeze(0)

        # temperature
        if temperature != 1.0:
            output = output / temperature

        probs = torch.nn.functional.softmax(output, dim=-1)

        # top_k
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()

        # top_p
        if top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove[0]:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        token_index = torch.multinomial(probs, 1)
        return token_index.unsqueeze(0)
    
    def _generate_one_step(
        self,
        inputs_embeds: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.1,
        **kwargs
    ) -> Tuple[torch.LongTensor, List[torch.FloatTensor]]:
        """
        Generates the model's next output based on the current input and state.

        Parameters:
        - inputs: The input tensor containing the model's input data.
        - stat: The current state information used to control the generation process.
        - top_p: The threshold for controlling top-p sampling.
        - top_k: The threshold for controlling top-k sampling.
        - temperature: Controls the randomness of sampling.

        Returns:
        - last_id: The index of the last generated token.
        - stat: The updated state information.
        - past_key_values: The model's historical key-value pairs, used for cross-step memory.
        - hidden_state: The model's hidden state, used to maintain cross-step contextual information.
        """
        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )
        last_hidden_state = outputs['hidden_states'][-1]
        last_logit = self.lm_head(last_hidden_state[:, -1:, :])
        last_id = self._post_decode(last_logit, temperature=temperature, top_k=top_k, top_p=top_p)
        return_tts_state = last_hidden_state[:, -1:, :]
        return last_id, outputs['past_key_values'], return_tts_state


AutoConfig.register("omni_speech_qwen", OmniSpeechConfig)
AutoModelForCausalLM.register(OmniSpeechConfig, OmniSpeechQwen2ForCausalLM)
