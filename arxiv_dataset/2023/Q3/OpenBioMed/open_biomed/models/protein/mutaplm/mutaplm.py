import contextlib
from typing import Any, Dict, List, Optional, Tuple

import logging
import torch
import torch.nn as nn

from huggingface_hub import snapshot_download
import os
from transformers import PreTrainedTokenizer, LlamaTokenizer, LlamaConfig, LlamaForCausalLM, EsmTokenizer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType

from open_biomed.data import Protein, Text
from open_biomed.models.task_models.mutation_text_translation import MutationExplanationModel, MutationEngineeringModel
from open_biomed.models.protein.mutaplm.modeling_esm import EsmForMutationDesign
from open_biomed.utils.collator import Collator, ClassLabelCollator, EnsembleCollator, ListCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class MutaPLMExplanationFeaturizer(Featurizer):
    def __init__(self, 
        protein_tokenizer: PreTrainedTokenizer,
        text_tokenizer: PreTrainedTokenizer,
        max_length_func: int=512,
        max_length_protein: int=1024,
        max_length_label: int=256,
        stage2_prompt: str="",
    ) -> None:
        super().__init__()
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length_func = max_length_func
        self.max_length_protein = max_length_protein
        self.max_length_label = max_length_label
        self.stage2_prompt = stage2_prompt

    def __call__(self, protein: Protein, mutation: str, label: Optional[Text]=None, function: Optional[Text]=None) -> Dict[str, Any]:
        pos = int(mutation[1:-1])
        wild_type = protein.sequence
        mutant = wild_type[:pos - 1] + mutation[-1] + wild_type[pos:]
        featurized = {}
        featurized["wild_type"] = self.protein_tokenizer(
            wild_type,
            max_length=self.max_length_protein,
            truncation=True,
            add_special_tokens=True,
        )
        featurized["mutant"] = self.protein_tokenizer(
            mutant,
            max_length=self.max_length_protein,
            truncation=True,
            add_special_tokens=True,
        )
        featurized["stage2_prompt"] = self.text_tokenizer(
            self.stage2_prompt.format(mutation[0], mutation[-1], mutation[1:-1]),
            max_length=self.max_length_label,
            truncation=True,
            add_special_tokens=False,
        )
        if label is not None:
            featurized["label"] = self.text_tokenizer(
                label.str,
                max_length=self.max_length_label,
                truncation=True,
                add_special_tokens=False,
            )
        if function is not None:
            featurized["gt_function"] = self.text_tokenizer(
                function.str,
                max_length=self.max_length_func,
                truncation=True,
                add_special_tokens=False,
            )
        
        return featurized
    
    def get_attrs(self) -> List[str]:
        return ["wild_type", "mutant", "stage2_prompt", "label", "gt_function"]

class MutaPLMEngineeringFeaturizer(Featurizer):
    def __init__(self,
        protein_tokenizer: PreTrainedTokenizer,
        text_tokenizer: PreTrainedTokenizer,
        max_length_func: int=512,
        max_length_protein: int=1024,
        max_length_prompt: int=256,
    ) -> None:
        super().__init__()
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length_func = max_length_func
        self.max_length_protein = max_length_protein
        self.max_length_prompt = max_length_prompt

    def __call__(self, protein: Protein, text: Text, label: Optional[str]=None, function: Optional[Text]=None) -> Dict[str, Any]:
        featurized = {}
        featurized["wild_type"] = self.protein_tokenizer(
            protein.sequence,
            max_length=self.max_length_protein,
            truncation=True,
            add_special_tokens=True,
        )
        featurized["prompt"] = self.text_tokenizer(
            text.str,
            max_length=self.max_length_prompt,
            truncation=True,
            add_special_tokens=False,
        )
        if label is not None:
            featurized["label"] = {
                "pos": torch.LongTensor(int(label[1:-1]) - 1),
                "aa": self.protein_tokenizer(label[-1], return_tensors='pt').input_ids,
            }
        if function is not None:
            featurized["gt_function"] = self.text_tokenizer(
                function.str,
                max_length=self.max_length_func,
                truncation=True,
                add_special_tokens=False,
            )
        
        return featurized

    def get_attrs(self) -> List[str]:
        return ["wild_type", "prompt", "label", "gt_function"]

class MutaPLM(MutationExplanationModel, MutationEngineeringModel):
    def __init__(self, model_cfg: Config) -> None:
        super(MutaPLM, self).__init__(model_cfg)
        
        # load esm
        logging.info("*** loading protein model...")
        if not os.path.exists(model_cfg.protein_model):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            snapshot_download(repo_id="facebook/esm2_t33_650M_UR50D", local_dir=model_cfg.hf_model_name_or_path, force_download=True)
        self.protein_model = EsmForMutationDesign.from_pretrained(model_cfg.protein_model, torch_dtype=torch.bfloat16) # delta decoder is here
        self.protein_tokenizer = EsmTokenizer.from_pretrained(model_cfg.protein_model)

        # load llm
        logging.info("*** loading llm tokenizer...")
        if not os.path.exists(model_cfg.protein_model):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            snapshot_download(repo_id="PharMolix/BioMedGPT-LM-7B", local_dir=model_cfg.hf_model_name_or_path, force_download=True)
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(model_cfg.llama_ckpt, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        logging.info(f"*** loading llm from {model_cfg.llama_ckpt}...")
        llm_cfg = LlamaConfig.from_pretrained(model_cfg.llama_ckpt)
        self.llm = LlamaForCausalLM(llm_cfg)
        self.llm.resize_token_embeddings(len(self.llm_tokenizer))
        
        # add lora
        logging.info("*** adding LoRA...")
        lora_config = LoraConfig(
            peft_type=TaskType.CAUSAL_LM, 
            inference_mode=True, 
            r=16, lora_alpha=16, 
            lora_dropout=0.05, 
            target_modules=["v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        
        # delta encoder and decoder
        logging.info("*** building delta network...")
        self.query_protein1 = nn.Parameter(
            torch.zeros(1, model_cfg.num_query_tokens_protein1, self.protein_model.config.hidden_size)
        )
        nn.init.normal_(self.query_protein1, 0, 0.02)
        self.query_protein2 = nn.Parameter(
            torch.zeros(1, model_cfg.num_query_tokens_protein2, self.protein_model.config.hidden_size)
        )
        nn.init.normal_(self.query_protein2, 0, 0.02)
        self.pooler_protein1 = nn.MultiheadAttention(
            embed_dim=self.protein_model.config.hidden_size,
            num_heads=model_cfg.ca_num_head,
            batch_first=True
        )
        self.pooler_protein2 = nn.MultiheadAttention(
            embed_dim=self.protein_model.config.hidden_size,
            num_heads=model_cfg.ca_num_head,
            batch_first=True
        )
        
        self.bop_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.eop_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.bom_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.eom_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.soft_tokens = nn.Parameter(torch.zeros(1, model_cfg.num_query_tokens_protein2, self.llm.config.hidden_size))
        nn.init.normal_(self.bop_embeds, 0, 0.02)
        nn.init.normal_(self.eop_embeds, 0, 0.02)
        nn.init.normal_(self.bom_embeds, 0, 0.02)
        nn.init.normal_(self.eom_embeds, 0, 0.02)
        nn.init.normal_(self.soft_tokens, 0, 0.02)
        
        # build proj
        self.proj_protein1 = nn.Linear(self.protein_model.config.hidden_size, self.llm.config.hidden_size)
        self.proj_protein2 = nn.Linear(self.protein_model.config.hidden_size, self.llm.config.hidden_size)
        self.proj_text = nn.Linear(self.llm.config.hidden_size, self.protein_model.config.hidden_size)
        
        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def load_ckpt(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict["model"])

    def featurizer_mutation_explanation(self) -> Tuple[Featurizer, Collator]:
        return MutaPLMExplanationFeaturizer(
            self.protein_tokenizer,
            self.llm_tokenizer,
            self.config.func_maxlen,
            self.config.protein_maxlen,
            self.config.text_maxlen,
            self.config.stage2_prompt,
        ), EnsembleCollator({
            "wild_type": DataCollatorWithPadding(self.protein_tokenizer, padding=True),
            "mutant": DataCollatorWithPadding(self.protein_tokenizer, padding=True),
            "label": ListCollator(),
            "gt_function": DataCollatorWithPadding(self.llm_tokenizer, padding=False), 
            "stage2_prompt": DataCollatorWithPadding(self.llm_tokenizer, padding=False),
        })
        
    def featurizer_mutation_engineering(self) -> Tuple[Featurizer, Collator]:
        return MutaPLMEngineeringFeaturizer(
            self.protein_tokenizer,
            self.llm_tokenizer,
            self.config.func_maxlen,
            self.config.protein_maxlen,
            self.config.text_maxlen,
        ), EnsembleCollator({
            "wild_type": DataCollatorWithPadding(self.protein_tokenizer, padding=True),
            "prompt": DataCollatorWithPadding(self.llm_tokenizer, padding=False),
            "label": EnsembleCollator({
                "pos": ClassLabelCollator(),
                "aa": ClassLabelCollator(),
            }),
            "gt_function": DataCollatorWithPadding(self.llm_tokenizer, padding=False),
        })

    def maybe_autocast(self, device="cuda:0", dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def _encode_protein(self, protein1: Featurized[Protein], protein2: Optional[Featurized[Protein]]):
        batch_size = protein1.input_ids.shape[0]
        device = protein1.input_ids.device

        with self.maybe_autocast(device):
            p_feature1_in = self.protein_model.esm(**protein1)     # last_hidden_states: [bs, prot_len, esm_hidden_size]
            query_protein1 = self.query_protein1.expand(batch_size, -1, -1)
            attn_mask_1 = (1 - protein1.attention_mask.repeat(self.config.ca_num_head, 1).unsqueeze(1).expand(-1, self.config.num_query_tokens_protein1, -1)).to(bool)
            p_feature1 = self.pooler_protein1(
                query_protein1,
                p_feature1_in[0],
                p_feature1_in[0],
                attn_mask = attn_mask_1
            )
            protein1_embeds = self.proj_protein1(p_feature1[0])

            if protein2 is not None:
                p_feature2_in = self.protein_model.esm(**protein2)
                query_protein2 = self.query_protein2.expand(batch_size, -1, -1)
                attn_mask_2 = (1 - protein2.attention_mask.repeat(self.config.ca_num_head, 1).unsqueeze(1).expand(-1, self.config.num_query_tokens_protein2, -1)).to(bool)
                delta_feature = p_feature2_in[0] - p_feature1_in[0]
                p_feature2 = self.pooler_protein2(
                    query_protein2,
                    delta_feature,
                    delta_feature,
                    attn_mask = attn_mask_2
                )
                protein2_embeds = self.proj_protein2(p_feature2[0])
        
        if protein2 is not None:
            return protein1_embeds, protein2_embeds
        else:
            return protein1_embeds


    def add_padding(self, wrapped_embeds: torch.Tensor, wrapped_attention_mask: torch.Tensor=None, targets: torch.Tensor=None, regress_ids: torch.Tensor=None, padding: str="right"):
        assert (targets is None) or (regress_ids is None)
        batch_size = len(wrapped_embeds)
        max_length_batch = max([x.shape[1] for x in wrapped_embeds])
        for i in range(batch_size):
            pad_len = max_length_batch - wrapped_embeds[i].shape[1]
            if padding == "right":
                wrapped_embeds[i] = torch.cat((
                    wrapped_embeds[i], 
                    torch.zeros((1, pad_len, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype).to(wrapped_embeds[i].device)
                ), dim=1)
                if wrapped_attention_mask:
                    wrapped_attention_mask[i] = torch.cat((
                        wrapped_attention_mask[i],
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device)
                    ), dim=1)
                if targets:
                    targets[i] = torch.cat((
                        targets[i],
                        torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100)
                    ), dim=1)
                if regress_ids:
                    regress_ids[i] = torch.cat((
                        regress_ids[i],
                        torch.zeros((pad_len), dtype=regress_ids[i].dtype).to(regress_ids[i].device)
                    ), dim=0)
            else:
                wrapped_embeds[i] = torch.cat((
                    torch.zeros((1, pad_len, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype).to(wrapped_embeds[i].device),
                    wrapped_embeds[i], 
                ), dim=1)
                if wrapped_attention_mask:
                    wrapped_attention_mask[i] = torch.cat((
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device),
                        wrapped_attention_mask[i],
                    ), dim=1)
                if targets:
                    targets[i] = torch.cat((
                        torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100),
                        targets[i],
                    ), dim=1)
                if regress_ids:
                    regress_ids[i] = torch.cat((
                        torch.zeros((pad_len), dtype=regress_ids[i].dtype).to(regress_ids[i].device),
                        regress_ids[i]
                    ), dim=0)
        
        if targets:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0), torch.cat(targets, dim=0)
        if regress_ids:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0), torch.stack(regress_ids, dim=0)
        if wrapped_attention_mask is None:
            return torch.cat(wrapped_embeds, dim=0)
        else:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0)
        
    def _wrapped_sentence_inference(
        self, 
        protein1_embeds: torch.Tensor=None, 
        protein2_embeds: Optional[torch.Tensor]=None, 
        stage2_prompt: Featurized[Text]=None, 
        predicted_function: Optional[Featurized[Text]]=None, 
        mutation_description: Optional[Featurized[Text]]=None,
    ):
        batch_size = protein1_embeds.shape[0]
        device = protein1_embeds.device
        input_emb = self.llm.get_input_embeddings()
        bos_tokens = self.llm_tokenizer('<s>', return_tensors='pt', add_special_tokens=False).to(device).input_ids
        bos_embeds = input_emb(bos_tokens)  # [1, 1, 4096]
        sys_prompt_tokens = self.llm_tokenizer(
            self.config.system_prompt,
            max_length=self.config.func_maxlen,
            padding=False,
            truncation=True,
            return_tensors='pt', 
            add_special_tokens=False,
        ).to(device).input_ids
        sys_embeds = input_emb(sys_prompt_tokens)
        if predicted_function is None:    # CoT stage 1
            sys_embeds = sys_embeds.expand(batch_size, -1, -1)
            bos_embeds = bos_embeds.expand(batch_size, -1, -1)
            bop_embeds = self.bop_embeds.expand(batch_size, -1, -1)
            eop_embeds = self.eop_embeds.expand(batch_size, -1, -1)
            bom_embeds = self.bom_embeds.expand(batch_size, -1, -1)
            eom_embeds = self.eom_embeds.expand(batch_size, -1, -1)
            wrapped_embeds = torch.cat([bos_embeds, sys_embeds, bop_embeds, protein1_embeds, eop_embeds], dim=1)
            attention_mask = torch.ones((batch_size, wrapped_embeds.shape[1]), dtype=torch.long, device=device)
            return wrapped_embeds, attention_mask
        
        else:   # CoT stage 2
            bop_embeds = self.bop_embeds.to(device)
            eop_embeds = self.eop_embeds.to(device)
            bom_embeds = self.bom_embeds.to(device)
            eom_embeds = self.eom_embeds.to(device)
            batched_embeds, batched_attn_mask = [], []
            if mutation_description is not None:
                batched_regress_ids = []
            for i in range(batch_size):
                function_tokens = predicted_function[i]
                func_embeds = input_emb(function_tokens)
                
                if mutation_description is not None:
                    mut_eff_embeds = input_emb(mutation_description.input_ids[i])
                    soft_embeds = self.soft_tokens.to(device)
                    regress_start_id = sys_embeds.shape[1] + self.config.num_query_tokens_protein1 + 4 + func_embeds.shape[0] + mut_eff_embeds.shape[0]
                    wrapped_embeds = torch.cat([
                        bos_embeds, sys_embeds, bop_embeds, protein1_embeds[i].unsqueeze(0), eop_embeds, 
                        func_embeds.unsqueeze(0), mut_eff_embeds.unsqueeze(0),
                        bom_embeds, soft_embeds
                    ], dim=1)
                    regress_ids = torch.cat([
                        torch.zeros(regress_start_id, dtype=torch.long, device=device),
                        torch.ones(self.config.num_query_tokens_protein2, dtype=torch.long, device=device),
                    ], dim=0).bool()
                    batched_regress_ids.append(regress_ids)
                else:
                    mutation_tokens = stage2_prompt.input_ids[i]
                    muta_embeds = input_emb(mutation_tokens)
                    wrapped_embeds = torch.cat([
                        bos_embeds, sys_embeds, bop_embeds, protein1_embeds[i].unsqueeze(0), eop_embeds, 
                        func_embeds.unsqueeze(0), muta_embeds.unsqueeze(0),
                        bom_embeds, protein2_embeds[i].unsqueeze(0), eom_embeds,
                    ], dim=1)
                wrapped_attn_mask = torch.ones((1, wrapped_embeds.shape[1]), dtype=torch.long, device=device)
                batched_embeds.append(wrapped_embeds)
                batched_attn_mask.append(wrapped_attn_mask)
        
            if mutation_description is None:
                batched_embeds, batched_attn_mask = self.add_padding(
                        batched_embeds, batched_attn_mask, targets=None, regress_ids=None, padding="left")
                return batched_embeds, batched_attn_mask
            else:
                batched_embeds, batched_attn_mask, batched_regress_ids = self.add_padding(
                        batched_embeds, batched_attn_mask, targets=None, regress_ids=batched_regress_ids, padding="left")
                return batched_embeds, batched_attn_mask, batched_regress_ids
    
    def forward_mutation_explanation(self, wild_type: Featurized[Protein], mutant: Featurized[Any], label: Featurized[Text], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Training MutaPLM is currently unavailable! Please see https://github.com/PharMolix/MutaPLM for training or use MutaPLM-mini-M2T instead.")

    @torch.no_grad()
    def predict_mutation_explanation(self, 
        wild_type: Featurized[Protein], 
        mutant: Featurized[Any], 
        stage2_prompt: Featurized[Text],
        gt_function: Optional[Featurized[Text]]=None, 
        **kwargs
    ) -> List[Text]:
        device = wild_type.input_ids.device
        with self.maybe_autocast(device):
            protein1_embeds, protein2_embeds = self._encode_protein(wild_type, mutant)
            if gt_function is None:
                # stage 1
                input_embeds, attn_mask = self._wrapped_sentence_inference(protein1_embeds, protein2_embeds)
                outputs_function = self.llm.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attn_mask,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    **self.config.text_generation.todict(),
                )
                outputs_function[outputs_function == 0] = 2 # convert output id 0 to 2 (eos_token_id)
                outputs_function = self.llm_tokenizer.batch_decode(outputs_function)
                logging.info(f"Predicted function: {outputs_function}")
                outputs_function = self.llm_tokenizer(
                    outputs_function,
                    max_length=self.config.func_maxlen,
                    padding=False,
                    truncation=True,
                    return_tensors='pt', 
                    add_special_tokens=False,
                ).to(device)
            else:
                outputs_function = gt_function
            # stage 2
            input_embeds, attn_mask = self._wrapped_sentence_inference(protein1_embeds, protein2_embeds, stage2_prompt, predicted_function=outputs_function.input_ids)
            outputs_effect = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                **self.config.text_generation.todict(),
            )
            outputs_effect[outputs_effect == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_effect_text = self.llm_tokenizer.batch_decode(outputs_effect, skip_special_tokens=True)
            output_effect_text = [text.strip() for text in output_effect_text]
        return output_effect_text

    def forward_mutation_engineering(self, wild_type: Featurized[Protein], prompt: Featurized[Text], label: Featurized[Any], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Training MutaPLM is currently unavailable! Please see https://github.com/PharMolix/MutaPLM for training or use MutaPLM-mini-T2M instead.")

    @torch.no_grad()
    def predict_mutation_engineering(self, 
        wild_type: Featurized[Protein], 
        prompt: Featurized[Text], 
        position: Optional[torch.LongTensor]=None, 
        gt_function: Optional[Featurized[Text]]=None, 
        **kwargs
    ) -> List[List[str]]:
        device = wild_type.input_ids.device
        with self.maybe_autocast(device):
            protein_embeds = self._encode_protein(wild_type, None)
            if gt_function is None:
                # stage 1
                input_embeds, attn_mask = self._wrapped_sentence_inference(protein1_embeds=protein_embeds)
                outputs_function = self.llm.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attn_mask,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    **self.config.text_generation.todict(),
                )
                outputs_function[outputs_function == 0] = 2 # convert output id 0 to 2 (eos_token_id)
                outputs_function = self.llm_tokenizer.batch_decode(outputs_function)
                logging.info(f"Predicted function: {outputs_function}")
                outputs_function = self.llm_tokenizer(
                    outputs_function,
                    max_length=self.config.func_maxlen,
                    padding=False,
                    truncation=True,
                    return_tensors='pt', 
                    add_special_tokens=False,
                ).to(device)
            else:
                outputs_function = gt_function

            # stage 2
            input_embeds, attn_mask, soft_ids = self._wrapped_sentence_inference(protein1_embeds=protein_embeds, predicted_function=outputs_function.input_ids, mutation_description=prompt)
            soft_output = self.llm.model(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True
            ).hidden_states[-1]
            soft_output = soft_output[soft_ids].contiguous()
            soft_output = self.proj_text(soft_output.view(wild_type.input_ids.shape[0], self.config.num_query_tokens_protein2, -1))
            scores = self.protein_model.lm_design(
                input_ids=wild_type.input_ids,
                attention_mask=wild_type.attention_mask,
                encoder_hidden_states=soft_output,
                encoder_attention_mask=torch.ones(soft_output.shape[:-1], dtype=torch.long).to(device)
            )
            outputs = []
            for i in range(scores.shape[0]):
                if position is None:
                    top50 = scores[i].flatten().topk(50).indices
                    pred_pos = top50 // len(self.protein_tokenizer)
                    pred_aa = top50 % len(self.protein_tokenizer)
                else:
                    pred_pos = position[i].repeat(len(self.protein_tokenizer))
                    pred_aa = scores[i][position[i]].sort(descending=True).indices
                wt_aa = self.protein_tokenizer.batch_decode(wild_type.input_ids[i][pred_pos])
                pred_aa = self.protein_tokenizer.batch_decode(pred_aa)
                outputs.append([f"{wt_aa[j]}{pred_pos[j]}{pred_aa[j]}" for j in range(50)])
            return outputs