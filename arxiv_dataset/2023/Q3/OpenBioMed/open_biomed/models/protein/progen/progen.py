from typing import Any, Dict, List, Optional, Tuple

import contextlib
from huggingface_hub import snapshot_download
import logging
import numpy as np
import os
import torch
from tokenizers import Tokenizer
from transformers import DataCollatorWithPadding

from open_biomed.data import Protein
from open_biomed.models.functional_model import ProteinDecoder
from open_biomed.models.protein.progen.modeling_progen import ProGenForCausalLM
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurized, ProteinFeaturizer, ProteinTransformersFeaturizer

class ProGenFeaturizer(ProteinFeaturizer):
    def __init__(self, 
        tokenizer: Tokenizer,
        max_length: int=1024,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, protein: Protein) -> Dict[str, Any]:
        seq = protein.sequence
        if len(seq) > self.max_length - 2:
            seq = seq[:self.max_length - 2]
        seq = '1' + seq + '2'
        return {
            "input_ids": self.tokenizer.encode(seq).ids,
            "attention_mask": [1 for i in range(len(seq))], 
        }

class ProGenCollator(Collator):
    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.encode('<|pad|>').ids[0]

    def __call__(self, protein: List[Any]) -> Dict[str, torch.Tensor]:
        max_length = np.max([len(x["input_ids"]) for x in protein])
        input_ids, attention_mask = [], []
        for x in protein:
            input_ids.append(torch.LongTensor(x["input_ids"] + [self.pad_token for i in range(max_length - len(x["input_ids"]))]))
            attention_mask.append(torch.LongTensor(x["attention_mask"] + [0 for i in range(max_length - len(x["input_ids"]))]))
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
        }

class ProGen(ProteinDecoder):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)
        if not os.path.exists(model_cfg.protein_model):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            snapshot_download(repo_id="hugohrban/progen2-base", local_dir=model_cfg.hf_model_name_or_path, force_download=True)
        self.model = ProGenForCausalLM.from_pretrained(model_cfg.model_name_or_path)
        if getattr(model_cfg, "fp16", False):
            self.model = self.model.to(torch.bfloat16)
        with open(os.path.join(model_cfg.model_name_or_path, "tokenizer.json"), "r") as f:
            self.tokenizer = Tokenizer.from_str(f.read())
        self.featurizer = ProGenFeaturizer(self.tokenizer, max_length=model_cfg.max_length)
        self.collator = ProGenCollator(self.tokenizer)

    def get_protein_processor(self) -> Tuple[ProteinFeaturizer, Collator]:
        return self.featurizer, self.collator

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.model.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def freeze_parameters(self):
        for name, param in self.named_parameters():
            # if not (("25" in name) or ("26" in name)):
            param.requires_grad = False

    def truncate(self, sample, terminals):
        pos = []
        for terminal in terminals:
            find_pos = sample.find(terminal, 1)
            if find_pos != -1:
                pos.append(find_pos)
        if len(pos) > 0:
            return sample[:(min(pos)+1)]
        else:
            return sample

    def generate_loss(self, 
        label: Featurized[Protein], 
        add_embeds: Optional[torch.Tensor]=None,           # For conditional generation (e.g. natural language)
        add_attention_mask: Optional[torch.Tensor]=None, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        with self.maybe_autocast():
            if add_embeds is not None:
                # Conditional generation
                protein_embeds = self.model.get_input_embeddings()(label["input_ids"])
                # Concatenate multiple tokenized results by putting the non-padding tokens together
                batch_size = protein_embeds.shape[0]
                embeds = torch.cat([add_embeds, protein_embeds], dim=1)
                attention_mask = torch.cat([add_attention_mask, label["attention_mask"]], dim=-1)
                non_padding_length = attention_mask.sum(-1)
                max_length = non_padding_length.max().item()

                new_embeds = []
                new_attention_mask = []
                new_labels = []
                for i in range(batch_size):
                    perm = torch.cat([
                        torch.where(attention_mask[i] == 1)[0],  # non-padding tokens
                        torch.where(attention_mask[i] == 0)[0],  # padding tokens
                    ])
                    new_embeds.append(embeds[i][perm[:max_length]])
                    new_attention_mask.append(attention_mask[i][perm[:max_length]])
                    new_labels.append(torch.cat([
                        torch.ones(add_attention_mask[i].sum().item()).to(add_attention_mask) * -100,
                        label["input_ids"][i][torch.where(label["attention_mask"][i] == 1)],
                        torch.ones(max_length - attention_mask[i].sum().item()).to(add_attention_mask) * -100
                    ], dim=0))
                # print(new_labels)
                return {"loss": self.model(
                    inputs_embeds=torch.stack(new_embeds, dim=0),
                    attention_mask=torch.stack(new_attention_mask, dim=0),
                    labels=torch.stack(new_labels, dim=0),
                    return_dict=True,
                ).loss}
            else:
                label_tokens = label["input_ids"].clone()
                label_tokens[torch.where(label["attention_mask"] == 0)] = -100
                return {"loss": self.model(
                    input_ids=label["input_ids"],
                    attention_mask=label["attention_mask"],
                    labels=label["input_ids"],
                    return_dict=True,
                ).loss}

    @torch.no_grad()
    def generate_protein(self, add_embeds: Optional[torch.Tensor]=None, add_attention_mask: Optional[torch.Tensor]=None, **kwargs) -> List[Protein]:
        with self.maybe_autocast():
            if add_embeds is not None:
                inputs = torch.ones((add_embeds.shape[0], 1), dtype=torch.long).to(self.model.device) * self.tokenizer.encode("1").ids[0]
                inputs_embeds = self.model.get_input_embeddings()(inputs)
                # Move padding to left
                new_embeds, new_attention_mask = [], []
                for i in range(add_embeds.shape[0]):
                    perm = torch.cat([
                        torch.where(add_attention_mask[i] == 1)[0],  # non-padding tokens
                        torch.where(add_attention_mask[i] == 0)[0],  # padding tokens
                    ])
                    new_embeds.append(torch.cat([add_embeds[i][perm], inputs_embeds[i]], dim=0))
                    new_attention_mask.append(torch.cat([add_attention_mask[i][perm], torch.ones(1, dtype=torch.long, device=self.model.device)], dim=0))
                outputs = self.model.generate(
                    inputs_embeds=torch.stack(new_embeds, dim=0),
                    attention_mask=torch.stack(new_attention_mask, dim=0),
                    pad_token_id=self.tokenizer.encode('<|pad|>').ids[0],
                    **self.config.generation.todict(),
                )
            else:
                inputs = torch.tensor(self.tokenizer.encode("1").ids).to(self.model.device)
                outputs = self.model.generate(
                    inputs.view([1, -1]),
                    pad_token_id=self.tokenizer.encode('<|pad|>').ids[0],
                    **self.config.generation.todict()
                )
            as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
            outputs = self.tokenizer.decode_batch(as_lists(outputs))
            return [Protein.from_fasta(self.truncate(output, ['1', '2'])[:-1]) for output in outputs]