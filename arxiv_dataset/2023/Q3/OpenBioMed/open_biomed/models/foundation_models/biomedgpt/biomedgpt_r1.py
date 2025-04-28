from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Self

import logging

import contextlib

import numpy as np
import torch
import torch.nn as nn
import re
import os
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, EsmModel, EsmConfig, EsmTokenizer, LlamaTokenizer, LlamaConfig, AutoTokenizer, Qwen2ForCausalLM

from open_biomed.data import Molecule, Protein, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.models.functional_model import ChatModel
from open_biomed.models.foundation_models.biomedgpt.modeling_llama import LlamaForCausalLM
from open_biomed.models.molecule.graphmvp import GNNGraphMVP
from open_biomed.utils.collator import Collator, PygCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized, ProteinTransformersFeaturizer
from open_biomed.utils.mol_featurizer import MolGraphFeaturizer

class BioMedGPTR1Featurizer(Featurizer):
    def __init__(self,
        molecule_max_atoms: int=256,
        esm_tokenizer: str=None,
        protein_max_length: int=1024,
        llama_tokenizer: str=None,
        text_max_length: int=256,
    ) -> None:
        super().__init__()
        self.molecule_featurizer = MolGraphFeaturizer({"name": "BaseGNN"})
        esm_tokenizer = EsmTokenizer.from_pretrained(esm_tokenizer)
        self.protein_featurizer = ProteinTransformersFeaturizer(esm_tokenizer, protein_max_length)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer, model_max_length=text_max_length, truncation=True, truncation_side="left")

        self.molecule_max_atoms = molecule_max_atoms
    
    def __call__(self, molecule: List[Molecule]=[], protein: List[Protein]=[], text: Text=None) -> Dict[str, Any]:
        text = text.str
        featurized_molecule = [self.molecule_featurizer(mol) for mol in molecule]
        featurized_protein = [self.protein_featurizer(prot) for prot in protein] 
        cur_mol, cur_prot = 0, 0
        
        all_tokens = []
        pattern = re.compile("<moleculeHere>|<proteinHere>")
        p_text = pattern.split(text)
        spec_tokens = pattern.findall(text)
        for j in range(len(p_text)):
            all_tokens.append(self.llm_tokenizer(
                p_text[j],
                return_tensors='pt',
                add_special_tokens=False
            ))
            if j < len(spec_tokens):
                if spec_tokens[j] == "<moleculeHere>":
                    all_tokens.append(-1023 * torch.ones(min(molecule[cur_mol].get_num_atoms(), self.molecule_max_atoms)))
                    cur_mol += 1
                elif spec_tokens[j] == "<proteinHere>":
                    all_tokens.append(-1024 * torch.ones(len(protein[cur_prot].sequence)))
                    cur_prot += 1
        all_tokens = torch.cat(all_tokens)
        return {
            "molecule": featurized_molecule,
            "protein": featurized_protein,
            "text": {
                "input_ids": all_tokens,
                "attention_mask": torch.ones_like(all_tokens)
            },
        }

    def get_attrs(self) -> List[str]:
        return ["molecule", "protein", "text"]

class BioMedGPTR1Collator(Collator):
    def __init__(self, 
        molecule_max_atoms: int=256,
        esm_tokenizer: PreTrainedTokenizer=None, 
        llama_tokenizer: PreTrainedTokenizer=None,
    ) -> None:
        super().__init__()
        self.molecule_collator = PygCollator()
        self.protein_collator = DataCollatorWithPadding(esm_tokenizer, padding=False)
        self.text_collator = DataCollatorWithPadding(llama_tokenizer, padding=False)

        self.molecule_max_atoms = molecule_max_atoms

    def __call__(self, molecule: List[List[Featurized[Molecule]]], protein: List[List[Featurized[Protein]]], text: List[Featurized[Text]]) -> Dict[str, Featurized[Any]]:
        collated = {}
        flatted_molecule, batch_molecule = [], []
        for i, mol in enumerate(molecule):
            for elem in mol:
                flatted_molecule.append(elem)
                if elem.x.shape[0] > self.molecule_max_atoms:
                    batch = -1 * torch.ones(elem.x.shape[0], dtype=torch.long)
                    perm = np.random.permutation(elem.x.shape[0])
                    batch[perm[:self.molecule_max_atoms]] = i
                else:
                    batch = i * torch.ones(elem.x.shape[0], dtype=torch.long)
                batch_molecule.append(batch)
        if len(flatted_molecule) > 0:
            collated_molecule = self.molecule_collator(flatted_molecule)
            collated_molecule["global_batch"] = torch.cat(batch_molecule, dim=-1)
            collated["molecule"] = collated_molecule
        
        flatted_protein, batch_protein = [], []
        for i, prot in enumerate(protein):
            for elem in prot:
                flatted_protein.append(elem)
                batch_protein.append(torch.ones(elem.input_ids.shape[0], dtype=torch.long) * i)
        if len(collated_protein) > 0:
            collated_protein = self.protein_collator(flatted_protein)
            collated_protein["global_batch"] = torch.cat(batch_protein, dim=-1)
            collated["protein"] = collated_protein

        collated["text"] = self.text_collator(text)

        return collated

class BioMedGPTR1(BaseModel):
    def __init__(self, config: Config):
        super(BioMedGPTR1, self).__init__(config)

        # load molecule structure encoder
        self.mol_structure_encoder = GNNGraphMVP(
            num_layer=config.molecule.gin_num_layers,
            emb_dim=config.molecule.gin_hidden_dim,
            gnn_type="gin",
            drop_ratio=config.molecule.drop_ratio,
            JK="last",
        )

        # load protein structure encoder
        self.prot_tokenizer = EsmTokenizer.from_pretrained(config.protein.model_name_or_path)
        self.prot_structure_config = EsmConfig.from_json_file(os.path.join(config.protein.model_name_or_path, "config.json"))
        self.prot_structure_encoder = EsmModel(self.prot_structure_config)
        if config.protein.use_float16:
            self.prot_structure_encoder = self.prot_structure_encoder.half()

        # load llm
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.llm.model_name_or_path, use_fast=False, truncation_side="left")
        logging.info("loading llm")
        self.llm_config = LlamaConfig.from_json_file(os.path.join(config.llm.model_name_or_path, "config.json"))
        self.llm = Qwen2ForCausalLM(self.llm_config)
        if config.llm.use_float16:
            self.llm = self.llm.half()
        self.llm.resize_token_embeddings(len(self.llm_tokenizer))

        self.proj_mol = nn.Linear(config.molecule.gin_hidden_dim, self.llm.config.hidden_size)
        self.proj_prot = nn.Linear(self.prot_structure_encoder.config.hidden_size, self.llm.config.hidden_size)

        self.featurizer = BioMedGPTR1Featurizer(esm_tokenizer=config.protein.model_name_or_path, llama_tokenizer=config.llm.model_name_or_path)
        self.collator = BioMedGPTR1Collator(esm_tokenizer=self.prot_tokenizer, llama_tokenizer=self.llm_tokenizer)

    def maybe_autocast(self, device=torch.device("cuda:0"), dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: Optional[str]=None) -> Self:
        if device is None:
            device = "cpu"
        config = Config(os.path.join(model_name_or_path, "config.yaml")).model
        model = cls(config, device)
        state_dict = torch.load(open(os.path.join(model_name_or_path, "pytorch_model.bin"), "rb"), map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model

    def add_padding(self, 
        wrapped_embeds: List[torch.Tensor], 
        wrapped_attention_mask: List[torch.Tensor], 
        targets: Optional[List[torch.Tensor]]=None, 
        padding: str="right"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(wrapped_embeds)
        max_length_batch = 0
        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] > max_length_batch:
                max_length_batch = wrapped_embeds[i].shape[1]
        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] < max_length_batch:
                pad_len = max_length_batch - wrapped_embeds[i].shape[1]
                padding_tokens = torch.ones((1, pad_len, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype, device=wrapped_embeds[i].device) * self.llm_tokenizer.pad_token_id
                if padding == "right":
                    wrapped_embeds[i] = torch.cat((
                        wrapped_embeds[i], 
                        padding_tokens
                    ), dim=1)
                    wrapped_attention_mask[i] = torch.cat((
                        wrapped_attention_mask[i],
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device)
                    ), dim=1)
                    if targets is not None:
                        targets[i] = torch.cat((
                            targets[i],
                            torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100)
                        ), dim=1)
                else:
                    wrapped_embeds[i] = torch.cat((
                        padding_tokens,
                        wrapped_embeds[i], 
                    ), dim=1)
                    wrapped_attention_mask[i] = torch.cat((
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device),
                        wrapped_attention_mask[i],
                    ), dim=1)
                    if targets is not None:
                        targets[i] = torch.cat((
                            torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100),
                            targets[i],
                        ), dim=1)
        if targets is not None:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0), torch.cat(targets, dim=0)
        else:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0)

    def get_input_embeddings(self,
        molecule: Optional[Featurized[Molecule]], 
        protein: Optional[Featurized[Protein]], 
        text: Featurized[Text],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = text.input_ids.device
        batch_size = text.input_ids.shape[0]
        with self.maybe_autocast(device):
            if molecule is not None:
                mol_feats = self.mol_structure_encoder(molecule)
                mol_feats = self.proj_mol(mol_feats)
            else:
                mol_feats = None
            
            if protein is not None:
                prot_feats = []
                for prot in protein:
                    h = self.prot_structure_encoder(**prot).last_hidden_state
                    prot_feats.append(self.proj_prot(h))
        
        wrapped_embeds, wrapped_attention_mask = [], []
        for i in range(batch_size):
            mol_pos = torch.where(text.input_ids[i] == -1023)
            prot_pos = torch.where(text.input_ids[i] == -1024)
            text_input = text.input_ids[i]
            text_input[mol_pos] = 1024
            text_input[prot_pos] = 1024
            text_embeds = self.llm.get_input_embeddings()(text_input)
            if molecule is not None:
                text_embeds[:, mol_pos] = mol_feats[torch.where(molecule.global_batch == i)]
            if protein is not None:
                cur_prot_feats = [prot_feats[j] for j in torch.where(protein.global_batch == i)]
                cur_prot_feats = torch.cat(cur_prot_feats, dim=1)
                text_embeds[:, prot_pos] = cur_prot_feats

            wrapped_embeds.append(torch.cat(text_embeds, dim=1))
            wrapped_attention_mask.append(torch.ones(wrapped_embeds[-1].shape[:-1]), dtype=torch.long, device=device)
        
        return wrapped_embeds, wrapped_attention_mask

    def forward(self,
        molecule: Optional[Featurized[Molecule]],
        protein: Optional[Featurized[Protein]],
        text: Featurized[Text],
        labels: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        with self.maybe_autocast():
            inputs_embeds, inputs_attention_mask = self.get_input_embeddings(molecule, protein, text)
            
            wrapped_embeds, wrapped_attention_mask, wrapped_targets = [], [], []
            for i in range(len(inputs_embeds)):
                eos_token = torch.ones((1, 1), dtype=labels[i].input_ids.dtype, device=labels[i].input_ids.device)
                labels[i].input_ids = torch.cat([labels[i].input_ids, eos_token * self.llm_tokenizer.eos_token_id], dim=1)
                labels[i].attention_mask = torch.cat([labels[i].attention_mask, eos_token], dim=1)
                output_embeds = self.llm.get_input_embeddings()(labels[i].input_ids)
                wrapped_embeds.append(torch.cat([inputs_embeds[i], output_embeds], dim=1))
                wrapped_attention_mask.append(torch.cat([inputs_attention_mask[i], labels[i].attention_mask], dim=1))
                # do not apply loss to the padding
                targets = labels[i].input_ids.masked_fill(
                    labels[i].input_ids == self.llm_tokenizer.pad_token_id, -100
                )
                # do not apply loss to the text inputs (i.e., instruction)
                empty_targets = torch.ones(inputs_attention_mask[i].shape, dtype=torch.long).to(inputs_embeds[i].device).fill_(-100)
                wrapped_targets.append(torch.cat([empty_targets, targets], dim=1))
                
            inputs_embeds, inputs_attention_mask, targets = self.add_padding(wrapped_embeds, wrapped_attention_mask, wrapped_targets)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_attention_mask,
                labels=targets,
                return_dict=True
            )
            return outputs.loss

    @torch.no_grad()
    def generate(self,
        molecule: Optional[Featurized[Molecule]],
        protein: Optional[Featurized[Protein]],
        text: Featurized[Text],
    ) -> List[Text]:
        with self.maybe_autocast():
            inputs_embeds, inputs_attention_mask = self.get_input_embeddings(molecule, protein, text)
            inputs_embeds, inputs_attention_mask = self.add_padding(inputs_embeds, inputs_attention_mask)
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_attention_mask,
                **self.config.text_generation.todict(),
            )
        
        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [Text.from_str(text.strip()) for text in output_text]

        return output_text


class BioMedGPTR14Chat(BioMedGPTR1, ChatModel):
    def __init__(self, config: Config, device: str="cuda:0"):
        super().__init__(config)
        self.reset()
        self.device = device
        self.role_dict = {
            ChatModel.Role.USER: "<｜User｜>",
            ChatModel.Role.ASSISTANT: "<｜Assistant｜>",
        }

    def append_molecule(self, molecule: Molecule):
        msg = f"<molecule><representation><moleculeHere></representation><SMILES>{molecule}</SMILES></molecule> "
        molecule = self.collator.molecule_collator([self.featurizer.molecule_featurizer(molecule)]).to(self.device)
        with self.maybe_autocast(self.device):
            mol_embeds = self.mol_structure_encoder(molecule)
            mol_embeds = self.proj_mol(mol_embeds)
            self.mol_embs.append(mol_embeds.unsqueeze(0))
        if len(self.messages) > 0 and self.messages[-1][0] == ChatModel.Role.USER:
            self.messages[-1][1] += msg
        else:
            self.add_message(ChatModel.Role.USER, msg)

    def append_protein(self, protein: Protein):
        protein = self.collator.protein_collator([self.featurizer.protein_featurizer(protein)]).to(self.device)
        with self.maybe_autocast(self.device):
            prot_embeds = self.prot_structure_encoder(**protein).last_hidden_state
            prot_embeds = self.proj_prot(prot_embeds)
            self.prot_embs.append(prot_embeds)
        msg = "<protein><proteinHere></protein> "
        if len(self.messages) > 0 and self.messages[-1][0] == ChatModel.Role.USER:
            self.messages[-1][1] += msg
        else:
            self.add_message(ChatModel.Role.USER, msg)
    
    def compose_context(self):
        ret = ""
        for role, message in self.messages:
            if message:
                ret += self.role_dict[role] + "\n" + message + "\n"
            else:
                ret += self.role_dict[role] + "<think>\n"
        return ret

    def chat(self, user_prompt: Text) -> Text:
        if len(self.messages) > 0 and self.messages[-1][0] == ChatModel.Role.USER:
            self.messages[-1][1] += user_prompt.str
        else:
            self.add_message(ChatModel.Role.USER, user_prompt.str)
        self.add_message(ChatModel.Role.ASSISTANT, None)
        inputs = self.compose_context()

        pattern = re.compile("<moleculeHere>|<proteinHere>")
        p_text = pattern.split(inputs)
        spec_tokens = pattern.findall(inputs)
        assert len(p_text) == len(self.mol_embs) + len(self.prot_embs) + 1, "Unmatched numbers of placeholders and molecules."
        seg_tokens = [
            self.llm_tokenizer([seg], return_tensors="pt", add_special_tokens=False).to(self.device)
            for i, seg in enumerate(p_text) 
        ]
        seg_embs = [self.llm.get_input_embeddings()(seg_token.input_ids) for seg_token in seg_tokens]
        input_embs = []
        cur_mol, cur_prot = 0, 0
        for i in range(len(p_text) - 1):
            input_embs.append(seg_embs[i])
            if spec_tokens[i] == "<moleculeHere>":
                input_embs.append(self.mol_embs[cur_mol])
                cur_mol += 1
            elif spec_tokens[i] == "<proteinHere>":
                input_embs.append(self.prot_embs[cur_prot])
                cur_prot += 1
        input_embs.append(seg_embs[-1])
        input_embs = torch.cat(input_embs, dim=1)

        if input_embs.shape[1] + self.config.text_generation.max_length > self.config.context_max_length:
            begin_idx = input_embs.shape[1] + self.config.text_generation.max_length - self.config.context_max_length
            input_embs = input_embs[:, begin_idx:]
            logging.warn(f"The number of tokens in current conversation exceeds the maximum context length. Only the latest {self.config.context_max_length} tokens are used.")
        output = self.llm.generate(
            inputs_embeds=input_embs,
            **self.config.text_generation.todict(),
        ).squeeze()
        output = self.llm_tokenizer.decode(output, add_special_tokens=False, skip_special_tokens=True)
        self.messages[-1][1] = output
        return output
        
    def reset(self):
        self.messages = []
        self.mol_embs = []
        self.prot_embs = []