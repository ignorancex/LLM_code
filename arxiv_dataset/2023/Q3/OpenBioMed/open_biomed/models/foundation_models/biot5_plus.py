from typing import Dict, List
from typing_extensions import Any

from huggingface_hub import snapshot_download
import logging
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput
import selfies as sf

from open_biomed.data import Molecule, Text
from open_biomed.models.task_models import TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import MoleculeTransformersFeaturizer, TextTransformersFeaturizer, Featurized
from open_biomed.utils.misc import concatenate_tokens


class BioT5_PLUS(TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel):
    def __init__(self, model_cfg: Config) -> None:
        super(BioT5_PLUS, self).__init__(model_cfg)

        if not os.path.exists(model_cfg.hf_model_name_or_path):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            snapshot_download(repo_id="QizhiPei/biot5-plus-base", local_dir=model_cfg.hf_model_name_or_path, force_download=True)
        self.main_model = T5ForConditionalGeneration.from_pretrained(model_cfg.hf_model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_cfg.hf_model_name_or_path)
        self.featurizers = {
            "molecule": MoleculeTransformersFeaturizer(
                tokenizer=self.tokenizer, 
                max_length=model_cfg.smiles_max_length,
                base='SMILES',
            ),
            "text": TextTransformersFeaturizer(
                tokenizer=self.tokenizer,
                max_length=model_cfg.text_max_length,
            )
        }
        self.collators = {
            "molecule": DataCollatorWithPadding(self.tokenizer, padding=True),
            "text": DataCollatorWithPadding(self.tokenizer, padding=True),
        }
        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def forward_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        concatenated = concatenate_tokens([molecule, text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        return {"loss": self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            decoder_attention_mask=label.attention_mask,
            labels=label.input_ids
        ).loss}

    @torch.no_grad()
    def predict_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
    ) -> List[Molecule]:
        concatenated = concatenate_tokens([molecule, text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            **self.config.predict.todict(),
        )
        preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        return [Molecule.from_smiles(sf.decoder("".join(sel.split(" ")))) for sel in preds]
    
    def forward_molecule_captioning(self, 
        molecule: Featurized[Molecule], 
        label: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        concatenated = concatenate_tokens([molecule])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        return {"loss": self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            decoder_attention_mask=label.attention_mask,
            labels=label.input_ids
        ).loss}

    @torch.no_grad()
    def predict_molecule_captioning(self, 
        molecule: Featurized[Molecule], 
    ) -> List[Text]:
        concatenated = concatenate_tokens([molecule])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            **self.config.predict.todict(),
        )
        preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        return [Text.from_str(text) for text in preds]

    def forward_text_guided_molecule_generation(self, 
        text: Featurized[Text], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        task_definition = 'Definition: You are given a molecule description. Your job is to generate the molecule SELFIES that fits the description.\n\n'
        featurizer = self.featurizers["text"]
        text_ori = featurizer.tokenizer.batch_decode(text["input_ids"], truncation=True, add_special_tokens=featurizer.add_special_tokens)
        text_input = [f"{task_definition}Now complete the following example -\nInput: {i}\nOutput: " for i in text_ori]
        text_tmp = [featurizer(Text.from_str(i)) for i in text_input]
        collator = DataCollatorWithPadding(featurizer.tokenizer, padding=True)
        text = collator(text_tmp)
        
        concatenated = concatenate_tokens([text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        return {"loss": self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            decoder_attention_mask=label.attention_mask,
            labels=label.input_ids
        ).loss}

    @torch.no_grad()
    def predict_text_guided_molecule_generation(self, 
        text: Featurized[Text], 
    ) -> List[Molecule]:
        task_definition = 'Definition: You are given a molecule description. Your job is to generate the molecule SELFIES that fits the description.\n\n'
        featurizer = self.featurizers["text"]
        text_ori = featurizer.tokenizer.batch_decode(text["input_ids"], truncation=True, add_special_tokens=featurizer.add_special_tokens)
        text_input = [f"{task_definition}Now complete the following example -\nInput: {i}\nOutput: " for i in text_ori]
        text_tmp = [featurizer(Text.from_str(i)) for i in text_input]
        collator = DataCollatorWithPadding(featurizer.tokenizer, padding=True)
        text = collator(text_tmp)
        
        concatenated = concatenate_tokens([text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            **self.config.predict.todict(),
        )
        preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        return [Molecule.from_smiles(sf.decoder("".join(sel.split(" ")))) for sel in preds]