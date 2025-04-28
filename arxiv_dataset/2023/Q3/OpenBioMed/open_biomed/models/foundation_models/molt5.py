from typing import Dict, List
from typing_extensions import Any

from huggingface_hub import snapshot_download
import logging
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput

from open_biomed.data import Molecule, Text
from open_biomed.models.task_models import TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel, MoleculeQAModel, ProteinQAModel
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import MoleculeTransformersFeaturizer, TextTransformersFeaturizer, Featurized
from open_biomed.utils.misc import concatenate_tokens

class MolT5(TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel, MoleculeQAModel):
    def __init__(self, model_cfg: Config) -> None:
        super(MolT5, self).__init__(model_cfg)
        if not os.path.exists(model_cfg.hf_model_name_or_path):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            snapshot_download(repo_id="laituan245/molt5-base", local_dir=model_cfg.hf_model_name_or_path, force_download=True)
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
        return [Molecule.from_smiles(smi) for smi in preds]
    
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
        concatenated = concatenate_tokens([text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            **self.config.predict.todict(),
        )
        preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        return [Molecule.from_smiles(smi) for smi in preds]

    def forward_molecule_question_answering(self, 
        text: Featurized[Text], 
        label: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        concatenated = concatenate_tokens([text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        return {"loss": self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            decoder_attention_mask=label.attention_mask,
            labels=label.input_ids
        ).loss}

    @torch.no_grad()
    def predict_molecule_question_answering(self, 
        text: Featurized[Text], 
    ) -> List[Text]:
        concatenated = concatenate_tokens([text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            **self.config.predict.todict(),
        )
        preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        return [Text.from_str(text) for text in preds]
    
    def forward_protein_question_answering(self, 
        text: Featurized[Text], 
        label: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        concatenated = concatenate_tokens([text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        return {"loss": self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            decoder_attention_mask=label.attention_mask,
            labels=label.input_ids
        ).loss}

    @torch.no_grad()
    def predict_protein_question_answering(self, 
        text: Featurized[Text], 
    ) -> List[Text]:
        concatenated = concatenate_tokens([text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            **self.config.predict.todict(),
        )
        preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        return [Text.from_str(text) for text in preds]