from typing import Dict, List
from typing_extensions import Any

from huggingface_hub import snapshot_download
import logging
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput

from open_biomed.data import Molecule, Protein, Text
from open_biomed.models.task_models import TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel, TextBasedProteinGenerationModel, MoleculeQAModel, ProteinQAModel
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import ProteinFeaturizer, MoleculeTransformersFeaturizer, TextTransformersFeaturizer, Featurized
from open_biomed.utils.misc import concatenate_tokens
import selfies as sf

class BioT5ProteinFeaturizer(ProteinFeaturizer):
    def __init__(self,
        model_name_or_path: str,
        max_length: int=1024,
        add_special_tokens: bool=False,
    ) -> None:
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, model_max_length=max_length, truncation=True)
        self.add_special_tokens = add_special_tokens

    def __call__(self, protein: Protein) -> Dict[str, Any]:
        inputs = "<bop>"
        for i in range(len(protein.sequence)):
            inputs += "<p>" + protein.sequence[i]
        inputs += "<eop>"
        output = self.tokenizer(
            inputs,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        )
        return output

    def decode(self, inputs: str) -> Protein:
        inputs = inputs.lstrip("<bop>").rstrip("<eop>").replace("<p>", "").replace(" ", "")
        return Protein.from_fasta(inputs)

class BioT5(TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel, TextBasedProteinGenerationModel, ProteinQAModel, MoleculeQAModel):
    def __init__(self, model_cfg: Config) -> None:
        super(BioT5, self).__init__(model_cfg)
        if not os.path.exists(model_cfg.hf_model_name_or_path):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            snapshot_download(repo_id="QizhiPei/biot5-base", local_dir=model_cfg.hf_model_name_or_path, force_download=True)
        self.main_model = T5ForConditionalGeneration.from_pretrained(model_cfg.hf_model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_cfg.hf_model_name_or_path)
        self.featurizers = {
            "molecule": MoleculeTransformersFeaturizer(
                tokenizer=self.tokenizer, 
                max_length=model_cfg.smiles_max_length,
                base='SELFIES',
            ),
            "protein": BioT5ProteinFeaturizer(
                model_name_or_path=model_cfg.hf_model_name_or_path, 
                max_length=model_cfg.protein_max_length,
                add_special_tokens=True,
            ),
            "text": TextTransformersFeaturizer(
                tokenizer=self.tokenizer,
                max_length=model_cfg.text_max_length,
            )
        }
        self.collators = {
            "molecule": DataCollatorWithPadding(self.tokenizer, padding=True),
            "protein": DataCollatorWithPadding(self.tokenizer, padding=True),
            "text": DataCollatorWithPadding(self.tokenizer, padding=True),
        }
        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def _forward(self, input: BatchEncoding, label: BatchEncoding) -> torch.Tensor:
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**input).last_hidden_state)
        return self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=input.attention_mask,
            decoder_attention_mask=label.attention_mask,
            labels=label.input_ids
        ).loss

    def _predict(self, input: BatchEncoding) -> torch.Tensor:
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**input).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=input.attention_mask,
            **self.config.predict.todict(),
        )
        return self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)

    def forward_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        return {"loss": self._forward(concatenate_tokens([molecule, text]), label)}

    @torch.no_grad()
    def predict_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
    ) -> List[Molecule]:
        preds = self._predict(concatenate_tokens([molecule, text]))
        return [Molecule.from_smiles(sf.decoder("".join(sel.split(" ")))) for sel in preds]
    
    def forward_molecule_captioning(self, 
        molecule: Featurized[Molecule], 
        label: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        return {"loss": self._forward(molecule, label)}

    @torch.no_grad()
    def predict_molecule_captioning(self, 
        molecule: Featurized[Molecule], 
    ) -> List[Text]:
        preds = self._predict(molecule)
        return [Text.from_str(text) for text in preds]

    def forward_text_guided_molecule_generation(self, 
        text: Featurized[Text], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        return {"loss": self._forward(text, label)}

    @torch.no_grad()
    def predict_text_guided_molecule_generation(self, 
        text: Featurized[Text], 
    ) -> List[Molecule]:
        preds = self._predict(text)
        return [Molecule.from_selfies("".join(sel.split(" "))) for sel in preds]

    def forward_molecule_question_answering(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text], 
        label: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        return {"loss": self._forward(concatenate_tokens([molecule, text]), label)}

    @torch.no_grad()
    def predict_molecule_question_answering(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text]
    ) -> List[Text]:
        preds = self._predict(concatenate_tokens([molecule, text]))
        return [Text.from_str(text) for text in preds]

    def forward_text_based_protein_generation(self,
        text: Featurized[Text],
        label: Featurized[Protein],
    ) -> Dict[str, torch.Tensor]:
        return {"loss": self._forward(text, label)}

    @torch.no_grad()
    def predict_text_based_protein_generation(self,
        text: Featurized[Text]
    ) -> List[Protein]:
        preds = self._predict(text)
        return [self.featurizers["protein"].decode(pred) for pred in preds]

    def forward_protein_question_answering(self, 
        protein: Featurized[Protein], 
        text: Featurized[Text], 
        label: Featurized[Text]
    ) -> Dict[str, torch.Tensor]:
        return {"loss": self._forward(concatenate_tokens([protein, text]), label)}

    @torch.no_grad()
    def predict_protein_question_answering(self, 
        protein: Featurized[Protein], 
        text: Featurized[Text]
    ) -> List[Text]:
        preds = self._predict(concatenate_tokens([protein, text]))
        return [Text.from_str(text) for text in preds]