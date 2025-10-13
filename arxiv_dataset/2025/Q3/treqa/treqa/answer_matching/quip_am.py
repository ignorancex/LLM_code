from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from .am_model import BaseAMModel

torch.set_float32_matmul_precision("high")


class TokenizedDataset(Dataset):
    def __init__(self, input_strings, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_strings = input_strings
        self.max_length = max_length

    def __len__(self):
        return len(self.input_strings)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.input_strings[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class QuipAM(BaseAMModel):
    def __init__(self, batch_size=256, use_sdpa=True, compile=False):
        super().__init__()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "alirezamsh/quip-512-mocha",
            _attn_implementation="sdpa" if use_sdpa else "eager",
        ).to(self.device)

        # Use half precision if CUDA is available
        if torch.cuda.is_available():
            self.model = self.model.bfloat16()

        # Support multi-GPU if available
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        if compile:
            self.model = torch.compile(self.model, dynamic=False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("alirezamsh/quip-512-mocha")
        self.maximum_val = 5.0
        self.minimum_val = 0.0

    def evaluate_answers(
        self,
        predicted_answers: list[str],
        questions: list[str] | None = None,
        reference_answers: list[str] | None = None,
        contexts: list[str] | None = None,
    ) -> list[float]:
        # Validate inputs
        if not (questions and reference_answers and contexts):
            raise ValueError("Questions, references, and contexts must be provided.")

        # Prepare input strings efficiently
        input_strings = [
            f"{question} <q> {gold_answer} <r> {pred_answer} <c> {context}"
            for question, gold_answer, pred_answer, context in zip(
                questions, reference_answers, predicted_answers, contexts
            )
        ]

        # Create dataset and dataloader
        dataset = TokenizedDataset(input_strings, self.tokenizer, max_length=512)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(4, torch.get_num_threads()),
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        # Perform inference
        results = []
        with torch.inference_mode():
            for batch in tqdm(data_loader):
                batch = {
                    key: val.squeeze(1).to(self.device) for key, val in batch.items()
                }
                output = self.model(**batch)
                results.extend(output.logits.cpu().tolist())

        return results

    def _collate_fn(self, batch):
        # Batch individual tokenized outputs into a single dictionary
        collated_batch = {}
        for key in batch[0]:
            collated_batch[key] = torch.cat([example[key] for example in batch], dim=0)
        return collated_batch
