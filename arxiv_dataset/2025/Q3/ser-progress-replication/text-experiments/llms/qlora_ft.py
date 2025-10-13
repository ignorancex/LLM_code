
import audmetric
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import os
import random
import torch
import yaml

from data import LabelEncoder
from datasets import Dataset
from peft import get_peft_model
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

class MSPDataset(torch.utils.data.Dataset):
    def __init__(self, df, path, tokenizer, target_transform=None):
        self.df = df
        self.tokenizer = tokenizer
        self.target_transform = target_transform
        self.path = path
        self.map_dict = {
            "A": "Angry",
            "H": "Happy",
            "N": "Neutral",
            "S": "Sad"
        }
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        file = self.df.loc[item, "FileName"]
        label = self.df.loc[item, "EmoClass"]
        
        with open(os.path.join(self.path, file.replace(".wav", ".txt")), "r") as fp:
            text = fp.read()
        features = self.tokenizer(
            "The emotion of the following transcript is. "
            "Answer with one of the following words: "
            "(Angry, Happy, Sad, Neutral). "
            f"Transcript: {text}"
        )
        if self.target_transform is not None:
            label = self.target_transform(label)
        else:
            label = self.map_dict[label]
        features["label"] = label
        return features


class AIBODataset(torch.utils.data.Dataset):
    def __init__(self, df, transliteration, target_transform=None, tokenizer=None):
        self.df = df
        self.target_transform = target_transform
        self.transliteration = transliteration
        self.indices = self.df.index
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        index = self.indices[item]
        file = self.df.loc[index, "id"]
        label = self.df.loc[index, "class"]
        if self.target_transform is not None:
            label = self.target_transform(label)
        text = self.transliteration.loc[file, "text"]
        features = self.tokenizer(
            "The emotion of the following transcript is. "
            "Answer with one of the following words: "
            "(Angry, Happy, Sad, Neutral). "
            f"Transcript: {text}"
        )
        features["label"] = label
        return features
        # return text, label

def compute_metrics(x):
    labels = x.label_ids
    predictions = x.predictions.argmax(axis=1)

    return {
        "uar": audmetric.unweighted_average_recall(labels, predictions),
        "f1": audmetric.unweighted_average_fscore(labels, predictions),
        "acc": audmetric.accuracy(labels, predictions)
    }

class MyTrainer(Trainer):
    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # print(outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss_fn = torch.nn.CrossEntropyLoss(self.weight.to(self.model.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def create_aibo(labels):
    df = pd.read_csv(
        labels,
        header=None,
        sep=" ",
    )
    df = df.rename(columns={0: "id", 1: "class", 2: "conf"})
    df["file"] = df["id"].apply(lambda x: x + ".wav")
    df["school"] = df["id"].apply(lambda x: x.split("_")[0])
    df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
    encoder = LabelEncoder(sorted(df["class"].unique().tolist()))
    df_test = df.loc[df["school"] == "Mont"]
    df_train_dev = df.loc[df["school"] == "Ohm"]
    speakers = sorted(df_train_dev["speaker"].unique())
    df_train = df_train_dev.loc[
        df_train_dev["speaker"].isin(speakers[:-2])
    ]
    df_dev = df_train_dev.loc[
        df_train_dev["speaker"].isin(speakers[-2:])
    ]
    return df_train, df_dev, df_test, encoder


def experiment(cfg):
    torch.manual_seed(cfg.hparams.seed)
    random.seed(cfg.hparams.seed)
    np.random.seed(cfg.hparams.seed)

    experiment_folder = os.path.join(cfg.meta.results, getattr(cfg.data, "task", "4cl"), cfg.model)
    os.makedirs(experiment_folder, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, token="", padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    if cfg.data.name == "msp":
        df = pd.read_csv(os.path.join(cfg.data.data, "Labels", "labels_consensus.csv"))
        df = df.loc[df["EmoClass"].isin(["N", "H", "A", "S"])]

        encoder = LabelEncoder(sorted(df["EmoClass"].unique().tolist()))
        df_train = df.loc[df["Split_Set"] == "Train"].reset_index(drop=True)
        df_dev = df.loc[df["Split_Set"] == "Development"].reset_index(drop=True)
        df_test = df.loc[df["Split_Set"] == "Test1"].reset_index(drop=True)
        if cfg.meta.trial:
            df_train = df_train[:100]
            df_dev = df_dev[:100]
            df_test = df_test[:100]
        train_dataset = MSPDataset(df_train, cfg.data.features, tokenizer=tokenizer, target_transform=encoder)
        x = train_dataset[0]
        print(x)
        dev_dataset = MSPDataset(df_dev, cfg.data.features, tokenizer=tokenizer, target_transform=encoder)
        test_dataset = MSPDataset(df_test, cfg.data.features, tokenizer=tokenizer, target_transform=encoder)
        target = "EmoClass"
    else:
        df_train, df_dev, df_test, encoder = create_aibo(os.path.join(cfg.data.labels, f"chunk_labels_{cfg.data.task}_corpus.txt"))
        with open(cfg.data.transcripts, "r") as fp:
            lines = fp.readlines()
        transliteration = pd.DataFrame.from_dict({x.split(" ")[0]: " ".join(x.split(" ")[1:-1]) for x in lines}, orient="index").reset_index()
        transliteration = transliteration.rename(columns={"index": "file", 0: "text"})
        transliteration = transliteration.set_index("file")
        train_dataset = AIBODataset(df_train, transliteration, tokenizer=tokenizer, target_transform=encoder)
        x = train_dataset[0]
        dev_dataset = AIBODataset(df_dev, transliteration, tokenizer=tokenizer, target_transform=encoder)
        test_dataset = AIBODataset(df_test, transliteration, tokenizer=tokenizer, target_transform=encoder)
        target = "class"


    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model,
        token="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",
        quantization_config=config,
        num_labels=len(encoder.labels)
    )
    model.config.pad_token_id = model.config.eos_token_id

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=experiment_folder,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_uar",
        save_total_limit=1,
        num_train_epochs=5
    )

    frequency = (
        df_train[target]
        .map(encoder)
        .value_counts()
        .sort_index()
        .values
    )
    weight = torch.tensor(1 / frequency, dtype=torch.float32)
    weight /= weight.sum()

    trainer = MyTrainer(
        weight=weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)
    predictions, labels, test_results = trainer.predict(test_dataset, metric_key_prefix="predict")
    np.save(os.path.join(experiment_folder, "predictions.npy"), predictions)
    print("TEST RESULTS")
    print(test_results)
    with open(f"{experiment_folder}/test.yaml", "w") as fp:
        yaml.dump(test_results, fp)



@hydra.main(version_base=None, config_path="configs-ft", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    experiment(cfg)


if __name__ == "__main__":
    main()

