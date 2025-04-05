import json
from pathlib import Path

import fire
import rouge
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq

ROUGE = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=4, limit_length=False, apply_avg=True,
                    stemming=True, ensure_compatibility=True)


def load_data(train_file, tokenizer, ):
    max_length = 1024 if "train" in str(train_file) else 4096
    model_inputs = []
    for ins in tqdm(list(map(json.loads, open(train_file))), desc="Loading...", ncols=80):
        src, tgt = ins["src"], ins["tgt"]
        inp = tokenizer(src, max_length=max_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt, max_length=max_length, truncation=True,
                               add_special_tokens=False).input_ids
        labels = [(x if x != tokenizer.pad_token_id else -100) for x in labels]
        inp["labels"] = labels
        model_inputs.append(inp)
    return model_inputs


def run(data_dir: str,
        output_dir: str,
        model_name: str = "google/t5-v1_1-base",
        epsilon: float = 0.1,
        save_steps: int = 5000,
        warmup_steps: int = 5000,
        max_steps: int = 50000,
        per_device_train_batch_size: int = 2,
        random_init: bool = False,
        seed: int = 765):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_dir = Path(data_dir)
    train = load_data(data_dir / "train.jsonl", tokenizer, )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if random_init:
        model.init_weights()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        max_grad_norm=0.1,
        label_smoothing_factor=epsilon,
        save_strategy="steps",
        save_steps=save_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        lr_scheduler_type="polynomial",
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=train,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == '__main__':
    fire.Fire(run)
