from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

import properties
import utils

_MAX_SEQ_LENGTH = 5020

# todo add prompt text
_PROMPT = """
### Input:
{}

### Response:
{}"""


def _formatting_prompt(examples, eos_token, is_testset = False):
    """
    Formats the given examples into prompts for model training/eval
    :param examples:
    :return:
    """

    # todo select correct parts of examples
    inputs = examples["Context"]
    outputs = examples["Response"]
    texts = []
    for input_, output in zip(inputs, outputs):
        if not is_testset:
            text = _PROMPT.format(input_, output) + eos_token
        else:
            text = _PROMPT.format(input_) + eos_token
        texts.append(text)
    return {"text": texts, }


def _train(model, tokenizer, train_dataset, dev_dataset, output_path, eos_token):
    """

    :param model:
    :param training_args:
    :param train_dataset:
    :param dev_dataset:
    :param test_dataset:
    :param output_path:
    :param do_training:
    :return:
    """
    # prep data
    train_dataset_formatted = _formatting_prompt(train_dataset, eos_token)
    dev_dataset_formatted = _formatting_prompt(dev_dataset, eos_token)

    # prep trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_formatted,
        eval_dataset=dev_dataset_formatted,
        dataset_text_field="text",
        max_seq_length=_MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=8,
            num_train_epochs=40,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir=output_path,
            seed=0,
        ),
    )
    trainer.train()


def _evaluate(model, tokenizer, test_dataset, eos_token):
    """

    :param model:
    :param test_dataset:
    :param output_path:
    :param eos_token:
    :return:
    """
    model = FastLanguageModel.for_inference(model)

    # we only want to add input and not output in the test dataset formatted
    test_dataset_formatted = _formatting_prompt(test_dataset, eos_token)
    inputs = tokenizer(test_dataset_formatted, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=5020, use_cache=True)
    answers = tokenizer.batch_decode(outputs)
    # answer = answers[0].split("### Response:")[-1]
    return answers


def run_scorer(model_path: str, dataset: properties.Dataset, train_dataset_path: str, dev_dataset_path: str,
               test_dataset_path: str, output_path: str, results_filename: str, samples_filenames: str,
               train_model: bool):
    """

    :param model_path:
    :param dataset:
    :param train_dataset_path:
    :param dev_dataset_path:
    :param test_dataset_path:
    :param output_path:
    :param results_filename:
    :param samples_filenames:
    :param train_model:
    :param train_bs:
    :param test_bs:
    :param epoch:
    :param calc_diff_base_data:
    :return:
    """
    # load training and eval data
    train_claims, train_evidences, train_labels = utils.read_vitaminc_dataset(train_dataset_path)
    eval_claims, dev_evidences, eval_labels = utils.read_vitaminc_dataset(dev_dataset_path)

    # test data
    if dataset in [properties.Dataset.AVERITEC, properties.Dataset.AVERITEC_AFTER_P4]:
        # select also for checkist tests properties.Dataset.AVERITEC
        # train_claims, train_evidences, train_labels = utils.read_averitec_dataset(train_dataset_path)
        test_claims, test_evidences, test_labels = utils.read_averitec_dataset(test_dataset_path)
        # eval_claims, dev_evidences, eval_labels = utils.read_averitec_dataset(dev_dataset_path)
    elif dataset == properties.Dataset.VITAMINC:
        # also used for train.jsonl and dev.jsonl => all
        test_claims, test_evidences, test_labels = utils.read_vitaminc_dataset(test_dataset_path)
    elif dataset == properties.Dataset.AVERITEC_MANUAL_EVAL:
        # evidence is reference evidence because humans evaluated based on that
        test_claims, test_evidences, test_labels = utils.read_averitec_manual_eval_data(test_dataset_path)
    else:
        raise Exception("Dataset provided does not match available datasets: {}".format(properties.Dataset))

    # load model, tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=_MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    print(model.print_trainable_parameters())

    # prep datasets
    train_dataset = utils.prepare_dataset(train_claims, train_evidences, train_labels, tokenizer)
    dev_dataset = utils.prepare_dataset(eval_claims, dev_evidences, eval_labels, tokenizer)
    test_dataset = utils.prepare_dataset(test_claims, test_evidences, test_labels, tokenizer)

    # finetune model
    _train(model=model, tokenizer=tokenizer, train_dataset=train_dataset, dev_dataset=dev_dataset,
           output_path=output_path, eos_token=tokenizer.eos_token)

    # evaluate with the testset
    results = _evaluate(model=model, tokenizer=tokenizer, test_dataset=test_dataset, eos_token=tokenizer.eos_token)

