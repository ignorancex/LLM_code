import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

import properties
import utils

_MAX_LENGTH = 512
_LABELS = {
    '0': 0,
    '1': 1,
    1: 1.0,
    0: 0.0,
}
_PATH_TOKENIZER = "Elron/bleurt-base-512"


def _load_data(path, dataset: properties.Dataset = None):
    """
    Loads data for finetuning BLEURT model for FC evidence evaluation.
    :return:
    """
    if dataset == properties.Dataset.AVERITEC_MANUAL_EVAL:
        if path.endswith(".csv") or path.endswith(".xlsx"):
            references = []
            targets = []
            labels = []
            if path.endswith(".csv"):
                dataset_manual_eval = pd.read_csv(path)
            elif path.endswith(".xlsx"):
                dataset_manual_eval = pd.read_excel(path, header=0)
            else:
                raise ValueError(
                    "Exception while reading Averitec manual eval data, 'dataset_manual_eval' should either be a .csv or a .xlsx file.")

            for i, row in dataset_manual_eval.iterrows():
                labels.append(row['verdict_agreement'])
                references.append(row['reference evidence'].replace("\n", " ").replace("\n", " "))
                targets.append(row['predicted evidence'].replace("\n", " ").replace("\n", " "))
        else:  # json file for checklist tests
            _, targets, _ = utils.read_averitec_dataset(path)
            path_base_data = os.path.join(os.path.dirname(path), "base_data.json")
            print("Path base data is: {}".format(path_base_data))
            _, references, _ = utils.read_averitec_dataset(path_base_data)
            labels = [1] * len(references)
            targets = [utils.normalize_text(utils.remove_special_characters(t)) for t in targets]
            references = [utils.normalize_text(utils.remove_special_characters(r)) for r in references]
    else:
        data = utils.load_jsonl_file(path)
        references = [entry['reference'] for entry in data]
        targets = [entry['target'] for entry in data]
        labels = [_LABELS[entry['score']] for entry in data]
    return references, targets, labels


def _prepare_dataset(path, tokenizer, dataset: properties.Dataset = None):
    references, targets, labels = _load_data(path, dataset)
    print("First reference: {}".format(references[0]))
    print("First target: {}".format(targets[0]))
    print("First label: {}".format(labels[0]))

    data_tokenized = tokenizer(references, targets,
                               max_length=_MAX_LENGTH,
                               return_tensors='pt',
                               padding='longest',
                               truncation=True)
    return utils.CustomDataset(data_tokenized, labels)


def _compute_metrics(pred):
    labels = pred.label_ids
    preds = np.rint(pred.predictions)
    avg_bleurt = np.average(pred.predictions)

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'avg_bleurt': avg_bleurt
    }


def _continue_training(model, training_args, train_dataset, dev_dataset, test_dataset, output_path):
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=_compute_metrics,
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def _train(model, training_args, train_dataset, dev_dataset, test_dataset, output_path, do_training=True):
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=_compute_metrics,
    )
    if do_training:
        trainer.train()
        trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    print("Result_dict: {}".format(result_dict.metrics))
    return result_dict


def run_reference_scorer(train_dataset_path: str, dev_dataset_path: str,
                         test_dataset_path: str, output_path: str, results_filename: str, samples_filenames: str,
                         _model_path: str, train=bool, continue_train=bool, epoch=5, train_bs=32, test_bs=64,
                         lr=1e-5, dataset: properties.Dataset = None, calc_diff_base_data: bool = False):
    # tokenizer = BleurtTokenizer.from_pretrained(hg_model_hub_name)
    # model = BleurtForSequenceClassification.from_pretrained(hg_model_hub_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(_PATH_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(_model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if train or continue_train:
        print("Log: Model will be trained!")
        model.train()

    training_args = TrainingArguments(
        output_dir=output_path,  # output directory
        num_train_epochs=epoch,  # total number of training epochs
        per_device_train_batch_size=train_bs,  # batch size per device during training
        per_device_eval_batch_size=test_bs,  # batch size for evaluation
        # todo check if parameters below need to be changed for BLEURT
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        # gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=3000,
        save_steps=3000,
        metric_for_best_model="eval_f1_micro",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=lr,
    )
    # todo preprocess manual eval data to be in the desired format

    train_dataset = _prepare_dataset(train_dataset_path, tokenizer=tokenizer)
    test_dataset = _prepare_dataset(test_dataset_path, tokenizer=tokenizer, dataset=dataset)
    dev_dataset = _prepare_dataset(dev_dataset_path, tokenizer=tokenizer)

    if continue_train:
        print("Continuing training for model saved at {}".format(_model_path))
        results = _continue_training(model=model, training_args=training_args, train_dataset=train_dataset,
                                     dev_dataset=dev_dataset, test_dataset=test_dataset, output_path=output_path)
    else:
        results = _train(model=model, training_args=training_args, train_dataset=train_dataset,
                         dev_dataset=dev_dataset, test_dataset=test_dataset, output_path=output_path,
                         do_training=train)
    with open(os.path.join(output_path, results_filename), "w") as f:
        if calc_diff_base_data:
            print("outputpath: {}".format(os.path.join(output_path, "results_base_data.json")))
            results_base = utils.load_json_file(os.path.join(output_path, "results_base_data.json"))
            results.metrics['diff_f1_micro'] = utils.percentage_difference(results_base['test_f1_micro'],
                                                                      results.metrics['test_f1_micro'])
            results.metrics['diff_avg_bleurt'] = utils.percentage_difference(results_base['test_avg_bleurt'],
                                                                      results.metrics['test_avg_bleurt'])
        json.dump(results.metrics, f, indent=2)

    with open(os.path.join(output_path, samples_filenames), "w") as f:
        for i, logits in enumerate(results.predictions.tolist()):
            # prediction = min(1, max(0, np.rint(np.array(logits))[0]))
            prediction = np.array(logits)[0]
            f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
            f.write(f"label: {_LABELS[results.label_ids.tolist()[i]]}\n")
            f.write(f"prediction: {prediction}\n\n")

    if dataset == properties.Dataset.AVERITEC_MANUAL_EVAL and (test_dataset_path.endswith(".csv") or test_dataset_path.endswith(".xlsx")):
        # save predictions as csv (incl. a field telling if prediction and label agree
        if test_dataset_path.endswith(".csv"):
            input_dataset = pd.read_csv(test_dataset_path)
        elif test_dataset_path.endswith(".xlsx"):
            input_dataset = pd.read_excel(test_dataset_path, header=0)
        else:
            raise ValueError(
                "Exception while reading Averitec manual eval data, 'file_path' should either be a .csv or a .xlsx file.")

        predictions_df = pd.DataFrame(columns=['id', 'claim', 'label', 'prediction'])
        for i, logits in enumerate(results.predictions.tolist()):
            pred = np.array(logits)[0]
            new_row = {
                'id': input_dataset.iloc[i]['id'],
                'claim': input_dataset.iloc[i]['claim'],
                'label': input_dataset.iloc[i]['verdict_agreement'],
                'prediction': pred,
            }
            predictions_df = pd.concat([predictions_df, pd.DataFrame([new_row])], ignore_index=True)
        predictions_df.to_csv(os.path.join(output_path, samples_filenames.split(".txt")[0] + ".csv"))
