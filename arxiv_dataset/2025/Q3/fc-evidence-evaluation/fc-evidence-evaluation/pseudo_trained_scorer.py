import json
import os
import re
import statistics

import evaluate
import numpy as np
import pandas as pd
import pymysql
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments

import properties
import scorer_utils
import utils

_MAX_LENGTH = 1024
# fever_dataset_path = os.path.join("data", "shared_task_test_annotations_evidence.jsonl")
_TBL_DELIM = " ; "
_ENTITY_LINKING_PATTERN = re.compile('#.*?;-*[0-9]+,(-*[0-9]+)#')
_WIKI_DB_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data"

_METRIC = evaluate.load("glue", "mrpc")
_PATH_TOKENIZER = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
_FEVER_DB_PW = open('/scratch/users/k20116188/fc_evidence_evaluation/credentials/fever_db_pw.txt', 'r').read()


def _softmax(logits):
    """Apply softmax to each row"""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))  # Stability improvement
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _compute_metrics_conf(eval_preds):
    """
    get the probabilities of gold labels
    :param eval_preds:
    :return:
    """
    logits, labels = eval_preds
    # Example logits array
    # logits = np.array([[2.5, 1.2, 0.3],
    #                    [0.1, 3.8, 1.1],
    #                    [1.1, 0.2, 2.3]])

    # Gold list indicating the class for which to get the probability
    # gold = [0, 0, 2]
    gold = labels

    # Get softmax probabilities
    probabilities = _softmax(logits)

    # Extract the probability for the specified class for each sample
    specified_class_probabilities = np.array([probabilities[i, gold[i]] for i in range(len(gold))])

    avg_score = statistics.mean(specified_class_probabilities)
    return {
        'accuracy': avg_score,
        'precision': avg_score,
        'recall': avg_score,
        'f1_macro': avg_score,
        'f1_micro': avg_score,
    }


def _compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions, average='weighted')
    recall = recall_score(y_true=labels, y_pred=predictions, average='weighted')
    f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
    }


def _train(model, training_args, train_dataset, dev_dataset, test_dataset, output_path, do_training=False):
    score_calc = _compute_metrics if do_training else _compute_metrics_conf
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=score_calc,
    )

    if do_training:
        # trainer.train()
        trainer.train(resume_from_checkpoint=True)  # todo remove later
        trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    print("Result_dict: {}".format(result_dict.metrics))
    return result_dict


def continue_training(model, training_args, train_dataset, dev_dataset, test_dataset, output_path):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=scorer_utils.compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def check_dataset(dataset):
    print("Check for NaNs..")
    for i, item in enumerate(dataset):
        print("process ", i)
        for k in item:
            if torch.isnan(item[k]).any():
                print("NaN in item ", i, " ", k)
    print("Check completed")


def run_nli_scorer(model_path: str, dataset: properties.Dataset, train_dataset_path: str, dev_dataset_path: str,
                   test_dataset_path: str, output_path: str, results_filename: str, samples_filenames: str,
                   train_model: bool, train_bs: int, test_bs: int, epoch: int, calc_diff_base_data: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(_PATH_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype="auto")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if train_model:
        print("Model will be trained!")
        model.train()

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epoch,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=test_bs,
        warmup_steps=50,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=10000,
        save_steps=10000,
        metric_for_best_model="eval_f1_micro",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=1e-06,
        fp16=True,  # mixed precision training
        # debug="underflow_overflow",
    )
    train_claims, train_evidences, train_labels = utils.read_vitaminc_dataset(train_dataset_path)
    eval_claims, dev_evidences, eval_labels = utils.read_vitaminc_dataset(dev_dataset_path)

    if dataset == properties.Dataset.FEVER:
        wiki_db = pymysql.connect(host="localhost", port=3306, user="root", password=_FEVER_DB_PW, db="fever").cursor()
        train_claims, train_evidences, train_labels = utils.read_fever_dataset(train_dataset_path, wiki_db)
        test_claims, test_evidences, test_labels = utils.read_fever_dataset(test_dataset_path, wiki_db)
        eval_claims, dev_evidences, eval_labels = utils.read_fever_dataset(dev_dataset_path, wiki_db)
    elif dataset == properties.Dataset.FEVER_REANNOTATION:
        test_claims, test_evidences, test_labels = utils.read_fever_dataset_reannotation(test_dataset_path)
    elif dataset in [properties.Dataset.AVERITEC, properties.Dataset.AVERITEC_AFTER_P4]:
        # select also for checkist tests properties.Dataset.AVERITEC
        # train_claims, train_evidences, train_labels = utils.read_averitec_dataset(train_dataset_path)
        test_claims, test_evidences, test_labels = utils.read_averitec_dataset(test_dataset_path)
        # eval_claims, dev_evidences, eval_labels = utils.read_averitec_dataset(dev_dataset_path)
    elif dataset == properties.Dataset.HOVER:
        wiki_db = utils.connect_to_db(os.path.join(_WIKI_DB_PATH, "hover", 'wiki_wo_links.db'))
        train_claims, train_evidences, train_labels = utils.read_hover_dataset(train_dataset_path, wiki_db)
        test_claims, test_evidences, test_labels = utils.read_hover_dataset(test_dataset_path, wiki_db)
        eval_claims, dev_evidences, eval_labels = utils.read_hover_dataset(dev_dataset_path, wiki_db)
    elif dataset == properties.Dataset.VITAMINC:
        # also used for train.jsonl and dev.jsonl => all
        test_claims, test_evidences, test_labels = utils.read_vitaminc_dataset(test_dataset_path)
    elif dataset == properties.Dataset.AVERITEC_MANUAL_EVAL:
        # evidence is reference evidence because humans evaluated based on that
        test_claims, test_evidences, test_labels = utils.read_averitec_manual_eval_data(test_dataset_path)
    else:
        raise Exception("Dataset provided does not match available datasets: {}".format(properties.Dataset))

    train_dataset = utils.prepare_dataset(train_claims, train_evidences, train_labels, tokenizer)
    dev_dataset = utils.prepare_dataset(eval_claims, dev_evidences, eval_labels, tokenizer)
    test_dataset = utils.prepare_dataset(test_claims, test_evidences, test_labels, tokenizer)

    results = _train(model, training_args, train_dataset=train_dataset,
                     dev_dataset=dev_dataset, test_dataset=test_dataset, output_path=output_path,
                     do_training=train_model)
    with open(os.path.join(output_path, results_filename), "w") as f:
        if calc_diff_base_data:
            results_base = utils.load_json_file(os.path.join(output_path, "results_base_data.json"))
            results.metrics['diff_f1_micro'] = utils.percentage_difference(results_base['test_f1_micro'],
                                                                           results.metrics['test_f1_micro'])
            results.metrics['diff_f1_macro'] = utils.percentage_difference(results_base['test_f1_macro'],
                                                                           results.metrics['test_f1_macro'])
        json.dump(results.metrics, f, indent=2)

    with open(os.path.join(output_path, samples_filenames), "w") as f:
        for i, logits in enumerate(results.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            # if predictions != results.label_ids.tolist()[i]: # this only outputs predictions if they are wrong (also in other files!)
            f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
            f.write(f"label: {properties.LABEL_DICT[properties.Label(results.label_ids.tolist()[i])]}\n")
            f.write(f"prediction: {properties.LABEL_DICT[properties.Label(predictions)]}\n\n")

    if dataset == properties.Dataset.AVERITEC_MANUAL_EVAL:
        # save predictions as csv (incl. a field telling if prediction and label agree
        if test_dataset_path.endswith(".csv"):
            input_dataset = pd.read_csv(test_dataset_path)
        elif test_dataset_path.endswith(".xlsx"):
            input_dataset = pd.read_excel(test_dataset_path, header=0)
        else:
            raise ValueError(
                "Exception while reading Averitec manual eval data, 'test_dataset_path' should either be a .csv or a .xlsx file.")

        predictions_df = pd.DataFrame(columns=['id', 'claim', 'label', 'prediction', 'score'])
        for i, logits in enumerate(results.predictions.tolist()):
            pred = np.argmax(logits, axis=-1)
            probabilities = _softmax(logits)
            label_ind = properties.LABEL_DICT[properties.Label(input_dataset.iloc[i]['label_majority'].strip().lower())]

            new_row = {
                'id': input_dataset.iloc[i]['id'],
                'claim': input_dataset.iloc[i]['claim'],
                'label': input_dataset.iloc[i]['label_majority'].strip(),
                'prediction': properties.LABEL_DICT_REVERSE[properties.LABEL_DICT[properties.Label(pred)]],
                'score': probabilities[label_ind],
            }
            predictions_df = pd.concat([predictions_df, pd.DataFrame([new_row])], ignore_index=True)

        # label_match is used for correlation analysis later
        predictions_df['label'] = predictions_df['label'].replace("contradicting information (some evidence parts "
                                                                  "support the claim whereas others refute it)",
                                                                  "not enough information")
        predictions_df['label_match'] = (predictions_df['label'] == predictions_df['prediction']).astype(int)
        predictions_df.to_csv(os.path.join(output_path, samples_filenames.split(".txt")[0] + ".csv"))
