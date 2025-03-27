import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data import VarDialDataset
from lid_dataset import LIDdataset
from tqdm import *
from roberta2way import DialectID_2
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from mappings import (
    label_to_int_mappings_3class,
    label_to_int_mappings_2class,
    int_to_label_mappings__2class,
    int_to_label_mappings__3class,
    label_mappings_en_3class,
    label_mappings_en_2class,
    label_mappings_es_3class,
    label_mappings_es_2class,
    label_mappings_pt_3class,
    label_mappings_pt_2class,
)


def inference_run(m1_name, m1_path, m2_name, m2_path, m3_name, m3_path):
    model = AutoModelForSequenceClassification.from_pretrained(
        "papluca/xlm-roberta-base-language-detection"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "papluca/xlm-roberta-base-language-detection"
    )
    test_df = VarDialDataset(
        "SharedTask/DSL-TL-test.tsv",
        label_mappings=label_to_int_mappings_3class,
        train=False,
        num_classes=2,
    )
    test = test_df.create_loader(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        true_labels = []
        en_samples = []
        es_samples = []
        pt_samples = []
        en_indices = []
        es_indices = []
        pt_indices = []
        counter = 0
        op_list = [0] * len(test_df)
        for batch in tqdm(test):
            text = tokenizer(
                text=list(batch[0]),
                return_attention_mask=True,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text = {k: v.to(device) for k, v in text.items()}
            labels = batch[1].to(device)
            outputs = model(**text).logits
            true_labels.append(batch[1])
            output_preds = torch.argmax(outputs, dim=-1)
            for i in range(len(output_preds)):
                if output_preds[i] == 13:
                    en_indices.append(counter)
                    en_samples.append(batch[0][i])
                elif output_preds[i] == 8:
                    es_indices.append(counter)
                    es_samples.append(batch[0][i])
                elif output_preds[i] == 6:
                    pt_indices.append(counter)
                    pt_samples.append(batch[0][i])
                counter += 1
    en_list = []
    es_list = []
    pt_list = []

    del model
    torch.cuda.empty_cache()
    with torch.no_grad():
        en_val = LIDdataset(en_samples, return_labels=False)
        en_loader = en_val.create_loader(8)
        model1 = DialectID_2(
            model_name=m1_name, label_mappings=label_mappings_en_3class
        )
        model1.load_state_dict(torch.load(m1_path))
        for batch in tqdm(en_loader):
            en_list.append(
                model1.inference(list(batch)).argmax(dim=-1).detach().cpu().numpy()
            )

    del model1
    torch.cuda.empty_cache()
    with torch.no_grad():
        es_val = LIDdataset(es_samples, return_labels=False)
        es_loader = es_val.create_loader(8)
        model2 = DialectID_2(
            model_name=m2_name,
            label_mappings=label_mappings_es_3class,
        )
        model2.load_state_dict(torch.load(m2_path))
        for batch in tqdm(es_loader):
            es_list.append(
                model2.inference(list(batch)).argmax(dim=-1).detach().cpu().numpy() + 2
            )

    del model2
    torch.cuda.empty_cache()
    with torch.no_grad():
        pt_val = LIDdataset(pt_samples, return_labels=False)
        pt_loader = pt_val.create_loader(8)
        model3 = DialectID_2(
            model_name=m3_name,
            label_mappings=label_mappings_pt_3class,
        )
        model3.load_state_dict(torch.load(m3_path))
        for batch in tqdm(pt_loader):
            pt_list.append(
                model3.inference(list(batch)).argmax(dim=-1).detach().cpu().numpy() + 4
            )

    en_list = np.concatenate(en_list)
    es_list = np.concatenate(es_list)
    pt_list = np.concatenate(pt_list)

    for i in range(len(en_indices)):
        op_list[en_indices[i]] = en_list[i]
    for i in range(len(es_indices)):
        op_list[es_indices[i]] = es_list[i]
    for i in range(len(pt_indices)):
        op_list[pt_indices[i]] = pt_list[i]
    print(
        classification_report(
            torch.concat(true_labels, dim=0).cpu().numpy(),
            op_list,
            digits=4,
        )
    )
    return op_list


def mapper(label_list, mapping):
    final = []
    for i in label_list:
        final.append(mapping[i])
    return final


def convert(pred_list, run_no, track_no):
    z = pd.DataFrame(pred_list)
    z.to_csv(
        f"SharedTask/DSLTL-closed-track{track_no}-run-{run_no}-VaidyaKane.tsv",
        sep="\t",
        index=False,
        header=False,
    )
