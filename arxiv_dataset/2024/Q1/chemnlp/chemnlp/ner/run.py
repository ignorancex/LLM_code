import numpy as np
import pandas as pd

# from scipy.special import softmax
import os

# import pathlib
from tokenizers.normalizers import BertNormalizer
from simpletransformers.ner import NERModel
import re
import time
from jarvis.db.jsonutils import dumpjson, loadjson

# from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score


def parse_file(f_name):
    f = open(f_name, "r")
    data = re.split(r"\n\s*\n", f.read().strip())
    f.close()
    tokens, labels = [], []
    for sent in data:
        sent_tokens, sent_labels = [], []
        for line in sent.split("\n"):
            # l = re.split(r" +", line)
            # if len(l) != 2:
            #    sent_tokens = []
            #    break
            # if len(l[0]) == 0:
            #    l[0] = " "
            # if len(l[1]) == 0:
            #    l[1] = "O"
            ll = re.split(r" +", line)
            if len(ll) != 2:
                sent_tokens = []
                break
            if len(ll[0]) == 0:
                ll[0] = " "
            if len(ll[1]) == 0:
                ll[1] = "O"
            sent_tokens.append(ll[0])
            sent_labels.append(ll[1])
        if len(sent_tokens) > 0:
            tokens.append(sent_tokens)
            labels.append(sent_labels)
    return tokens, labels


def get_matscholar_data():
    data_dir = "."
    train_X, train_y = parse_file(os.path.join(data_dir, "train.txt"))
    val_X, val_y = parse_file(os.path.join(data_dir, "dev.txt"))
    test_X, test_y = parse_file(os.path.join(data_dir, "test.txt"))
    return train_X, train_y, val_X, val_y, test_X, test_y


def generate_dataset():
    f = open(os.path.join("vocab_mappings.txt"), "r")
    mappings = f.read().strip().split("\n")
    f.close()

    mappings = {m[0]: m[2:] for m in mappings}

    norm = BertNormalizer(
        lowercase=False,
        strip_accents=True,
        clean_text=True,
        handle_chinese_chars=True,
    )

    def normalize(text):
        text = [norm.normalize_str(s) for s in text.split("\n")]
        out = []
        for s in text:
            norm_s = ""
            for c in s:
                norm_s += mappings.get(c, " ")
            out.append(norm_s)
        return "\n".join(out)

    data = get_matscholar_data()

    norm_data = []
    for split in data:
        norm_split = []
        for s in split:
            norm_split.append(normalize("\n".join(s)).split("\n"))
        norm_data.append(norm_split)

    # f=open('train.txt','r')
    # lines=f.read().splitlines()
    # f.close()
    # for ii,i in enumerate(lines):
    # tmp=i.split()
    # if len(tmp)==2:# and tmp[1] in proj:
    #     train_data.append([ii,tmp[0],tmp[1]])
    #     #train_data.append([ii,tmp[0],proj[tmp[1]]])
    train_data = []
    count = 0
    for i, j in zip(norm_data[0], norm_data[1]):
        count += 1
        for m, n in zip(i, j):
            train_data.append([count, m, n])
    train_df = pd.DataFrame(
        train_data, columns=["sentence_id", "words", "labels"]
    )
    print("train_df", train_df)
    # f=open('test.txt','r')
    # lines=f.read().splitlines()
    # f.close()
    # eval_data=[]
    # for ii,i in enumerate(lines):
    # tmp=i.split()
    # if len(tmp)==2:# and tmp[1] in proj:
    #     eval_data.append([ii,tmp[0],tmp[1]])
    #     #eval_data.append([ii,tmp[0],proj[tmp[1]]])
    eval_data = []
    # count = 0
    for i, j in zip(norm_data[2], norm_data[3]):
        count += 1
        for m, n in zip(i, j):
            eval_data.append([count, m, n])
    eval_df = pd.DataFrame(
        eval_data, columns=["sentence_id", "words", "labels"]
    )
    print("eval_df", eval_df)
    # Create a NERModel
    # https://github.com/ThilinaRajapakse/simpletransformers/blob/ce30db2260aa7b6e20c6fed8bee3ee6c6e5972be/tests/test_named_entity_recognition.py#L4

    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["idd"] = df.index
    df["id"] = df["idd"].apply(lambda x: "ms-" + str(x))
    print("df")
    print(df)
    mem = []
    for i, ii in df.iterrows():
        info = {}
        info["id"] = ii["id"]
        info["words"] = ii["words"]
        info["sentence_id"] = ii["sentence_id"]
        info["labels"] = ii["labels"]
        mem.append(info)

    dumpjson(data=mem, filename="mat_scholar_ner.json")
    train = {}
    test = {}
    for ii, i in enumerate(mem):
        if ii < 110522:
            train[i["id"]] = i["labels"]
        else:
            test[i["id"]] = i["labels"]
    m = {}
    m["train"] = train
    m["test"] = test
    dumpjson(data=m, filename="mat_scholar_ner_labels.json")
    return df[0 : len(train_df)], df[len(eval_df) :]
    # return train_df, eval_df


def train_model(custom_labels=[], train_df=[], eval_df=[]):
    if not custom_labels:
        custom_labels = [
            "O",
            "B-MAT",
            "I-MAT",
            "B-PRO",
            "I-PRO",
            "B-SMT",
            "I-SMT",
            "B-CMT",
            "I-CMT",
            "B-DSC",
            "I-DSC",
            "B-SPL",
            "I-SPL",
            "B-APL",
            "I-APL",
        ]
    model = NERModel(
        # "bert",
        # "bert-base-cased",
        # "bert",
        # "bert-base-uncased",
        # "roberta",
        # "roberta-base",
        # "bigbird",
        # "google/bigbird-roberta-base",
        # "longformer",
        # "allenai/longformer-base-4096",
        # "xlm",
        # "xlm-mlm-17-1280",
        # "albert",
        # "albert-base-v1",
        "xlnet",
        "xlnet-base-cased",
        # "layoutlmv2",
        # "layoutlmv2-base-cased",
        labels=custom_labels,
        args={
            "max_seq_length": 128,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "num_train_epochs": 50,
            "train_batch_size": 54,
            "manual_seed": 123,
            # "learning_rate":6e-5,
            "learning_rate": 5e-5,
        },
    )
    print("args=", model.args)
    # # Train the model
    model.train_model(train_df, eval_data=eval_df.values)
    # model.train_model(train_df, eval_data=eval_data)
    print("args=", model.args)
    # # Evaluate the model
    result, model_outputs, predictions = model.eval_model(eval_df)
    print("result", result)

    # Predictions on arbitary text strings
    # sentences = ["Some arbitary sentence", "Simple Transformers sentence"]
    # predictions, raw_outputs = model.predict(sentences)

    # print(predictions)
    # print('model_outputs',model_outputs)
    # print('predictions',predictions)
    print("shape", len(eval_df), np.concatenate(predictions).shape)

    pred = model.predict(eval_df["words"].values)[0]
    # return model,result, pred
    x = []
    y = []
    f = open("pred_result.csv", "w")
    f.write("id,target,prediction\n")
    for i, ii in (eval_df.reset_index()).iterrows():
        # print (ii['id'],ii['labels'],list(pred[i][0].values())[0])
        line = (
            ii["id"]
            + ","
            + ii["labels"]
            + ","
            + list(pred[i][0].values())[0]
            + "\n"
        )
        f.write(line)
        x.append(ii["labels"])
        y.append(list(pred[i][0].values())[0])
    f.close()
    print("Accuracy", accuracy_score(x, y))
    print("F1", f1_score(x, y, average="micro"))

    # n_train = 110521
    # n_test = 12745

    return model, result, pred


if __name__ == "__main__":
    t1 = time.time()
    train_df, eval_df = generate_dataset()
    d = loadjson("mat_scholar_ner.json")
    df = pd.DataFrame(d)
    train_df = df[:110521]
    eval_df = df[-12745:]
    print(train_df)
    print(eval_df)
    model, result, pred = train_model(train_df=train_df, eval_df=eval_df)
    t2 = time.time()
    print("Time taken", t2 - t1)
