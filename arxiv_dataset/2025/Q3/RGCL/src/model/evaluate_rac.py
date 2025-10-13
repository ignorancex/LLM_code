import sys
sys.path.append("../")

import torch
import torch.nn as nn
import faiss
import numpy as np
from easydict import EasyDict
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from utils.metrics import eval_metrics, compute_metrics_retrieval, evaluate_linear_probe
import wandb
import json
import os
import pickle
import torchmetrics

def iterate_dl(args, dl, classifier):
    # A function to iterate through the dataloader and get all the
    # ids, labels, and predicted labels
    with torch.no_grad():
        ids = []
        for step, batch in enumerate(dl):
            ids.extend(batch["ids"])
            if step == 0:
                labels = batch["labels"].detach().cpu()
                predicted, embed = classifier(batch["image_feats"].to(
                    args.device), batch["text_feats"].to(args.device), return_embed=True)
                predicted = predicted.detach().cpu()
                embed = embed.detach().cpu()

            else:
                labels = torch.cat(
                    (labels, batch["labels"].detach().cpu()), dim=0)
                new_pred, new_embed = classifier(batch["image_feats"].to(
                    args.device), batch["text_feats"].to(args.device), return_embed=True)
                predicted = torch.cat(
                    (predicted, new_pred.detach().cpu()), dim=0)
                embed = torch.cat((embed, new_embed.detach().cpu()), dim=0)
    return ids, labels, predicted, embed



def retrieve_evaluate_RAC(
    train_dl, evaluate_dl, model, largest_retrieval=100, threshold=0.5, args=None, eval_name=None, epoch=None,
):
    model.eval()
    # Get the features and labels
    train_ids = []

    pickle_dict = EasyDict()
    if epoch != None:
        epoch_name = "epoch_"+str(epoch)
    else:
        epoch_name = "final_eval"
    pickle_save_path = os.path.join(
        args.output_path, eval_name+epoch_name+"_retrieval_logging_dict.pkl")

    for i, batch in enumerate(train_dl):
        train_ids.extend(batch["ids"])
        _, all_feats = model(batch["image_feats"].to(
            args.device), batch["text_feats"].to(args.device), return_embed=True)
        if i == 0:
            if args.Faiss_GPU:
                train_feats = all_feats
                train_labels = batch["labels"]
            else:
                train_feats = all_feats.cpu().detach().numpy().astype("float32")
                train_labels = batch["labels"].cpu(
                ).detach().numpy().astype("int")
        else:

            if args.Faiss_GPU:
                # For GPU implementation
                train_feats = torch.cat((train_feats, all_feats), dim=0)
                train_labels = torch.cat(
                    (train_labels, batch["labels"]), dim=0)
            else:
                # For cpu implementation
                train_feats = np.concatenate(
                    (train_feats, all_feats.cpu().detach().numpy().astype("float32")))
                train_labels = np.concatenate(
                    (train_labels, batch["labels"].cpu().detach().numpy().astype("int")))

    """print("train_feats.shape: ", train_feats.shape)
    print("train_labels.shape: ", train_labels.shape)
    print("train_ids.shape: ", len(train_ids))"""

    evaluate_ids = []
    evaluate_feats = np.array([[]])
    evaluate_labels = np.array([[]])

    for i, batch in enumerate(evaluate_dl):
        evaluate_ids.extend(batch["ids"])
        _, all_feats = model(batch["image_feats"].to(
            args.device), batch["text_feats"].to(args.device), return_embed=True)
        if i == 0:

            if args.Faiss_GPU:
                evaluate_feats = all_feats
                evaluate_labels = batch["labels"]
            else:
                evaluate_feats = all_feats.cpu().detach().numpy().astype("float32")
                evaluate_labels = batch["labels"].cpu(
                ).detach().numpy().astype("int")
        else:
            if args.Faiss_GPU:
                evaluate_feats = torch.cat((evaluate_feats, all_feats), dim=0)
                evaluate_labels = torch.cat(
                    (evaluate_labels, batch["labels"]), dim=0)
            else:
                evaluate_feats = np.concatenate(
                    (evaluate_feats, all_feats.cpu().detach().numpy().astype("float32")))
                evaluate_labels = np.concatenate(
                    (evaluate_labels, batch["labels"].cpu().detach().numpy().astype("int")))

    # Perform dense retrieval
    # faiss.normalize_L2(train_feats)
    # faiss.normalize_L2(evaluate_feats)
    """pickle_dict["train_feats"] = train_feats if not args.Faiss_GPU else train_feats.cpu(
    ).detach().numpy().astype("float32")
    pickle_dict["train_labels"] = train_labels if not args.Faiss_GPU else train_labels.cpu(
    ).detach().numpy().astype("int")
    pickle_dict["train_ids"] = train_ids

    pickle_dict["evaluate_feats"] = evaluate_feats if not args.Faiss_GPU else evaluate_feats.cpu(
    ).detach().numpy().astype("float32")
    pickle_dict["evaluate_labels"] = evaluate_labels if not args.Faiss_GPU else evaluate_labels.cpu(
    ).detach().numpy().astype("int")
    pickle_dict["evaluate_ids"] = evaluate_ids"""

    # Get the dimension of the features
    dim = all_feats.shape[1]
    # Initialize the index
    # For different loss functions, we need to change the index type
    if args.metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    if args.Faiss_GPU:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        if args.metric != "ip":
            train_feats = torch.nn.functional.normalize(
                train_feats, p=2, dim=1)
            evaluate_feats = torch.nn.functional.normalize(
                evaluate_feats, p=2, dim=1)

    else:
        if args.metric != "ip":
            faiss.normalize_L2(train_feats)
            faiss.normalize_L2(evaluate_feats)
    index.add(train_feats)
    D, I = index.search(evaluate_feats, largest_retrieval)

    # pickle_dict["train_feats_normalized"] = train_feats if not args.Faiss_GPU else train_feats.cpu().detach().numpy().astype("float32")
    # pickle_dict["evaluate_feats_normalized"] = evaluate_feats if not args.Faiss_GPU else evaluate_feats.cpu().detach().numpy().astype("float32")

    logging_dict = EasyDict()
    for i, row in enumerate(D):
        # a list to record the ids of the retrieved example
        retrieved_ids = []
        # a list to record the similarity scores of the retrieved example
        retrieved_scores = []
        # a list to record the retrieved example's label
        retrieved_label = []
        for j, value in enumerate(row):
            # You have to retrieve at least one, no matter what the similarity score is
            if j == 0:
                retrieved_ids.append(train_ids[I[i, j]])
                retrieved_scores.append(value)
                retrieved_label.append(train_labels[I[i, j]].item())
            # if image is similar
            else:
                if value < threshold or threshold == -1:
                    # for the temp list, we use the image ids rather than the ordered number
                    retrieved_ids.append(train_ids[I[i, j]])
                    retrieved_scores.append(value)
                    retrieved_label.append(train_labels[I[i, j]].item())
                # if larger than threshold,
                # then we can break the inside loop,
                # since the rest of the values are larger than the threshold
                else:
                    break
        # Record the number of images retrieved for each query
        no_retrieved = len(retrieved_ids)

        logging_dict[evaluate_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
        }
    pickle_dict["logging_dict"] = logging_dict
    if args.save_embed:
        with open(pickle_save_path, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #print("pickle of evaluation results saved to: ", pickle_save_path)

    return logging_dict, evaluate_labels


def final_evaluation(
    train_dl, dev_dl, model, args, artifact, test_seen_dl=None, test_unseen_dl=None,
):
    # create a wandb.Table() with corresponding columns
    logging_columns = [
        "id",
        "ground_truth",
        "image",
        "retrieved_images",
        "no_retrieved",
        "retrieved_ids",
        "retrieved_scores",
        "retrieved_labels",
    ]
    metrics_table = wandb.Table(
        columns=["split", "acc", "roc", "pre", "recall", "f1"])
    eval_name_list = ["dev"]
    eval_dl_list = [dev_dl]
    if test_seen_dl != None:
        eval_name_list.append("test_seen")
        eval_dl_list.append(test_seen_dl)
    if test_unseen_dl != None:
        eval_name_list.append("test_unseen")
        eval_dl_list.append(test_unseen_dl)
    for eval_name, eval_dl in zip(
        eval_name_list, eval_dl_list
    ):
        logging_dict, eval_labels = retrieve_evaluate_RAC(
            train_dl, eval_dl, model, largest_retrieval=args.topk, threshold=args.similarity_threshold,
            args=args, eval_name=eval_name)
        acc, roc, pre, recall, f1, _, _ = compute_metrics_retrieval(
            logging_dict, eval_labels, majority_voting=args.majority_voting, topk=args.topk
        )
        metrics_table.add_data(
            eval_name, acc, roc, pre, recall, f1)
        print("Final Evaluation {}: acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
            eval_name, acc, roc, pre, recall, f1))

        logging_table = wandb.Table(columns=logging_columns)
        os.makedirs("{}/{}/".format(args.output_path,
                    args.dataset), exist_ok=True)
        with open(
            "{}/{}/Fusion_{}_Threshold_{}_topK_{}_{}_{}.json".format(
                args.output_path,
                args.dataset,
                args.fusion_mode,
                args.similarity_threshold,
                args.topk,
                args.model,
                eval_name,
            ),
            "w",
        ) as f:
            for (key, value), label in zip(logging_dict.items(), eval_labels):
                # The values in the logging dict contains:
                # "no_retrieved"
                # "retrieved_ids"
                # "retrieved_scores"
                # "retrieved_label"

                value_list = value.values()

                # Add the data to the wandb table
                logging_table.add_data(
                    key,
                    label,
                    artifact.get("{}.png".format(key)),
                    [
                        artifact.get("{}.png".format(
                            value["retrieved_ids"][i]))
                        for i in range(len(value["retrieved_ids"]))
                    ],
                    *value_list,  # unpack the list,
                )

                # Change the type of the float values in the logging dict to string
                # so that it can be dumped into json file

                logging_dict[key]["retrieved_scores"] = str(
                    logging_dict[key]["retrieved_scores"]
                )

                json.dump([key, logging_dict[key]], f)

                f.write("\n")

        wandb.log({"logging_table_{}".format(eval_name): logging_table})
        wandb.log({"Final_metrics_table": metrics_table})

    return None

# Construct the linear probe model for evaluation
# A simple one layer logistic regression


def retrieve_evaluate_RAC_(
    train_dl, evaluate_dl, model, largest_retrieval=100, threshold=0.5, args=None, eval_name=None, epoch=None,
):
    model.eval()
    # Get the features and labels
    train_ids = []
    

    if epoch != None:
        epoch_name = "epoch_"+str(epoch)
    else:
        epoch_name = "final_eval"
    pickle_save_path = os.path.join(
        args.output_path, eval_name+epoch_name+"_retrieval_logging_dict.pkl")
    pickle_dict = EasyDict()
    if type(train_dl) == list:
        train_dl_rest = train_dl[1:]
        train_dl = train_dl[0]
        train_dl_is_list = True
    else:
        train_dl_is_list = False
    for i, batch in enumerate(train_dl):
        train_ids.extend(batch["ids"])
        out, all_feats = model(batch["image_feats"].to(
            'cuda'), batch["text_feats"].to('cuda'), return_embed=True)
        if i == 0:

            train_feats = all_feats
            train_labels = batch["labels"]
            train_out = out
        else:


            # For GPU implementation
            train_feats = torch.cat((train_feats, all_feats), dim=0)
            train_labels = torch.cat(
                (train_labels, batch["labels"]), dim=0)
            train_out = torch.cat((train_out, out), dim=0)
    if train_dl_is_list:
        for train_dl_ in train_dl_rest:
            for batch in train_dl_:
                train_ids.extend(batch["ids"])
                out, all_feats = model(batch["image_feats"].to(
                    'cuda'), batch["text_feats"].to('cuda'), return_embed=True)

                # For GPU implementation
                train_feats = torch.cat((train_feats, all_feats), dim=0)
                train_labels = torch.cat(
                    (train_labels, batch["labels"]), dim=0)
                train_out = torch.cat((train_out, out), dim=0)
    """print("train_feats.shape: ", train_feats.shape)
    print("train_labels.shape: ", train_labels.shape)
    print("train_ids.shape: ", len(train_ids))"""

    evaluate_ids = []
    evaluate_feats = np.array([[]])
    evaluate_labels = np.array([[]])
    for i, batch in enumerate(evaluate_dl):
        evaluate_ids.extend(batch["ids"])
        out, all_feats = model(batch["image_feats"].to(
            'cuda'), batch["text_feats"].to('cuda'), return_embed=True)
        if i == 0:

            evaluate_feats = all_feats
            evaluate_labels = batch["labels"]
            eval_out = out
        else:

            evaluate_feats = torch.cat((evaluate_feats, all_feats), dim=0)
            evaluate_labels = torch.cat(
                (evaluate_labels, batch["labels"]), dim=0)
            eval_out = torch.cat((eval_out, out), dim=0)

    # Get the dimension of the features
    dim = all_feats.shape[1]
    # Initialize the index
    # For different loss functions, we need to change the index type
    index = faiss.IndexFlatIP(dim)

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    train_feats = torch.nn.functional.normalize(
        train_feats, p=2, dim=1)
    evaluate_feats = torch.nn.functional.normalize(
        evaluate_feats, p=2, dim=1)
    index.add(train_feats)
    D, I = index.search(evaluate_feats, largest_retrieval)

    # pickle_dict["train_feats_normalized"] = train_feats if not args.Faiss_GPU else train_feats.cpu().detach().numpy().astype("float32")
    # pickle_dict["evaluate_feats_normalized"] = evaluate_feats if not args.Faiss_GPU else evaluate_feats.cpu().detach().numpy().astype("float32")

    logging_dict = EasyDict()
    for i, row in enumerate(D):
        # a list to record the ids of the retrieved example
        retrieved_ids = []
        # a list to record the similarity scores of the retrieved example
        retrieved_scores = []
        # a list to record the retrieved example's label
        retrieved_label = []
        retrieved_out = []
        for j, value in enumerate(row):
            # You have to retrieve at least one, no matter what the similarity score is
            if j == 0:
                retrieved_ids.append(train_ids[I[i, j]])
                retrieved_scores.append(value)
                retrieved_label.append(train_labels[I[i, j]].item())
                retrieved_out.append(train_out[I[i, j]].cpu().detach())
            # if image is similar
            else:
                if value < threshold or threshold == -1:
                    # for the temp list, we use the image ids rather than the ordered number
                    retrieved_ids.append(train_ids[I[i, j]])
                    retrieved_scores.append(value)
                    retrieved_label.append(train_labels[I[i, j]].item())
                    retrieved_out.append(train_out[I[i, j]].cpu().detach())
                # if larger than threshold,
                # then we can break the inside loop,
                # since the rest of the values are larger than the threshold
                else:
                    break
        # Record the number of images retrieved for each query
        no_retrieved = len(retrieved_ids)

        logging_dict[evaluate_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
            "retrieved_out": torch.cat(retrieved_out),
            "eval_out": eval_out[i].cpu().detach(),
        }
    pickle_dict["logging_dict"] = logging_dict
    if args.save_embed:
        with open(pickle_save_path, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #print("pickle of evaluation results saved to: ", pickle_save_path)


    return logging_dict, evaluate_labels