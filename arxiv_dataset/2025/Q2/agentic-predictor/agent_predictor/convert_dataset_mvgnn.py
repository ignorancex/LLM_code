import os
import sys
import torch
import json
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import f1_score

from utils.file_io import read_jsonl_file
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from utils.configs import OPENAI_KEY

from tqdm import tqdm
import time
import pickle


def get_pretrain_dataloader(args):
    data_list = []
    branch = "train"
    jsonl_path = args.data_path + f"/{branch}.jsonl"
    for domain in ["CodingAF", "MathAF", "ReasonAF"]:
        data_path = f"agent_predictor/mv_datasets/{domain}"
        dataset = CustomGraphDataset(
            root=f"{data_path}/{branch}", jsonl_path=jsonl_path, branch=branch
        )
        data_list = data_list + dataset.data_list[: int(len(dataset.data_list) * 0.5)]
    print(f"length of {branch} dataset: {len(data_list)}")

    return DataLoader(data_list, batch_size=args.batch_size, shuffle=True)


def get_dataloader(args, branches):
    loaders = []
    for branch in branches:
        jsonl_path = args.data_path + f"/{branch}.jsonl"
        dataset = CustomGraphDataset(
            root=f"{args.data_path}/{branch}", jsonl_path=jsonl_path, branch=branch
        )
        if "train" in branch:
            dataset.data_list = dataset.data_list[
                : int(len(dataset.data_list) * args.label_ratio)
            ]
        print(f"length of {branch} dataset: {len(dataset.data_list)}")
        loader = DataLoader(
            dataset.data_list,
            batch_size=args.batch_size,
            shuffle=True if "train" in branch else False,
        )
        loaders.append(loader)
    return loaders


def process_pyg_dataset(dataset):
    for data in dataset:
        data.edge_index = data.edge_index.type(torch.long)


def get_numerical_node_id(nodes: dict, node_id: str):
    return list(nodes.keys()).index(node_id)


def get_numerical_edge_index(nodes, str_edge_index):
    edge_index = []
    for edge in str_edge_index:
        # import pdb;pdb.set_trace()
        edge_index.append(
            [
                get_numerical_node_id(nodes, edge[0]),
                get_numerical_node_id(nodes, edge[1]),
            ]
        )
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index


def construct_node_embedding(workflow: dict, node_attributes_memory: dict, encoder):
    features = []
    for node_id in workflow["nodes"].keys():
        prompt = workflow["nodes"][node_id]
        try:
            feature = node_attributes_memory[prompt]
        except:
            feature = encoder.encode(prompt)
            node_attributes_memory[prompt] = feature
        features.append(feature.reshape(-1))
    features = torch.stack(features)
    return features, node_attributes_memory


class CustomGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        jsonl_path,
        branch,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.jsonl_path = jsonl_path
        self.branch = branch
        super(CustomGraphDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data_list = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):

        # text_encoder = SentenceTransformer("all-mpnet-base-v2")
        text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        code_encoder = SentenceTransformer(
            "nomic-ai/CodeRankEmbed", trust_remote_code=True
        )

        data_list = []

        # graph (node features with edges): prompts, operator code blocks, implementation function calls
        # text (full instruction prompt): full_prompts
        # code (full workflow code): workflow_code
        # task embedding: task
        # process into multi-view data loaders
        print(f"Convert textual graph to PyG data from {self.jsonl_path}")

        raw_data_list = read_jsonl_file(self.jsonl_path)  # [:10]
        raw_tasks, raw_instructions, raw_workflows = [], [], []
        for workflow in raw_data_list:
            raw_tasks.append(workflow["task"])
            raw_instructions.append(workflow["full_prompts"])
            raw_workflows.append(workflow["workflow_code"])

        task_embeddings = text_encoder.encode(raw_tasks, convert_to_numpy=False)
        instruction_texts = text_encoder.encode(
            raw_instructions, convert_to_numpy=False
        )
        workflow_codes = code_encoder.encode(raw_workflows, convert_to_numpy=False)

        for idx, workflow in enumerate(
            tqdm(raw_data_list, desc="processing workflow data")
        ):
            edge_index = torch.tensor(workflow["edge_index"]).t().contiguous()

            raw_graph_prompts, raw_graph_ops, raw_graph_codes = [], [], []

            for node_id in workflow["nodes"].keys():
                if workflow["operator_nodes"][node_id] == "":
                    workflow["operator_nodes"][
                        node_id
                    ] = "# This is the first line comment"

                raw_graph_prompts.append(workflow["nodes"][node_id])
                raw_graph_ops.append(workflow["operator_nodes"][node_id])
                raw_graph_codes.append(workflow["code_nodes"][node_id])

            graph_prompts = torch.stack(
                text_encoder.encode(raw_graph_prompts, convert_to_numpy=False)
            )
            graph_ops = torch.stack(
                code_encoder.encode(raw_graph_ops, convert_to_numpy=False)
            )
            graph_codes = torch.stack(
                code_encoder.encode(raw_graph_codes, convert_to_numpy=False)
            )

            num_nodes = len(workflow["nodes"])
            y = torch.tensor(workflow["label"], dtype=torch.long)
            data = Data(
                # default sitting for num_nodes infer
                x=graph_prompts,
                # multi-graph views
                graph_prompts=graph_prompts,
                graph_ops=graph_ops,
                graph_codes=graph_codes,
                # global-context views
                instruction_text=instruction_texts[idx],
                workflow_code=workflow_codes[idx],
                # task-context view
                task_embedding=task_embeddings[idx],
                edge_index=edge_index,
                y=y,
                workflow_id=workflow["workflow_id"],
                task_id=workflow["task_id"],
            )

            data.edge_index = data.edge_index.type(torch.long)
            data_list.append(data)

        raw_data_list = None
        torch.save(data_list, self.processed_paths[0])

    def len(self):
        return len(self.data_list)
