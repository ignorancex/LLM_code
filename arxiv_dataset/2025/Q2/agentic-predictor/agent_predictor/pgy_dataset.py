import os
import sys
import torch
import json
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from scripts.optimize.utils.embedding import get_encoder
from tqdm import tqdm


def get_mv_graph(data, domain):
    
    WORKFLOW_TEMPLATE = """from typing import Literal
    import workplace.{domain}.workflows.template.operator as operator
    import workplace.{domain}.workflows.round_{round}.prompt as prompt_custom
    from metagpt.provider.llm_provider_registry import create_llm_instance
    from metagpt.utils.cost_manager import CostManager

    DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

    {graph}
    """

    TEST_PROMPT = "Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution"
    TEST_OPS = """test_response = await self.test(problem={problem}, solution={solution}, entry_point={entry_point})"""
    TEST_INIT = """self.test = operator.Test(self.llm)"""

    ANSWER_GENERATE_PROMPT = "Think step by step and solve the problem"
    ANSWER_GENERATE_OPS = (
        """answer_response = await self.answer_generate(input={input})"""
    )
    ANSWER_GENERATE_INIT = (
        """self.answer_generate = operator.AnswerGenerate(self.llm)"""
    )

    CUSTOM_CODE_PROMPT = "Fill CodeBlock"
    CUSTOM_CODE_OPS = """code_response = await self.custom_code_generate(problem={problem}, entry_point={entry_point}, instruction={instruction})"""
    CUSTOM_CODE_INIT = (
        """self.custom_code_generate = operator.CustomCodeGenerate(self.llm)"""
    )

    SC_PROMPT = "Several answers have been generated to a same question"
    SC_OPS = """ensemble_response = await self.sc_ensemble(solutions={solutions})"""
    SC_INIT = """self.sc_ensemble = operator.ScEnsemble(self.llm)"""

    SC_EXT_PROMPT = "Several answers have been generated to a same question"
    SC_EXT_OPS = """ensemble_response = await self.sc_ensemble(solutions={solutions}, problem={problem})"""
    SC_EXT_INIT = """self.sc_ensemble = operator.ScEnsemble(self.llm)"""

    PROGRAMMER_PROMPT = "You are a professional Python programmer"
    PROGRAMMER_OPS = """programmer_response = await self.programmer(problem={problem}, analysis={analysis})"""
    PROGRAMMER_INIT = """self.programmer = operator.Programmer(self.llm)"""

    if domain not in ["CodingAF", "MathAF"]:
        AGENT_PROMPTS = [
            TEST_PROMPT,
            ANSWER_GENERATE_PROMPT,
            CUSTOM_CODE_PROMPT,
            SC_PROMPT,
            PROGRAMMER_PROMPT,
        ]

        AGENT_INIT = [
            TEST_INIT,
            ANSWER_GENERATE_INIT,
            CUSTOM_CODE_INIT,
            SC_INIT,
            PROGRAMMER_INIT,
        ]

        AGENT_OPS = [
            "test_response",
            "answer_response",
            "code_response",
            "ensemble_response",
            "programmer_response",
        ]

        OPS_CLASS = [
            "Test.py",
            "AnswerGenerate.py",
            "CustomCodeGenerate.py",
            "ScEnsembleExt.py",
            "Programmer.py",
        ]

        OPS_DESC = [
            "Tests the solution using public test cases. If the solution fails, it reflects on the errors and attempts to modify the solution. Returns True and the solution if all tests pass after modifications. Returns False and the current solution if it still fails after modifications.",
            "Generate step by step based on the input. The step by step thought process is in the field of 'thought', and the final answer is in the field of 'answer'.",
            "Generates code based on customized input and instruction.",
            "Uses self-consistency to select the solution that appears most frequently in the solution list, improve the selection to enhance the choice of the best solution.",
            "Automatically writes, executes Python code, and returns the solution based on the provided problem description and analysis. The `output` only contains the final answer. If you want to see the detailed solution process, it's recommended to retrieve the `code`.",
        ]
    else:
        AGENT_PROMPTS = [
            TEST_PROMPT,
            ANSWER_GENERATE_PROMPT,
            CUSTOM_CODE_PROMPT,
            SC_EXT_PROMPT,
            PROGRAMMER_PROMPT,
        ]

        AGENT_INIT = [
            TEST_INIT,
            ANSWER_GENERATE_INIT,
            CUSTOM_CODE_INIT,
            SC_EXT_INIT,
            PROGRAMMER_INIT,
        ]

        AGENT_OPS = [
            "test_response",
            "answer_response",
            "code_response",
            "ensemble_response",
            "programmer_response",
        ]

        OPS_CLASS = [
            "Test.py",
            "AnswerGenerate.py",
            "CustomCodeGenerate.py",
            "ScEnsemble.py",
            "Programmer.py",
        ]

        OPS_DESC = [
            "Tests the solution using public test cases. If the solution fails, it reflects on the errors and attempts to modify the solution. Returns True and the solution if all tests pass after modifications. Returns False and the current solution if it still fails after modifications.",
            "Generate step by step based on the input. The step by step thought process is in the field of 'thought', and the final answer is in the field of 'answer'.",
            "Generates code based on customized input and instruction.",
            "Uses self-consistency to select the solution that appears most frequently in the solution list, improve the selection to enhance the choice of the best solution.",
            "Automatically writes, executes Python code, and returns the solution based on the provided problem description and analysis. The `output` only contains the final answer. If you want to see the detailed solution process, it's recommended to retrieve the `code`.",
        ]
    
    nodes, edges = data["nodes"], data["edge_index"]
    # View 2: Individual Operator Code [x] -> ours
    data["operator_nodes"] = {idx: "" for idx in nodes}
    # View 3: Global System Prompt + Operator Descriptions [x] -> ours
    data["full_prompts"] = ""
    with open("agent_predictor/empty_workflow.py") as f:
        empty_graph = f.read()
    
    init_code = ""
    init_ops = []
    implement_code = ""
    implement_ops = [""]
    for idx, inst in nodes.items():
        op = None
        # load operator file for EACH node as in the system prompt
        if idx == 0:
            init_code = WORKFLOW_TEMPLATE.format(
                graph=empty_graph, round=1, domain=domain
            )
            if domain == "CodingAF":
                implement_code = (
                    "async def __call__(self, problem: str, entry_point: str):\n"
                )
            else:
                implement_code = "async def __call__(self, problem: str):\n"
        else:
            try:
                # predefined agents
                agent = AGENT_PROMPTS.index(inst.strip().split(".")[0].strip())
                init = AGENT_INIT[agent]
                init_ops.append(f"\t{init}")
                ops = AGENT_OPS[agent]
                implement_ops.append(ops)
                data[
                    "full_prompts"
                ] += f"{OPS_CLASS[agent].split('.')[0]} Agent: {OPS_DESC[agent]}\n"
                with open("agent_predictor/operator_files/" + OPS_CLASS[agent]) as f:
                    data["operator_nodes"][int(idx)] = f.read()
            except:
                # custom agents
                data[
                    "full_prompts"
                ] += "Custom Agent: Generates anything based on customized input and instruction.\n"
                init_ops.append("\tself.custom = operator.Custom(self.llm)")
                implement_ops.append("""custom_response""")
                with open("agent_predictor/operator_files/Custom.py") as f:
                    data["operator_nodes"][int(idx)] = f.read()
    # View 5: Individual Workflow Code [x] -> ours                    
    implement_flows = {idx: "\t# Implementation of the workflow" for idx in nodes}

    for idx, edge in enumerate(edges):
        inp, res = edge
        in_ops_type = implement_ops[inp].strip()
        ops_type = implement_ops[res].strip()
        if inp > 0:
            if ops_type == "test_response":
                implement_flows[int(res)] = "\t" + TEST_OPS.format(
                    problem="problem",
                    solution=f"{in_ops_type}['response']",
                    entry_point="entry_point",
                )
            elif ops_type == "answer_response":
                implement_flows[int(res)] = "\t" + ANSWER_GENERATE_OPS.format(
                    input=f"{in_ops_type}['response']"
                )
            elif ops_type == "code_response":
                implement_flows[int(res)] = "\t" + CUSTOM_CODE_OPS.format(
                    problem="problem",
                    entry_point="entry_point",
                    instruction=f'"{nodes[int(res)].strip()}"',
                )
            elif ops_type == "ensemble_response" and domain in ["MathAF", "CodingAF"]:
                implement_flows[int(res)] = "\t" + SC_EXT_OPS.format(
                    solutions="candidates", problem="problem"
                )
            elif ops_type == "ensemble_response":
                implement_flows[int(res)] = "\t" + SC_OPS.format(solutions="candidates")
            elif ops_type == "programmer_response":
                implement_flows[int(res)] = "\t" + PROGRAMMER_OPS.format(
                    problem="problem", analysis=f"{in_ops_type}['response']"
                )
            else:
                implement_flows[int(res)] = (
                    f"\t{ops_type} = await self.custom(input={in_ops_type}['response'], instruction='{nodes[int(res)].strip()}')"
                )
        else:
            if ops_type == "test_response":
                implement_flows[int(res)] = "\t" + TEST_OPS.format(
                    problem="problem", solution="problem", entry_point="entry_point"
                )
            elif ops_type == "answer_response":
                implement_flows[int(res)] = "\t" + ANSWER_GENERATE_OPS.format(
                    input="problem"
                )
            elif ops_type == "code_response":
                implement_flows[int(res)] = "\t" + CUSTOM_CODE_OPS.format(
                    problem="problem",
                    entry_point="entry_point",
                    instruction=f'"{nodes[int(res)].strip()}"',
                )
            elif ops_type == "ensemble_response" and domain in ["MathAF", "CodingAF"]:
                implement_flows[int(res)] = "\t" + SC_EXT_OPS.format(
                    solutions="problem", problem="problem"
                )
            elif ops_type == "ensemble_response":
                implement_flows[int(res)] = "\t" + SC_OPS.format(solutions="problem")
            elif ops_type == "programmer_response":
                implement_flows[int(res)] = "\t" + PROGRAMMER_OPS.format(problem="problem", analysis="problem")
            else:
                implement_flows[int(res)] = (f"\t{ops_type} = await self.custom(input=problem, instruction='{nodes[int(res)].strip()}')")
        if idx == len(edges) - 1:
            last_node = list(implement_flows.keys())[-1]
            ops_type = implement_flows[last_node].split("=")[0].strip()
            implement_flows[last_node] = implement_flows[last_node].replace(ops_type, "solution", 1)
    # init_code += "\n".join(list(set(init_ops)))
    # implement_code += "\n".join(implement_flows.values())
    # workflow_code = f"""{init_code}

    # {implement_code}
    # \treturn solution['response'], self.llm.cost_manager.total_cost"""
    # data["workflow_code"] = workflow_code  # View 4: Global Workflow Code [x] -> ours
    data["code_nodes"] = implement_flows

    return data


def get_dataloader(args, branches):
    loaders = []
    for branch in branches:
        jsonl_path = args.data_path + f"/{branch}.jsonl"
        dataset = CustomMultiViewDataset(
            root=f"{args.data_path}/{branch}", jsonl_path=jsonl_path, brach=branch
        )
        print(f"length of {branch} dataset: {len(dataset)}")
        # if dataset[0].edge_index.dtype != torch.long:
        #     process_pyg_dataset(dataset)
        loader = DataLoader(
            dataset.data_list,
            batch_size=args.batch_size,
            shuffle=True if "train" in "branch" else False,
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


class CustomMultiViewDataset(InMemoryDataset):
    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        raw_data_list=None,
        encoder=None,
        memory=None,
    ):
        # self.jsonl_path = jsonl_path
        # self.branch = branch
        self.raw_data_list = raw_data_list
        self.encoder = encoder
        self.memory = memory
        # if not os.path.exists(self.processed_paths[0]):
        #     self.encoder = get_encoder('llama3_1_8b', cache_dir="./model", batch_size=1)
        super(CustomMultiViewDataset, self).__init__(root, transform, pre_transform)
        self.data_list = self.raw_process(raw_data_list, memory)

        # self.data_list = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def processed_file_names(self):
        return "data.pt"

    def raw_process(self, raw_data_list, memory):
        text_encoder = self.encoder["text_encoder"]
        code_encoder = self.encoder["code_encoder"]

        data_list = []

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
            )

            data.edge_index = data.edge_index.type(torch.long)
            data_list.append(data)
        return data_list

    def process(self):
        pass

    def len(self):
        return len(self.data_list)


if __name__ == "__main__":

    # for branch in ['train','task_ood_workflow_id','task_id_workflow_ood','task_ood_workflow_ood']:
    #     root = f"/home/yuanshuozhang/GDesigner-3063/workflowbench/dataset/mmlu_aflow_2400_20_4o_mini_12_16"
    #     jsonl_path = root + f"/{branch}.jsonl"
    #     root += f"/{branch}"
    for branch in ["train"]:
        root = "experiments/MMLU"
        jsonl_path = root + f"/{branch}.jsonl"
        root = root + f"/{branch}"
        dataset = CustomMultiViewDataset(
            root=root, jsonl_path=jsonl_path, branch=branch
        )
    # branch = "composed"
    # root = f"/home/yuanshuozhang/GDesigner-3063/workflowbench/dataset/mmlu_aflow_2400_20_4o_mini_12_16"
    # jsonl_path = root + f"/dataset.jsonl"
    # root += f"/{branch}"
    # dataset = CustomMultiViewDataset(root=root, jsonl_path=jsonl_path,branch=branch)
