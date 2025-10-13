import ast
import copy
import pydoc

import torch

PYTORCH_IMPORTS = """
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch as th
import torch.linalg as la
from torch.nn import Parameter
import torch.linalg as linalg
"""


# model level
def model_to_cuda(model: torch.nn.Module, inputs: list[torch.Tensor]):
    model_cuda = copy.deepcopy(model)
    for attr in dir(model_cuda):
        if isinstance(getattr(model_cuda, attr), torch.Tensor):
            setattr(model_cuda, attr, getattr(model_cuda, attr).to("cuda"))
    model_cuda.to("cuda")
    new_inputs = []
    to_kwargs = {"device": "cuda"}
    for x in inputs:
        if not isinstance(x, torch.Tensor):
            new_inputs += [x]
            continue
        new_inputs.append(x.clone().to(**to_kwargs))
    return model_cuda, new_inputs


# api level
def titanfuzz_to_cuda(py_code: str) -> str:
    tree = ast.parse(py_code)
    transfer_code = ""

    # print(ast.dump(tree, indent=4)) # debug
    special_func = {
        "view",
        "reshape",
        "squeeze",
        "unsqueeze",
        "transpose",
        "permute",
        "flatten",
        "contiguous",
        "expand",
        "repeat",
        "view_as",
    }
    for node in tree.body:
        code = ast.unparse(node)
        flag = True
        for special_func_name in special_func:
            if f".{special_func_name}(" in code:
                transfer_code += code + ".cuda()\n"
                flag = False
                break
        if not flag:
            continue
        try:
            if (
                isinstance(node, ast.Expr)
                or isinstance(node, ast.Assign)
                or isinstance(node, ast.AugAssign)
            ):
                if isinstance(node.value, ast.Call):

                    def dfs(x_node):
                        try:
                            _ = x_node.value
                            return dfs(x_node.value) + "." + x_node.attr
                        except KeyError:
                            return "." + x_node.id

                    func_name = dfs(node.value.func)[1:]
                    if "torch." in func_name:
                        def_info = pydoc.render_doc(func_name).split("\n")[3]
                        if "device" in def_info:
                            code += ".cuda()"
        except Exception as e:
            print(e)

        transfer_code += code + "\n"

    return transfer_code


def titanfuzz_to_compile(py_code: str) -> str:
    code = "def func():\n"
    for line in py_code.split("\n"):
        code += f"    {line}\n"

    code += "compile_func = torch.compile(func)\n"
    code += "compile_func()\n"
    return code


if __name__ == "__main__":
    py_code = """
input_data = torch.arange(1, 17, dtype=torch.float).view(4, 4)
output_data = torch.triu_indices(row=4, col=4, offset=0)


input_data = torch.arange(1, 17, dtype=torch.float).view(4, 4)
output_data = torch.triu_indices(row=4, col=4, offset=0)
    """
    input_data = torch.arange(1, 17, dtype=torch.float).view(4, 4).cuda()
    print(titanfuzz_to_cuda(py_code))
