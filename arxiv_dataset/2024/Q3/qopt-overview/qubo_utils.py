"""The module contains common auxiliary code for working with QUBOs."""

import json
from glob import glob
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def save_QUBO(Qp, Pp, Const, inst_type, orig_name, orig_file, inst_id,
              filename=None, directory="./instances/QUBO/", comment=None):
    """Saves an instance to JSON.

    Args:
        Qp, Pp, Const: problem parameters,
        inst_type(string): instance type string ("TSP", "MWC", or "UDMWIS")
        orig_name(string): original instance name (e.g., in TSP lib),
        orig_file(string): original instance filename,
        inst_id(string): instance ID,
        filename(string): name for the saved file,
        directory(string): directory for the saved files
        comment(string): optional comment (to be appended to the 'comment' field).

    Notes:
        - Assumed problem form is:

            min (1/2) x' Q x + x' P + C
                (for binary vector x)

        - Naming conventions:
            For IDs: <problem-type><number-within-type>,
            so that we'd have: TSP1, ... TSPn, MWC1, MWC2, ..., MWIS1, ... .

            For filenames: <inst-id>_<size>_<one-word-description>.qubo.json
            So that we'd have, e.g., `TSP0_169_burma14.qubo.json` for instance
            with id="TSP0" of 169 binary vars from original TSP "burma14".

        - the code does *not* check these conventions, though.
    """
    instance = {"Q": Qp.tolist(), "P": Pp.tolist(), "Const":Const}
    instance["description"] = {
        "instance_id": inst_id,
        "instance_type": inst_type,
        "original_instance_name": orig_name,
        "original_instance_file": orig_file,
        "contents": "QUBO",
        "comment": f"Matrix Q, vector P, and Const for QUBO problem. {comment}"
    }

    inst_json = json.dumps(instance, indent=4)
    if filename is None:
        filename = directory + f"{inst_id}_{len(Pp)}_{orig_name}.qubo.json"

    with open(filename, "w") as outfile:
        outfile.write(inst_json)

    return filename


def load_QUBO(filename):
    """Loads a QUBO from `filename`.

    Returns:
        Q (np.array), P (np.array), Const (float), js(dict): problem
        parameters (Q, P, Const) along with the original JSON dictionary.

    """
    with open(filename, 'r') as openfile:
        myjson = json.load(openfile)

    if ("description" not in myjson) or ("contents"
                                         not in myjson["description"]):
        raise ValueError(
            f"{filename}: wrong file format: contents field not found. (Check specification in qubo_utils.save_QUBO)."
        )
    elif myjson["description"]["contents"] != "QUBO":
        raise ValueError(
            f"Wrong file format: 'QUBO' expected, {myjson['contents']} found.(Check specification in qubo_utils.save_QUBO)"
        )

    Q = np.array(myjson['Q'])
    P = np.array(myjson['P'])
    C = float(myjson['Const'])

    return Q, P, C, myjson


def solve_QUBO(Q, P, C, quiet=False, timeout=None):
    """Solves the QUBO specified by matrix Q, vector P, and constant C.

    Returns:
        gurobipy model, dict of variables.
    """
    model = gp.Model("QUBO")
    N = len(Q)
    x = model.addMVar(( N,),vtype=GRB.BINARY)
    model.setObjective(0.5 * x @ Q @ x + P @ x + C, GRB.MINIMIZE)

    if quiet:
        model.setParam("OutputFlag", 0)

    if timeout is not None:
        model.setParam("TimeLimit", timeout)

    model.update()
    model.optimize()
    return model, x

def solve_QUBO_soft_timeout(Q, P, C, gap, soft_timeout, overtime, quiet=False):
    """Solves the QUBO specified by matrix Q, vector P, and constant C.

    Version of the function with the maximum gap parameter. Attempts to solve
    the problem within a soft timeout of ``soft_timeout`` seconds and proceeds with
    another ``overtime`` seconds or until the given MIP Gap ``gap`` is reached.

    Notes:
        Gurobi knowledge base `article <https://support.gurobi.com/hc/en-us/articles/360013419411-How-do-I-set-multiple-termination-criteria-for-a-model>`_.

    Returns:
        gurobipy model, dict of variables.

    """
    model = gp.Model("QUBO")
    N = len(Q)
    x = model.addMVar(( N,),vtype=GRB.BINARY)
    model.setObjective(0.5 * x @ Q @ x + P @ x + C, GRB.MINIMIZE)

    if quiet:
        model.setParam("OutputFlag", 0)

    model.setParam("TimeLimit", soft_timeout)

    # model.update()
    model.optimize()
    if model.status == GRB.TIME_LIMIT:
        if not quiet:
            print(f"Timeout reached.")
        if model.MIPGap > gap:
            if not quiet:
                print(f"Resuming, until gap <= {gap}")

            model.setParam("TimeLimit", overtime)
            model.setParam("MIPGap", gap)
            model.optimize()

    return model, x


def get_QUBO_by_ID(inst_id, folder="./instances/QUBO"):
    """Helper: returns filename given the instance ID and the folder."""
    files = [filename for filename in glob(f"{folder}/{inst_id}_*.qubo.json")]
    if len(files)>1:
        ValueError(f"Several files with ID {inst_id} found in {folder}:\n" + "\n- ".join(files))
        return None
    elif len(files) == 0:
        ValueError(f"No files with ID {inst_id} found in {folder}.")
        return None
    else:
        return files[0]

def get_instance_size_by_ID(inst_id, folder="./instances/QUBO"):
    """Helpers: returns number of binary vars for QUBO given by inst_id."""
    filename = get_QUBO_by_ID(inst_id, folder)
    if filename is None:
        raise ValueError(f"{inst_id}: couldn't find the instance in {folder}.")
        return None
    else:
        with open(filename, 'r') as infile:
            js = json.load(infile)

        return len(js['Q'])

def instance_present_in_folder(inst_id, instance_dir="./instances"):
    """Checks if the original QUBO and instance files are present (by instance id)."""
    QUBO_files = [filename for filename in glob(f"{instance_dir}/QUBO/{inst_id}_*.qubo.json")]
    orig_files = [filename for filename in glob(f"{instance_dir}/orig/{inst_id}_*.orig.json")]

    return ((len(QUBO_files)==1) and (len(orig_files)==1))
