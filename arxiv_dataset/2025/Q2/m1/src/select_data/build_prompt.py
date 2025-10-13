"""
Medical Subject Headings
https://www.nlm.nih.gov/mesh/meshhome.html

MeSH Qualifiers List
https://www.nlm.nih.gov/mesh/subhierarchy.html

MeSH Qualifiers with Scope Notes
https://www.nlm.nih.gov/mesh/qualifiers_scopenotes.html


Follow s1: https://github.com/simplescaling/s1/blob/main/data/featurization.py
"""

import json
import string
from collections import OrderedDict
from pathlib import Path

import pandas as pd

qualifier_scope_note_path = "src/select_data/qualifier_scope_note.tsv"

qualifier_scope_note_df = pd.read_csv(
    qualifier_scope_note_path,
    sep="\t",
)

qualifier_scope_note = (
    qualifier_scope_note_df[["Name", "Scope Notes"]]
    .set_index("Name")
    .to_dict()["Scope Notes"]
)


mesh_qualifier_prompt = []
for idx, (mesh_qualifier, scope_note) in enumerate(qualifier_scope_note.items()):
    code = f"{idx:02d}"
    title = mesh_qualifier
    description = scope_note
    prompt = f"### {title}\n* Code: {code}\n* Description: {description}"

    mesh_qualifier_prompt.append(
        {
            "code": code,
            "title": title,
            "description": description,
            "prompt": prompt,
        }
    )

with open("src/select_data/mesh_qualifier.json", "w") as f:
    json.dump(mesh_qualifier_prompt, f, indent=2)


raw_mesh_qualifier_list_path = Path("src/select_data/raw_mesh_qualifer_list.txt")


qualifier_hierarchy = dict()
current_qualifer = None
with open(raw_mesh_qualifier_list_path, "r") as f:
    for line in f.readlines():
        if len(line.strip()) == 0:
            continue
        if not line.strip().startswith("-"):
            current_qualifer = line.strip()
            qualifier_hierarchy[current_qualifer] = []
        else:
            if current_qualifer is None:
                raise ValueError("First line must be a qualifier")
            qualifier_hierarchy[current_qualifer].append(line.rstrip())

_qualifier_hierarchy = {}
for k, v in qualifier_hierarchy.items():
    k = string.capwords(k)
    if len(v) == 0:
        v = "  - Null"
    else:
        v = "\n".join(v)
        v = string.capwords(v)
    _qualifier_hierarchy[k] = v

qualifier_hierarchy = _qualifier_hierarchy


mesh_qualifer_dict_list = []
for qualifer, sub_hierarchical_qualifer in qualifier_hierarchy.items():
    print(f"MeSH Qualifier: {qualifer}")
    print(f"Scope Note: {qualifier_scope_note[qualifer]}")
    print(f"Sub Hierarchical Qualifiers: {sub_hierarchical_qualifer}")
    print("\n")

    mesh_qualifer_dict_list.append(
        {
            "MeSH Qualifier": qualifer,
            "Scope Note": qualifier_scope_note[qualifer],
            "Sub Qualifiers (in Hierarchy)": sub_hierarchical_qualifer,
        }
    )

mesh_qualifier_prompt = []
for idx, mesh_qualifer_dict in enumerate(mesh_qualifer_dict_list):
    code = f"{idx:02d}"
    title = mesh_qualifer_dict["MeSH Qualifier"]
    description = mesh_qualifer_dict["Scope Note"]
    sub_title_str = mesh_qualifer_dict["Sub Qualifiers (in Hierarchy)"]
    prompt = f"### {title}\n* Code: {code}\n* Description: {description}\n* Sub Qualifiers Hiearchy:\n{sub_title_str}"

    mesh_qualifier_prompt.append(
        {
            "code": code,
            "title": title,
            "description": description,
            "sub_qualifiers": sub_title_str,
            "prompt": prompt,
        }
    )

with open("src/select_data/mesh_qualifier_hierarchical.json", "w") as f:
    json.dump(mesh_qualifier_prompt, f, indent=2)
