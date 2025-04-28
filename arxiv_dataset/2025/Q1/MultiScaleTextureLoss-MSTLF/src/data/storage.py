import json

# Patients Test 1
test_1_ids = ["L067"]

# Patients Test 2 (+ test 1)
test_2_ids = ["C002", "C004", "C030", "C067", "C004", "C120", "C124", "C166", "L067"]

# Patients Test 3
test_3_ids = ["p_342", "p_350", "p_527", "p_545", "p_785", "p_865", "p_869", "p_895"]

# Patients ELCAP
test_4_ids = ["W0001", "W0002", "W0003", "W0004", "W0005", "W0006", "W0007", "W0008", "W0009", "W0010",
              "W0011", "W0012", "W0013", "W0014", "W0015", "W0016", "W0017", "W0018", "W0019", "W0020",
              "W0021", "W0022", "W0023", "W0024", "W0025", "W0026", "W0027", "W0028", "W0029", "W0030",
              "W0031", "W0032", "W0033", "W0034", "W0035", "W0036", "W0037", "W0038", "W0039", "W0040",
              "W0041", "W0042", "W0043", "W0044", "W0045", "W0046", "W0047", "W0049", "W0050", "W0048"
              ] # , "W0048"

def init_storing(patient_list, metrics):
    return {patient: {metric: [] for metric in metrics} for patient in patient_list}


def init_image_buffer(patient_list):
    return {patient: [] for patient in patient_list}


def empty_Dictionary(dictionary, nesting=2):
    if nesting == 2:
        [dictionary[p][key].clear() for p in dictionary.keys() for key in dictionary[p].keys()]
    elif nesting == 1:
        [dictionary[p].clear() for p in dictionary.keys()]


def save_to_json(data, path):
    # Save to JSON
    with open(f"{path}.json", "w") as file:
        json.dump(data, file)
    with open('formatted.json', 'w') as f:
        json.dump(data, f, indent=4)


def load_from_json(path):
    # Load from JSON
    with open(f"{path}.json", "r") as json_file:
        loaded_metrics_data = json.load(json_file)
    return loaded_metrics_data


if __name__ == "__main__":

    import numpy as np
    import torch

    t = torch.ones((1, 3, 256, 256))

    a = init_image_buffer(test_2_ids)
    a['C002'].append(t)
    a['C002'].append(t)
    print(a['C002'])
    r = torch.cat(a['C002'], dim=0)
    print(r.shape)
    empty_dictionary(a, nesting=1)
    print(a)