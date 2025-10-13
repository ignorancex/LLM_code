import os 
import csv
import json
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

vis_folder = "visualize"
os.makedirs(vis_folder, exist_ok=True)

workspace = "../" # your created workspace
with open(os.path.join(workspace, "data.json"), "r") as f:
    all_MC = json.load(f)
    print(len(all_MC))
workspace = os.path.join(workspace, "results")

#########################################################################################################################################
# helpers for summerizing
#########################################################################################################################################

VQAtype_dict = {
    "M": "Modality Recognition",
    "A": "Anatomy Identification",
    "S": "Symptom Recognition",
    "D": "Disease Diagnosis",
    "SO": "Surgery & Operation",
}

def prompt_to_correctChoice(prompt, Answer):
    prompt = prompt.strip()
    choice_content = prompt.split("Choices : ")[1]
    choices = choice_content.split(" | ")
    for choice in choices:
        if choice.endswith(Answer):
            # print (choice.split(". ")[0])
            return choice.split(". ")[0]
    raise ValueError("Answer not found")

def prompt_to_distractor(prompt, choice):
    choice_content = prompt.split("Choices : ")[1]
    choices = choice_content.split(" | ")
    for c in choices:
        if c.startswith(choice):
            return c.split(". ")[1]
    raise ValueError("Distractor not found")



#########################################################################################################################################
# Order of models in the result summary
#########################################################################################################################################

order_modelNames = [
    ###########################################################
    # closed general
    "gemini-2.5-pro-preview-03-25",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-2024-11-20",
    "claude-3-7-sonnet-20250219",

    ##########################################################
    # open general
    "InternVL3-78B",
    "InternVL3-38B",
    "InternVL3-14B",
    "InternVL3-9B",
    "InternVL3-8B",
    "InternVL3-2B",
    "InternVL3-1B",

    "llava-onevision-qwen2-72b-ov-chat",
    "llava-onevision-qwen2-7b-ov-chat",
    "llava-onevision-qwen2-0.5b-ov",

    "Ovis2-34B",
    "Ovis2-16B",
    "Ovis2-8B",
    "Ovis2-4B",
    "Ovis2-2B",
    "Ovis2-1B",

    "Qwen2.5-VL-72B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-3B-Instruct",

    "InternVL2.5-78B",
    "InternVL2.5-38B",
    "InternVL2.5-26B",
    "InternVL2.5-8B",
    "InternVL2.5-4B",
    "InternVL2.5-2B",
    "InternVL2.5-1B",

    "deepseek-vl2-tiny",
    "deepseek-vl2-small",

    ############################################
    # open meidcal
    "HealthGPT-L14",
    "HealthGPT-M3",

    "HuatuoGPT-Vision-34B",
    "HuatuoGPT-Vision-7B",

    "MedDr",

    ############################################
    # closed reasoning general
    "o4-mini-2025-04-16",
    "claude-3-7-sonnet-thinking",
    ############################################
    # open reasoning general
    "VL-Reasoner-72B",
    "VL-Reasoner-7B",

    "Skywork-R1V2-38B",
    "Skywork-R1V-38B",

    "Kimi-VL-A3B-Thinking",
    ############################################
    # open reasoning medical
    "MedVLM-R1"
]

reformat_modelName = {
    ###########################################################
    # closed general
    "gemini-2.5-pro-preview-03-25": "Gemini2.5-Pro(03-25)",
    "gpt-4.1-mini-2025-04-14": "GPT4.1-mini", #  (2025-04-14)
    "gpt-4.1-2025-04-14": "GPT4.1", #  (2025-04-14)
    "gpt-4o-2024-11-20": "GPT-4o", #  (2024-11-20)
    "claude-3-7-sonnet-20250219": "Claude-3.7-Sonnet", #  (2025-02-19)

    ##########################################################
    # open general

    "llava-onevision-qwen2-72b-ov-chat": "LLaVA-OV-72B",
    "llava-onevision-qwen2-7b-ov-chat": "LLaVA-OV-7B",
    "llava-onevision-qwen2-0.5b-ov": "LLaVA-OV-0.5B",

    "deepseek-vl2-tiny": "DeepSeek-VL2-Tiny",
    "deepseek-vl2-small": "DeepSeek-VL2-Small",

    ############################################
    # closed reasoning general
    "claude-3-7-sonnet-thinking": "Claude3.7-Sonnet-Thinking",

}


#########################################################################################################################################
# helpers for visualization
#########################################################################################################################################


def build_empty_dict(length=0):
    global Organ_to_key, bodyPart_to_key, modality_to_key, department_to_key
    label_res_dict = {
        "Modality": {"Other": {"SUM": length, "Other": {"SUM": length}}},
                    # system            # subsystem          # bodypart        # organ
        "Anatomy": {"Other": {"SUM": length, "Other": {"SUM": length, "Other": {"SUM": length, "Other": {"SUM": length}}}}},
        "Department": {"Other": {"SUM": length, "Other": {"SUM": length}}},
    }

    ANATOMY = './labels/Anatomy.csv'
    ANATOMY = pd.read_csv(ANATOMY)
    for i in range(len(ANATOMY)):
        System = ANATOMY['System'][i].strip()
        if System not in label_res_dict["Anatomy"]:
            label_res_dict["Anatomy"][System] = {"SUM": length, "Other": {"SUM": length}}
        SubSystem = ANATOMY['SubSystem'][i].strip()
        if SubSystem not in label_res_dict["Anatomy"][System]:
            label_res_dict["Anatomy"][System][SubSystem] = {"SUM": length, "Other": {"SUM": length}}
        BodyPart = ANATOMY['BodyPart'][i].strip()
        if BodyPart not in label_res_dict["Anatomy"][System][SubSystem]:
            label_res_dict["Anatomy"][System][SubSystem][BodyPart] = {"SUM": length, "Other": {"SUM": length}}
        Organ = ANATOMY['Organ'][i].strip()
        if Organ not in label_res_dict["Anatomy"][System][SubSystem][BodyPart]:
            label_res_dict["Anatomy"][System][SubSystem][BodyPart][Organ] = {"SUM": length}

    Organ_to_key = {}
    for i in range(len(ANATOMY)):
        anatomy_Organ = ANATOMY["Organ"][i].strip()
        Organ_to_key[anatomy_Organ] = (ANATOMY['System'][i].strip(), ANATOMY['SubSystem'][i].strip(), ANATOMY['BodyPart'][i].strip())

    bodyPart_to_key = {}
    for i in range(len(ANATOMY)):
        anatomy_BodyPart = ANATOMY["BodyPart"][i].strip()
        bodyPart_to_key[anatomy_BodyPart] = (ANATOMY['System'][i].strip(), ANATOMY['SubSystem'][i].strip())


    MODALITY = './labels/Modality.csv'
    MODALITY = pd.read_csv(MODALITY)
    for i in range(len(MODALITY)):
        Category = MODALITY['Category'][i].strip()
        if Category not in label_res_dict["Modality"]:
            label_res_dict["Modality"][Category] = {"SUM": length}
        Modality = MODALITY['FullName'][i].strip()
        if Modality not in label_res_dict["Modality"][Category]:
            label_res_dict["Modality"][Category][Modality] = {"SUM": length}
    modality_to_key = {}
    for i in range(len(MODALITY)):
        modality_to_key[MODALITY['FullName'][i].strip()] = MODALITY['Category'][i].strip()

    DEPARTMENT = './labels/Department.csv'
    DEPARTMENT = pd.read_csv(DEPARTMENT)
    for i in range(len(DEPARTMENT)):
        MedicalGeneral = DEPARTMENT['MedicalGeneral'][i].strip()
        if MedicalGeneral not in label_res_dict["Department"]:
            label_res_dict["Department"][MedicalGeneral] = {"SUM": length}
        MedicalDepartment = DEPARTMENT['MedicalDepartment'][i].strip()
        if MedicalDepartment not in label_res_dict["Department"][MedicalGeneral]:
            label_res_dict["Department"][MedicalGeneral][MedicalDepartment] = {"SUM": length}
    department_to_key = {}
    for i in range(len(DEPARTMENT)):
        department_to_key[DEPARTMENT['MedicalDepartment'][i].strip()] = DEPARTMENT['MedicalGeneral'][i].strip()

    return label_res_dict


############################################################################################################################
def sort_dict(data_dict):
    data_dict = data_dict.copy()
    if "SUM" in data_dict:
        SUM = data_dict.pop("SUM")
        data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1]['SUM'], reverse=True))
        data_dict["SUM"] = SUM
    else:
        data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1]['SUM'], reverse=True))
    return data_dict

def sort_whole_dict(label_res_dict):
    for mode in label_res_dict:
        label_res_dict[mode] = sort_dict(label_res_dict[mode])
        for L1 in label_res_dict[mode]:
            if L1 != "SUM" and type(label_res_dict[mode][L1]) == dict:
                label_res_dict[mode][L1] = sort_dict(label_res_dict[mode][L1])
                for L2 in label_res_dict[mode][L1]:
                    if L2 != "SUM" and type(label_res_dict[mode][L1][L2]) == dict:
                        label_res_dict[mode][L1][L2] = sort_dict(label_res_dict[mode][L1][L2])
                        for L3 in label_res_dict[mode][L1][L2]:
                            if L3 != "SUM" and type(label_res_dict[mode][L1][L2][L3]) == dict:
                                label_res_dict[mode][L1][L2][L3] = sort_dict(label_res_dict[mode][L1][L2][L3])

    return label_res_dict


########################################################################################################################
def build_num_dict(label_res_dict, reformated_data):
    for item in reformated_data:
        accuracy = item["item_accuracy"]
        MC_item = [i for i in all_MC if i["_id"] == item["_id"]][0]
        label = MC_item["LABEL"]
        for mode in ["Modality", "Anatomy", "Department"]:
            L1 = label[mode][0]
            L2 = label[mode][1]
            label_res_dict[mode][L1]["SUM"] += 1
            label_res_dict[mode][L1][L2]["SUM"] += 1
            if mode == "Anatomy":
                L3 = label[mode][2]
                L4 = label[mode][3]
                label_res_dict[mode][L1][L2][L3]["SUM"] += 1
                label_res_dict[mode][L1][L2][L3][L4]["SUM"] += 1

    return label_res_dict

def remove_empty(label_res_dict, reformated_data, refer_dict=None):
    returned_dict = copy.deepcopy(label_res_dict)
    if refer_dict is None:
        refer_dict = build_empty_dict()
        refer_dict = build_num_dict(refer_dict, reformated_data)
    for mode in label_res_dict:
        for L1 in label_res_dict[mode]:
            if L1 != "SUM": 
                if L1 not in refer_dict[mode] or refer_dict[mode][L1]["SUM"] == 0:
                    returned_dict[mode].pop(L1)
                    continue
                for L2 in label_res_dict[mode][L1]:
                    if L2 != "SUM":
                        if L2 not in refer_dict[mode][L1] or refer_dict[mode][L1][L2]["SUM"] == 0:
                            returned_dict[mode][L1].pop(L2)
                            continue
                        if mode == "Anatomy":
                            for L3 in label_res_dict[mode][L1][L2]:
                                if L3 != "SUM":
                                    if L3 not in refer_dict[mode][L1][L2] or refer_dict[mode][L1][L2][L3]["SUM"] == 0:
                                        returned_dict[mode][L1][L2].pop(L3)
                                        continue
                                    for L4 in label_res_dict[mode][L1][L2][L3]:
                                        if L4 != "SUM":
                                            if L4 not in refer_dict[mode][L1][L2][L3] or refer_dict[mode][L1][L2][L3][L4]["SUM"] == 0:
                                                returned_dict[mode][L1][L2][L3].pop(L4)

    return returned_dict

def average_dict(numerator_dict, denominator_dict):
        
    dict_average = numerator_dict.copy()
    for mode in dict_average:
        for L1 in dict_average[mode]:
            for L2 in dict_average[mode][L1]:
                if L2 == "SUM":
                    if denominator_dict[mode][L1][L2] != 0:
                        dict_average[mode][L1][L2] = numerator_dict[mode][L1][L2]/denominator_dict[mode][L1][L2]
                else:
                    for L3 in dict_average[mode][L1][L2]:
                        if L3 == "SUM":
                            if denominator_dict[mode][L1][L2][L3] != 0:
                                dict_average[mode][L1][L2][L3] = numerator_dict[mode][L1][L2][L3]/denominator_dict[mode][L1][L2][L3]
                        else:
                            for L4 in dict_average[mode][L1][L2][L3]:
                                if L4 == "SUM":
                                    if denominator_dict[mode][L1][L2][L3][L4] != 0:
                                        dict_average[mode][L1][L2][L3][L4] = numerator_dict[mode][L1][L2][L3][L4]/denominator_dict[mode][L1][L2][L3][L4]
                                else:
                                    if denominator_dict[mode][L1][L2][L3][L4]["SUM"] != 0:
                                        dict_average[mode][L1][L2][L3][L4]["SUM"] = numerator_dict[mode][L1][L2][L3][L4]["SUM"]/denominator_dict[mode][L1][L2][L3][L4]["SUM"]

    dict_average = sort_whole_dict(dict_average)
    return dict_average

def faltten_two_to_one(data_dict):
    data_dict = data_dict.copy()
    faltten_dict = {}
    for key1, value1 in data_dict.items():
        if isinstance(value1, dict):
            for key2, value2 in value1.items():
                if isinstance(value2, dict):
                    if key2 == "Other" and value2["SUM"] != 0:
                        faltten_dict[f"{key1}_{key2}"] = value2
                    else:
                        faltten_dict[key2] = value2
    return faltten_dict


def draw_onelevel_barchart(data_dict, level1, threshold=0, colors='#4682B4', 
                           title_fontsize=16, title_fontweight='bold', figsize=(10, 6),
                           title_fontfamily='sans-serif', title_pad=10, file_name=None):
    import matplotlib.pyplot as plt
    import numpy as np

    data_dict = data_dict.copy()

    # Remove "SUM" key if present
    if "SUM" in data_dict:
        data_dict.pop("SUM")

    labels = []
    values = []
    for key1, value1 in data_dict.items():
        if isinstance(value1, dict) and "SUM" in value1:
            labels.append(key1)
            values.append(value1["SUM"])

    # Normalize values
    # total = sum(values)
    # values = [value / total for value in values]

    # Filter labels and values based on the threshold
    filtered_labels = []
    filtered_values = []
    for label, value in zip(labels, values):
        if value >= threshold:
            if label == "Spine (vertebral column)":
                label = "Spine"
            if label == "General Photo of Affected Area":
                label = "General Photo"
            filtered_labels.append(label)
            filtered_values.append(value)

    # Bar chart
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(filtered_labels))
    plt.barh(y_pos, filtered_values, color=colors)

    # Add labels and title
    plt.yticks(y_pos, filtered_labels, fontsize=6)
    plt.xlabel('avg_accuracy(a M on a VQA)', fontsize=6)
    plt.title(level1, fontsize=title_fontsize, fontweight=title_fontweight, fontfamily=title_fontfamily, pad=title_pad)

    # Save as SVG before showing the plot
    if file_name is None:
        file_name = f"{vis_folder}/{level1}.svg"
    plt.savefig(file_name, format='svg', bbox_inches='tight', dpi=300)

    # Display the plot
    plt.show()

##############################################################################
def draw_barchart_level(dict_statistics, level = "L1", colors='#4682B4'):
    directory = {
        "L1": {
            "Modality": "ModalityCategory",
            "Department": "DepartmentCategory",
            "Anatomy": "System",
            "figsize": (8, 6)
        },
        "L2": {
            "Modality": "Modality",
            "Department": "Department",
            "Anatomy": "SubSystem",
            "figsize": (8, 6)
        },
        "L3": {
            "Modality": "Modality",
            "Department": "Department",
            "Anatomy": "BodyPart",
            "figsize": (8, 6)
        },
        "L4": {
            "Modality": "Modality",
            "Department": "Department",
            "Anatomy": "Organ",
            "figsize": (8, 6)
        }
    }

    falten = dict_statistics.copy() # ModalityCategory, DepartmentCategory, System
    if level == "L2" or level == "L3" or level == "L4":
        for mode, d in falten.items():
            falten[mode] = faltten_two_to_one(d) # Modality, Department, SubSystem
    if level == "L3" or level == "L4":
        falten["Anatomy"] = faltten_two_to_one(falten["Anatomy"]) # Modality, Department, BodyPart
    if level == "L4":
        falten["Anatomy"] = faltten_two_to_one(falten["Anatomy"]) # Modality, Department, Organ

    for mode in ["Modality", "Department", "Anatomy"]:
        if mode == "Anatomy":
            draw_onelevel_barchart(falten[mode], directory[level][mode],file_name=f"visualize/{directory[level][mode]}.svg", figsize=directory[level]["figsize"], colors=colors)
        else:
            draw_onelevel_barchart(falten[mode], directory[level][mode],file_name=f"visualize/{directory[level][mode]}.svg", colors=colors)

def filter_threadhold(label_res_dict, reformated_data, threadhold=200):
    returned_dict = copy.deepcopy(label_res_dict)
    num_dict = build_empty_dict()
    num_dict = build_num_dict(num_dict, reformated_data)
    for mode in label_res_dict:
        for L1 in label_res_dict[mode]:
            if L1 != "SUM": 
                if num_dict[mode][L1]["SUM"] < threadhold:
                    returned_dict[mode].pop(L1)
                    continue
                for L2 in label_res_dict[mode][L1]:
                    if L2 != "SUM":
                        if num_dict[mode][L1][L2]["SUM"] < threadhold:
                            returned_dict[mode][L1].pop(L2)
                            continue
                        if mode == "Anatomy":
                            for L3 in label_res_dict[mode][L1][L2]:
                                if L3 != "SUM":
                                    if num_dict[mode][L1][L2][L3]["SUM"] < threadhold:
                                        returned_dict[mode][L1][L2].pop(L3)
                                        continue
                                    for L4 in label_res_dict[mode][L1][L2][L3]:
                                        if L4 != "SUM":
                                            if num_dict[mode][L1][L2][L3][L4]["SUM"] < threadhold:
                                                returned_dict[mode][L1][L2][L3].pop(L4)
    return returned_dict

########################################################################################################################
def build_model_accuracy_dict(label_res_dict, reformated_data, model):
    for item in reformated_data:
        accuracy = item["result"][model]
        MC_item = [i for i in all_MC if i["_id"] == item["_id"]][0]
        label = MC_item["LABEL"]
        for mode in ["Modality", "Anatomy", "Department"]:

            L1 = label[mode][0]
            L2 = label[mode][1]
            label_res_dict[mode][L1]["SUM"] += accuracy
            label_res_dict[mode][L1][L2]["SUM"] += accuracy
            if mode == "Anatomy":
                L3 = label[mode][2]
                L4 = label[mode][3]
                label_res_dict[mode][L1][L2][L3]["SUM"] += accuracy
                label_res_dict[mode][L1][L2][L3][L4]["SUM"] += accuracy

    return label_res_dict

def save_as_csv(dSingleModel_dictict_, level="L1"):
    directory = {
        "L1": {
            "Modality": "ModalityCategory",
            "Department": "DepartmentCategory",
            "Anatomy": "System"
        },
        "L2": {
            "Modality": "Modality",
            "Department": "Department",
            "Anatomy": "SubSystem"
        },
        "L3": {
            "Modality": "Modality",
            "Department": "Department",
            "Anatomy": "BodyPart"
        },
        "L4": {
            "Modality": "Modality",
            "Department": "Department",
            "Anatomy": "Organ"
        }
    }

    # Prepare the output directory structure
    for mode in ["Modality", "Department", "Anatomy"]:
        file_name = f"{vis_folder}/{directory[level][mode]}.csv"
        rows = []
        headers = ["Model"]

        # Loop through each model
        for model, model_data in dSingleModel_dictict_.items():
            if mode in model_data:
                # Flatten the data for the current model and mode
                flattened_data = faltten_two_to_one(model_data[mode]) if level in ["L2", "L3", "L4"] else model_data[mode]
                if level in ["L3", "L4"] and mode == "Anatomy":
                    flattened_data = faltten_two_to_one(flattened_data)

                # Add headers dynamically
                if len(headers) == 1:
                    headers.extend(flattened_data.keys())

                # Prepare the row for the current model
                row = [model]
                for label in headers[1:]:
                    n = flattened_data.get(label, {}).get('SUM', 0)*100
                    n = round(n, 2)
                    row.append(n if n != 0 else 0.0)
                rows.append(row)

        # Write the CSV file
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

def visualize_csv_heatmap(csv_file, title, cmap='Blues', figsize=(10, 10), min_max = [None, None]):

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Set the index to the first column (Model names)
    df.set_index(df.columns[0], inplace=True)

    # Create a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, cbar=True, linewidths=.5, annot_kws={"size": 11}, vmin=min_max[0], vmax=min_max[1])
    plt.title(title)
    # Save the heatmap as an image
    plt.savefig(f"{vis_folder}/{title}.svg", format='svg', bbox_inches='tight', dpi=300)
    plt.show()

# find the smallest and largest numeric values in all csv files
def find_min_max():
    min_value = float('inf')
    max_value = float('-inf')
    for csv_file in os.listdir(vis_folder):
        if not csv_file.endswith(".csv") or "Result_summary" in csv_file:
            continue
        df = pd.read_csv(os.path.join(vis_folder, csv_file))
        # Ensure only numeric columns are considered
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            min_value_in_file = numeric_df.min().min()
            if min_value_in_file < min_value:
                min_value = min_value_in_file
            max_value_in_file = numeric_df.max().max()
            if max_value_in_file > max_value:
                max_value = max_value_in_file
    return min_value, max_value