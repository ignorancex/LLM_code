import os
import random

import numpy as np
import pandas as pd


random.seed(42)
np.random.seed(42)


bcn20000_label_map = {
    "NV": "Nevus",
    "MEL": "Melanoma",
    "BCC": "Basal cell carcinoma",
    "BKL": "Benign keratosis",
    "AK": "Actinic keratosis",
    "SCC": "Squamous carcinoma",
    "DF": "Dermatofibroma",
    "VASC": "Vascular",
}

padufes20_label_map = {
    "ACK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "NEV": "Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "SEK": "Seborrheic Keratosis",
}


def sample_prompt(prompts: list) -> tuple:
    # Sample a question type
    question_type = random.choice(prompts)

    # Sample a variation of the question type
    variation = random.choice(question_type["variations"])

    return question_type["question_type"], variation


def prepare_metadata_derm12345(row: pd.Series) -> str:
    """Prepare metadata string for the DERM12345 dataset"""
    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {row['super_class']}, {row['malignancy']}, {row['main_class_1']},"
        f" {row['main_class_2']}, {row['sub_class']}"
    )
    return metadata_context


def prepare_metadata_hiba(row: pd.Series) -> str:
    """Use diagnosis_1	diagnosis_2	diagnosis_3 features to generate metadata"""
    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {row['diagnosis_1']}, {row['diagnosis_2']}, {row['diagnosis_3']}"
    )
    return metadata_context


def prepare_metadata_padufes20(row: pd.Series) -> str:
    """Prepare metadata string for the DERM12345 dataset"""
    label = padufes20_label_map[row["diagnostic"]]
    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {label}\n"
        f"- Age: {row['age']}\n"
    )
    return metadata_context


def prepare_metadata_bcn20000(row: pd.Series) -> str:
    """
    Prepare metadata string for the BCN20000 dataset

    Sample metadata string for the BCN20000 dataset:
        bcn_filename           BCN_0000000001.jpg
        age_approx                           55.0
        anatom_site_general        anterior torso
        diagnosis                             MEL
        lesion_id                     BCN_0003884
        capture_date                   2012-05-16
        sex                                  male
        split                               train
    """
    label = bcn20000_label_map[row["diagnosis"]]

    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"- Label: {label}\n"
        f"- Age: {row['age_approx']}\n"
        f"- Anatomical Site: {row['anatom_site_general']}\n",
        f"- Sex: {row['sex']}\n",
    )
    return metadata_context


def prepare_metadata_scin(row: pd.Series) -> str:
    """
    Prepare metadata string for the SCIN dataset.
    Explains the weighted label structure to the model.
    """
    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"The following shows the possible diagnoses with their confidence weights in parentheses.\n"
        f"The weights sum to 1.0, where higher weights indicate higher confidence in that diagnosis.\n"
        f"- Labels: {row['labels']}"
    )
    return metadata_context


def prepare_metadata_scin_clinical(row: pd.Series) -> str:
    """
    Prepare detailed clinical metadata string for the SCIN dataset.
    Includes symptoms, affected body parts, physical characteristics, and diagnostic information.

    Args:
        row (pd.Series): A row from the processed SCIN dataframe containing:
            - labels: weighted diagnosis labels
            - symptoms: list of reported symptoms
            - body_parts: list of affected body parts
            - textures: list of physical characteristics
            - demographics: dictionary of demographic information

    Returns:
        str: Formatted metadata context string
    """
    # Format symptoms list
    symptoms_str = ", ".join(row["symptoms"])

    # Format body parts list
    body_parts_str = ", ".join(row["body_parts"])

    # Format textures list
    textures_str = ", ".join(row["textures"])

    # Format demographics
    demographics = row["demographics"]
    demo_str = (
        f"Age Group: {demographics['age_group']}, "
        f"Sex: {demographics['sex_at_birth']}, "
        f"Skin Type: {demographics['fitzpatrick_skin_type']}, "
        f"Duration: {demographics['condition_duration']}"
    )

    metadata_context = (
        f"Metadata of the lesion/disease image:\n"
        f"Clinical Presentation:\n"
        f"- Reported Symptoms: {symptoms_str}\n"
        f"- Affected Areas: {body_parts_str}\n"
        f"- Physical Characteristics: {textures_str}\n"
        f"- Patient Demographics: {demo_str}\n\n"
        f"Diagnostic Assessment:\n"
        f"The following shows the possible diagnoses with their confidence weights in parentheses.\n"
        f"The weights sum to 1.0, where higher weights indicate higher confidence in that diagnosis.\n"
        f"- Diagnostic Considerations: {row['labels']}"
    )

    return metadata_context


def prepare_api_prompt(
    metadata_context: str, prompt_text: str, question_type: str
) -> str:
    """
    Function to prepare API prompt with metadata and question text.
    """
    prompt = (
        f"***** INSTRUCTIONS *****\n"
        f"You are a dermatologist examining a lesion/skin disease image.\n"
        f"You have no direct access to the patient's metadata. "
        f"Although some context is provided below, do NOT reference or rely on it in your answer. "
        f"Base your answer solely on the visible characteristics of the lesion/skin disease in the image.\n"
        f"Provide your answer in the exact format specified below.\n\n"
        f"***** CONTEXT (FOR INTERNAL REFERENCE ONLY) *****\n"
        f"{metadata_context}\n\n"
        f"***** QUESTION *****\n"
        f"{prompt_text}\n\n"
        f"***** RESPONSE FORMAT *****\n"
    )
    if question_type == "Creativity":
        prompt += f'{{"question": "<question>", "answer": "<answer>"}}'
    else:
        prompt += f'{{"answer": "<answer>"}}'

    return prompt


def generate_prompts_dataset(
    metadata: pd.DataFrame, prompt_samples: dict, dataset: str, dataset_path: str
) -> list:
    """
    Generate prompts for each image in the dataset.

    Args:
        metadata: DataFrame containing metadata of the images
        prompt_samples: Dictionary containing question types and variations

    Returns:
        List of API-ready requests for each image in the dataset
    """
    api_requests = []

    if dataset == "derm12345":
        prepare_metadata = prepare_metadata_derm12345
    elif dataset == "bcn20000":
        prepare_metadata = prepare_metadata_bcn20000
    elif dataset == "pad-ufes-20":
        prepare_metadata = prepare_metadata_padufes20
    elif dataset == "scin":
        prepare_metadata = prepare_metadata_scin
    elif dataset == "scin_clinical":
        prepare_metadata = prepare_metadata_scin_clinical
    elif dataset == "hiba":
        prepare_metadata = prepare_metadata_hiba
    else:
        raise ValueError("Invalid dataset name. Please provide a valid dataset name.")

    for idx, row in metadata.iterrows():
        # Metadata for the current image
        metadata_context = prepare_metadata(row)

        # Sample a general question
        general_question_type, general_prompt = sample_prompt(prompt_samples["general"])

        # Sample a dataset-specific question (HAM10000 in this case)
        dataset_question_type, dataset_prompt = sample_prompt(prompt_samples[dataset])

        # No need to replace placeholders in prompts for DERM12345 dataset
        if dataset == "derm12345":
            image_id = row["image_id"]
            general_prompt_filled = general_prompt
            dataset_prompt_filled = dataset_prompt
            image_path = os.path.join(
                dataset_path, "train", row["label"], (row["image_id"] + ".jpg")
            )

        elif dataset == "bcn20000":
            """
            df columns: ['bcn_filename', 'age_approx', 'anatom_site_general', 'diagnosis',
            'lesion_id', 'capture_date', 'sex', 'split'],
            """
            image_id = row["bcn_filename"]

            localization = row["anatom_site_general"]
            age = int(row["age_approx"]) if not np.isnan(row["age_approx"]) else None
            sex = row["sex"]

            if not localization or not age or not sex:
                """
                Resample the dataset question without the "Anatomical Context"
                and "Comprehensive Assessment" question types
                """
                dataset_question_type, dataset_prompt = sample_prompt(
                    [prompt_samples[dataset][i] for i in range(5)]
                )

            # Replace placeholders in prompts with actual metadata
            general_prompt_filled = general_prompt
            dataset_prompt_filled = dataset_prompt.format(
                anatom_site_general=localization, age_approx=age, sex=sex
            )
            image_path = os.path.join(dataset_path, row["bcn_filename"])

        elif dataset == "pad-ufes-20":
            image_id = row["img_id"].replace(".png", "")
            label = row["diagnostic"]
            age = row["age"]
            general_prompt_filled = general_prompt
            dataset_prompt_filled = dataset_prompt
            image_path = os.path.join(dataset_path, image_id + ".png")

        elif dataset == "scin":
            image_id = row["image_id"]
            general_prompt_filled = general_prompt
            dataset_prompt_filled = dataset_prompt
            image_path = row["image_path"]

        elif dataset == "scin_clinical":
            # Overwrite general question type from dataset-specific question types for SCIN Clinical
            general_question_type, general_prompt = sample_prompt(
                prompt_samples[dataset]
            )
            image_id = row["image_id"]

            # Format symptoms list
            symptoms_str = ", ".join(row["symptoms"])
            symptoms_str = f"({symptoms_str})"
            # Format body parts list
            body_parts_str = ", ".join(row["body_parts"])
            body_parts_str = f"({body_parts_str})"
            # Format textures list
            textures_str = ", ".join(row["textures"])
            textures_str = f"({textures_str})"

            general_prompt_filled = general_prompt.format(
                textures=textures_str, symptoms=symptoms_str, body_parts=body_parts_str
            )
            dataset_prompt_filled = dataset_prompt.format(
                textures=textures_str, symptoms=symptoms_str, body_parts=body_parts_str
            )

            image_path = row["image_path"]

        elif dataset == "hiba":
            image_id = row["isic_id"]
            general_prompt_filled = general_prompt
            dataset_prompt_filled = dataset_prompt
            image_path = os.path.join(dataset_path, row["isic_id"] + ".jpg")

        # Prepare API-ready requests
        general_api_prompt = prepare_api_prompt(
            metadata_context,
            general_prompt_filled,
            general_question_type,
        )
        dataset_api_prompt = prepare_api_prompt(
            metadata_context,
            dataset_prompt_filled,
            dataset_question_type,
        )

        api_requests.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "q1_general": {
                    "question_type": general_question_type,
                    "prompt": general_api_prompt,
                    "question": general_prompt_filled,
                },
                "q2_dataset_specific": {
                    "question_type": dataset_question_type,
                    "prompt": dataset_api_prompt,
                    "question": dataset_prompt_filled,
                },
            }
        )

    return api_requests
