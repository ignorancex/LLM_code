"""
data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json
has mismatched keys: {'modality'}
Change it to {'modality_type'}

All the others have:
['dataset', 'gt_answer', 'image_path', 'modality_type', 'option_A', 'option_B', 'option_C', 'option_D', 'question', 'question_id', 'question_type']
['dataset', 'gt_answer', 'image_path', 'modality_type', 'option_A', 'option_B', 'option_C', 'option_D', 'question', 'question_id', 'question_type']
But
['dataset', 'gt_answer', 'image_path', 'modality', 'option_A', 'option_B', 'option_C', 'option_D', 'question', 'question_id', 'question_type']
"""

import dotenv

dotenv.load_dotenv(override=True)
import json
from pathlib import Path
from shutil import copyfile


def fix_key_in_json_file(data, old_key: str, new_key: str):
    """
    Fix the key in the JSON data.
    """
    for sample in data:
        if old_key in sample:
            sample[new_key] = sample.pop(old_key)
    return data


def main():
    # Path to the JSON file
    json_file_path = Path(
        "data/OmniMedVQA/QA_information/Open-access/Chest CT Scan.json"
    )
    backup_old_file_dir = Path("data/OmniMedVQA/QA_information/backup")
    backup_old_file_dir.mkdir(parents=True, exist_ok=True)

    # Backup the old file
    backup_file_path = backup_old_file_dir / json_file_path.name
    if not backup_file_path.exists():
        # Copy the original file to the backup directory
        # This will only copy if the file does not already exist in the backup directory
        copyfile(json_file_path, backup_file_path)
        print(f"Backup created at: {backup_file_path}")
    else:
        print(f"Backup already exists at: {backup_file_path}")
        print("Exiting without making changes.")
        return

    # Load the JSON data
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # Fix the key in the JSON data
    fixed_data = fix_key_in_json_file(data, "modality", "modality_type")

    # Save the fixed JSON data back to the file
    with open(json_file_path, "w") as file:
        json.dump(fixed_data, file, indent=4)
    print(f"Fixed key 'modality' to 'modality_type' in {json_file_path}")


if __name__ == "__main__":
    main()
