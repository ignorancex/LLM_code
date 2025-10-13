# %%
import logging
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
ORGANIZATION = "UCSC-VLAA"
BASE_PATHS = {
    "sft-m23k": "~/efs/xhuan192/codes/med-vlrm/outputs/sft-m23k-converted/",
    "sft-pmc_vqa": "~/efs/xhuan192/codes/med-vlrm/outputs/sft-pmc_vqa-converted/",
    "converted": "~/efs/xhuan192/codes/med-vlrm/outputs/converted/",
}

# Model mapping from local names to HuggingFace repo names
MODEL_MAPPING = {
    # SFT models
    "3b-m23k-checkpoint-4401": "MedVLThinker-3B-SFT_m23k",
    "7b-m23k-checkpoint-4401": "MedVLThinker-7B-SFT_m23k",
    "3b-pmc_vqa-checkpoint-12594": "MedVLThinker-3B-SFT_PMC",
    "7b-pmc_vqa-checkpoint-12594": "MedVLThinker-7B-SFT_PMC",
    # RL models
    "train-qwen2_5_vl_3b-pmc_vqa-m23k_sft_epoch_3-step_1150": "MedVLThinker-3B-SFT_m23k-RL_PMC",
    "train-qwen2_5_vl_3b-m23k-step_320": "MedVLThinker-3B-RL_m23k",
    "train-qwen2_5_vl_3b-pmc_vqa-step_451": "MedVLThinker-3B-RL_PMC",
    "train-qwen2_5_vl_3b-pmc_vqa-m23k_rl-step_1805": "MedVLThinker-3B-RL_m23k-RL_PMC",
    "train-qwen2_5_vl_7b-pmc_vqa-m23k_sft_epoch_3-step_1805": "MedVLThinker-7B-SFT_m23k-RL_PMC",
    "train-qwen2_5_vl_7b-m23k-step_320": "MedVLThinker-7B-RL_m23k",
    "train-qwen2_5_vl_7b-pmc_vqa-step_451": "MedVLThinker-7B-RL_PMC",
    "train-qwen2_5_vl_7b-pmc_vqa-m23k_rl-step_1805": "MedVLThinker-7B-RL_m23k-RL_PMC",
    "train-qwen2_5_vl_32b-m23k-step_645": "MedVLThinker-32B-RL_m23k",
}

# Initialize HuggingFace API
api = HfApi()


def expand_path(path):
    """Expand ~ and resolve absolute path"""
    return Path(path).expanduser().resolve()


def get_model_path(local_name):
    """Get the full path to a model based on its local name"""
    if local_name in ["3b-m23k-checkpoint-4401", "7b-m23k-checkpoint-4401"]:
        base_path = BASE_PATHS["sft-m23k"]
    elif local_name in ["3b-pmc_vqa-checkpoint-12594", "7b-pmc_vqa-checkpoint-12594"]:
        base_path = BASE_PATHS["sft-pmc_vqa"]
    else:
        base_path = BASE_PATHS["converted"]

    return expand_path(base_path) / local_name


print("Setup complete. Ready to upload models to UCSC-VLAA organization.")


# %%
def create_model_card(model_name, local_name):
    """Generate a model card for the uploaded model"""

    # Determine model size and training type
    if "3B" in model_name:
        size = "3B"
        base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif "7B" in model_name:
        size = "7B"
        base_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif "32B" in model_name:
        size = "32B"
        base_model = "Qwen/Qwen2.5-VL-32B-Instruct"

    training_type = ""
    if "SFT" in model_name and "RL" in model_name:
        training_type = "Supervised Fine-tuning + Reinforcement Learning"
    elif "SFT" in model_name:
        training_type = "Supervised Fine-tuning"
    elif "RL" in model_name:
        training_type = "Reinforcement Learning"

    dataset_info = ""
    if "m23k" in model_name and "PMC" in model_name:
        dataset_info = "Med23k + PMC-VQA datasets"
    elif "m23k" in model_name:
        dataset_info = "Med23k dataset"
    elif "PMC" in model_name:
        dataset_info = "PMC-VQA dataset"

    model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- vision-language
- medical
- multimodal
- qwen2.5-vl
datasets:
- med-vlrm/med-vlm-m23k
- med-vlrm/med-vlm-pmc_vqa
language:
- en
pipeline_tag: image-text-to-text
---

# {model_name}

## Model Description

{model_name} is a {size} parameter medical vision-language model based on Qwen2.5-VL. 
This model has been trained using {training_type.lower()} on {dataset_info}.

## Model Details

- **Base Model**: {base_model}
- **Model Size**: {size} parameters
- **Training Method**: {training_type}
- **Training Data**: {dataset_info}

## Usage

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "UCSC-VLAA/{model_name}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("UCSC-VLAA/{model_name}")

# Example usage
messages = [
    {{
        "role": "user",
        "content": [
            {{
                "type": "image",
                "image": "path/to/medical/image.jpg",
            }},
            {{"type": "text", "text": "What can you see in this medical image?"}},
        ],
    }}
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## Citation

```bibtex
@article{{medvlthinker2025,
  title={{MedVLThinker: Simple Baselines for Multimodal Medical Reasoning}},
  author={{Your Team}},
  journal={{arXiv preprint}},
  year={{2025}}
}}
```

## License

This model is released under the Apache 2.0 license.
"""
    return model_card


def upload_model(local_name, repo_name, commit_message=None):
    """Upload a model to HuggingFace Hub"""

    model_path = get_model_path(local_name)

    # Check if model path exists
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    repo_id = f"{ORGANIZATION}/{repo_name}"

    try:
        # Create repository if it doesn't exist
        logger.info(f"Creating repository: {repo_id}")
        create_repo(repo_id=repo_id, exist_ok=True, private=False, repo_type="model")

        # Generate and upload model card
        logger.info(f"Generating model card for {repo_name}")
        model_card = create_model_card(repo_name, local_name)

        # Save model card to a temporary file
        # readme_path = model_path / "README.md"
        # with open(readme_path, "w", encoding="utf-8") as f:
        #     f.write(model_card)

        # Upload the entire model folder
        logger.info(f"Uploading model from {model_path} to {repo_id}")

        if commit_message is None:
            commit_message = f"Upload {repo_name} model weights"

        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            allow_patterns=None,  # Upload all files
            ignore_patterns=[".git*", "__pycache__*", "*.pyc"],
        )

        logger.info(f"Successfully uploaded {repo_name} to {repo_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload {repo_name}: {str(e)}")
        return False


print("Model upload functions defined.")


# %%
def check_model_exists(local_name):
    """Check if a model exists locally"""
    model_path = get_model_path(local_name)
    exists = model_path.exists()
    logger.info(
        f"Model {local_name}: {'EXISTS' if exists else 'NOT FOUND'} at {model_path}"
    )
    return exists


def upload_all_models(dry_run=True, models_to_upload=None):
    """Upload all models or a specific subset"""

    if models_to_upload is None:
        models_to_upload = list(MODEL_MAPPING.keys())

    logger.info(
        f"{'DRY RUN: ' if dry_run else ''}Planning to upload {len(models_to_upload)} models"
    )

    # First, check which models exist
    existing_models = []
    missing_models = []

    for local_name in models_to_upload:
        if check_model_exists(local_name):
            existing_models.append(local_name)
        else:
            missing_models.append(local_name)

    if missing_models:
        logger.warning(f"Missing models: {missing_models}")

    if dry_run:
        logger.info(f"DRY RUN: Would upload {len(existing_models)} models:")
        for local_name in existing_models:
            repo_name = MODEL_MAPPING[local_name]
            logger.info(f"  {local_name} -> UCSC-VLAA/{repo_name}")
        return existing_models

    # Actually upload the models
    successful_uploads = []
    failed_uploads = []

    for local_name in existing_models:
        repo_name = MODEL_MAPPING[local_name]
        logger.info(f"Uploading {local_name} as {repo_name}...")

        if upload_model(local_name, repo_name):
            successful_uploads.append((local_name, repo_name))
        else:
            failed_uploads.append((local_name, repo_name))

    # Summary
    logger.info(f"Upload completed!")
    logger.info(f"Successful uploads: {len(successful_uploads)}")
    for local_name, repo_name in successful_uploads:
        logger.info(f"  ✓ {local_name} -> UCSC-VLAA/{repo_name}")

    if failed_uploads:
        logger.error(f"Failed uploads: {len(failed_uploads)}")
        for local_name, repo_name in failed_uploads:
            logger.error(f"  ✗ {local_name} -> UCSC-VLAA/{repo_name}")

    return successful_uploads, failed_uploads


def upload_single_model(local_name, dry_run=True):
    """Upload a single model"""
    if local_name not in MODEL_MAPPING:
        logger.error(f"Unknown model: {local_name}")
        logger.info(f"Available models: {list(MODEL_MAPPING.keys())}")
        return False

    if not check_model_exists(local_name):
        logger.error(f"Model {local_name} does not exist locally")
        return False

    repo_name = MODEL_MAPPING[local_name]

    if dry_run:
        logger.info(f"DRY RUN: Would upload {local_name} -> UCSC-VLAA/{repo_name}")
        return True

    return upload_model(local_name, repo_name)


print("Batch upload functions defined.")

# %%
# Check which models are available locally
print("Checking local model availability...")
available_models = upload_all_models(dry_run=False)

# %%
# Example: Upload a single model (dry run first)
# Uncomment and modify as needed

# model_to_upload = "3b-m23k-checkpoint-4401"  # Change this to the model you want to upload
# print(f"Testing upload for: {model_to_upload}")
# upload_single_model(model_to_upload, dry_run=True)

# %%
# ACTUAL UPLOAD - UNCOMMENT TO EXECUTE
# WARNING: This will actually upload models to HuggingFace Hub
# Make sure you have the necessary permissions and HF_TOKEN set

# Option 1: Upload a single model
# model_name = "3b-m23k-checkpoint-4401"  # Replace with desired model
# result = upload_single_model(model_name, dry_run=False)
# print(f"Upload result: {result}")

# Option 2: Upload specific models
# models_to_upload = [
#     "3b-m23k-checkpoint-4401",
#     "7b-m23k-checkpoint-4401",
#     # Add more models as needed
# ]
# successful, failed = upload_all_models(dry_run=False, models_to_upload=models_to_upload)

# Option 3: Upload ALL available models (use with caution!)
# successful, failed = upload_all_models(dry_run=False)

# %% [markdown]
# # HuggingFace Upload Setup Instructions
#
# ## Prerequisites
#
# 1. **Install required packages:**
#    ```bash
#    pip install huggingface_hub transformers
#    ```
#
# 2. **Set up HuggingFace authentication:**
#    - Get your HF token from https://huggingface.co/settings/tokens
#    - Set environment variable: `export HF_TOKEN=your_token_here`
#    - Or login via CLI: `huggingface-cli login`
#
# 3. **Ensure you have write access to UCSC-VLAA organization**
#
# ## Model Mapping
#
# The script will upload models with the following mapping:
#
# - `3b-m23k-checkpoint-4401` → `UCSC-VLAA/MedVLThinker-3B-SFT_m23k`
# - `7b-m23k-checkpoint-4401` → `UCSC-VLAA/MedVLThinker-7B-SFT_m23k`
# - `3b-pmc_vqa-checkpoint-12594` → `UCSC-VLAA/MedVLThinker-3B-SFT_PMC`
# - `7b-pmc_vqa-checkpoint-12594` → `UCSC-VLAA/MedVLThinker-7B-SFT_PMC`
# - And more RL models...
#
# ## Usage
#
# 1. **Run the check cell first** to see which models are available locally
# 2. **Test with dry run** to verify everything looks correct
# 3. **Execute actual upload** by uncommenting the appropriate cells
#
# ## Troubleshooting
#
# - If upload fails, check your HF token and organization permissions
# - Large models may take a long time to upload
# - The script will automatically generate README.md files for each model
