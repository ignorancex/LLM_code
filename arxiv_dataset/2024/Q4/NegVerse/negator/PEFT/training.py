
from .config import *
import warnings
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data_preprocess import TextGenerationSetup, process_dataframe, Trainer_preprocess, process_dataframe_general, process_data, load_data, split_dataset,process_and_blank_sentences
from .training_helper import setup_model,create_model_directories, create_peft_model, train_engine
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model,PeftModel
from .inference import get_outputs, NegationModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from ..Evaluation.evaluate import evaluate_model
from .config import TEST_SIZE


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

# Suppress all warnings
warnings.filterwarnings("ignore")

set_random_seed(SEED)

# Setup the model
text_format = TextGenerationSetup(MODEL_PATH)

# Load dataset
all_dataset = load_data(text_format)

# Split data in Train - Validation
train_dataset, val_dataset = split_dataset(all_dataset)

# Tokenize
train_dataset = train_dataset.map(text_format.tokenize_function)
val_dataset = val_dataset.map(text_format.tokenize_function)

# Load model setup
device,tokenizer, model = setup_model(MODEL_PATH)

# Prepare the dataset for training
print("\n Creating data loader")
prep_train = Trainer_preprocess(tokenizer, BATCH_SIZE)
train_dataloader, train_data, tok_data = prep_train.create_data_loader(train_dataset)
valid_dataloader, valid_data, _ = prep_train.create_data_loader(val_dataset)

# Create directory to save the trained model
working_dir="./"
output_dir_name="peft_outputs"
output_directory = create_model_directories(working_dir, output_dir_name)

# Training process
print("\n Training")

# Create new model for soft tuning 
model_peft = create_peft_model(NUM_VIRTUAL_TOKENS)
model_peft.to(device)

trained_model = train_engine(train_dataloader,valid_dataloader,NUM_EPOCHS, model_peft, device,NUM_VIRTUAL_TOKENS)

# Save the trained model
trained_model.save_pretrained(output_directory)

# Inference
input_prompt = "His behavior is always responsible. <|perturb|> [negation] [BLANK]"

negation_model = NegationModel(output_directory)
outputs = negation_model.infer(input_prompt, num_beams=5, num_return_sequences=3)

print("Trained Model Outputs:")
print(outputs["trained_model_outputs"])
print("\nOriginal Model Outputs:")
print(outputs["original_model_outputs"])

data_path = './negator/data/nli/SNLI.txt'
test_data  = process_and_blank_sentences(data_path, sample_size=300, max_blank=2,max_sent=6)

#test_data = [text.split('[SEP]')[0].strip() for text in test_data['input_text']]
#test_data = test_data[:8]

evaluate_model(test_data)

