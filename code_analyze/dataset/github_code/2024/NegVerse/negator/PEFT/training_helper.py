
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from .config import *

from transformers import AutoTokenizer, AutoModelForCausalLM
from .data_preprocess import TargetType
from ..generation_processing import Special_tokens
from torch.optim import AdamW
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model,PeftModel
from sklearn.model_selection import train_test_split

def setup_model(model_path: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  tokenizer.pad_token = None #self.tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

  # Add a new pad token if it doesn't exist and set it to ID 0
  if tokenizer.pad_token is None:
      tokenizer.add_special_tokens({'pad_token': '[PAD]'})
      tokenizer.pad_token_id = TargetType.PAD.value

  return device,tokenizer, model


def tts_to_labels(inputs, tts, label_tts):
  # Extract only the relevant token ids
  selector = torch.zeros_like(inputs, dtype=torch.bool)
  for tt in label_tts:
    selector |= tts == tt.value
  return torch.where(
      selector,
      inputs,
      torch.full_like(inputs, -1))


def plot_training_loss(train_history, valid_history):
    """
    Plot the training and validation loss over epochs.

    Args:
        train_history (list): List of training loss values for each epoch.
        valid_history (list): List of validation loss values for each epoch.
    """
    train_num_epochs = len(train_history)
    
    # Ensure the validation history has the same length as training history
    if len(valid_history) != train_num_epochs:
        raise ValueError("Training and validation history must have the same length.")
    
    epochs = np.arange(train_num_epochs)  # Array of epoch indices
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_history, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(epochs, valid_history, marker='o', linestyle='-', color='r', label='Validation Loss')
    
    plt.title('Training and Validation Loss per Epoch',fontsize=16, fontweight='bold')
    plt.xlabel('Epoch',fontsize=15)
    plt.ylabel('Loss',fontsize=15)
    
    plt.xticks(ticks=epochs, fontsize=12)  # Set x-axis ticks to integer values with font size
    plt.yticks(fontsize=12)  # Set y-axis ticks with font size
    
    plt.grid(True)
    plt.legend(fontsize=15)
    
    plt.savefig('./negator/figures/training_validation_loss.png')
    plt.close()




def train_fn(train_dataloader,model,device,optimizer,num_virtual_tokens: int)-> float:

  total_loss = 0
  train_context = True
  model.train()

  for batch in train_dataloader:

    inputs, tts = tuple(t.to(device) for t in batch)
    labels_context = tts_to_labels(inputs, tts, [TargetType.CONTEXT,TargetType.CONTEXT_INFILL_SEP])
    labels_infill = tts_to_labels(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL])

    outputs = model(inputs)
    logits =  outputs.logits
    #logits_relevant = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
    logits_relevant = logits[:, num_virtual_tokens:-1].contiguous().view(-1, logits.shape[-1])
    #logits_relevant = logits[:, num_virtual_tokens:].contiguous().view(-1, logits.shape[-1])

    loss_context = F.cross_entropy(
        logits_relevant,
        labels_context[:, 1:].contiguous().view(-1),
        ignore_index=-1)
    
    loss_infill = F.cross_entropy(
        logits_relevant,
        labels_infill[:, 1:].contiguous().view(-1),
        ignore_index=-1)

    loss_context_item = loss_context.item()
    loss_infill_item = loss_infill.item()

    loss = loss_infill

    if train_context:
      loss += loss_context

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    total_loss += loss.item()

  avg_train_loss = total_loss / len(train_dataloader)
  return avg_train_loss


def eval_fn(valid_dataloader, model, device, num_virtual_tokens):
  total_loss = 0
  train_context = True
  model.eval()

  with torch.no_grad():
     for batch in valid_dataloader:
        inputs, tts = tuple(t.to(device) for t in batch)
        labels_context = tts_to_labels(inputs, tts, [TargetType.CONTEXT,TargetType.CONTEXT_INFILL_SEP])
        labels_infill = tts_to_labels(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL])

        outputs = model(inputs)
        logits =  outputs.logits

        logits_relevant = logits[:, num_virtual_tokens:-1].contiguous().view(-1, logits.shape[-1])

        loss_context = F.cross_entropy(
          logits_relevant,
          labels_context[:, 1:].contiguous().view(-1),
          ignore_index=-1)
    
        loss_infill = F.cross_entropy(
          logits_relevant,
          labels_infill[:, 1:].contiguous().view(-1),
          ignore_index=-1)
        
        loss = loss_infill

        if train_context:
           loss += loss_context

        total_loss += loss.item()

  avg_valid_loss = total_loss / len(valid_dataloader)
  return avg_valid_loss


def train_engine(train_loader,valid_loader,train_num_epochs,model, device,num_virtual_tokens):

  train_history = []
  valid_history = []

  train_learning_rate = TRAIN_LEARNING_RATE # 5e-5
  train_adam_epsilon = TRAIN_ADAM_EPSILON #1e-8
  weight_decay = 1e-3

  optimizer = AdamW(
      model.parameters(),
      lr=train_learning_rate,
      weight_decay=weight_decay  # Add weight decay
  )

  for i in range(train_num_epochs):
    train_loss = train_fn(train_loader, model, device,optimizer,num_virtual_tokens)
    valid_loss = eval_fn(valid_loader, model, device, num_virtual_tokens)
    train_history.append(train_loss)
    valid_history.append(valid_loss)
    print(f"Epoch {i} , Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}")

  # Call the plotting function
  plot_training_loss(train_history,valid_history)

  return model


def create_peft_model(num_virtual_tokens: int, model_path: str = "uw-hai/polyjuice"):
  
  foundational_model = AutoModelForCausalLM.from_pretrained(model_path)
  # Define prompt tuning configuration
  peft_config = PromptTuningConfig(
      task_type=TaskType.CAUSAL_LM, # This type indicates the model will generate text.
      prompt_tuning_init=PromptTuningInit.RANDOM,  # The added virtual tokens are initialized with random numbers
      num_virtual_tokens=num_virtual_tokens, # Number of virtual tokens to be added and trained.
      tokenizer_name_or_path=model_path # The pre-trained model.
  )

  peft_model = get_peft_model(foundational_model, peft_config)
  print(peft_model.print_trainable_parameters())

  return peft_model


def create_model_directories(base_dir: str, output_dir_name: str) -> str:
    """
    Creates the base and output directories for storing models if they do not exist.
    
    Parameters:
    base_dir (str): The base working directory.
    output_dir_name (str): The name of the output directory.
    
    Returns:
    str: The path to the output directory.
    """
    # Create the name of the output directory
    output_directory = os.path.join(base_dir, output_dir_name)
    
    # Create the base directory if it does not exist
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    return output_directory