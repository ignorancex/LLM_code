"""
Minimal script to reproduce zero-shot results on pathological task
"""
from tqdm import tqdm
import json
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows, preprocess
from torch.utils.data import DataLoader

# Import your modules
from EEGClip.clip_models import EEGClipModel
from EEGClip.text_preprocessing import text_preprocessing
import configs.preprocess_config as preprocess_config

# Configuration
CHECKPOINT_PATH = "/nfs/norasys/notebooks/camaret/neuro_ai/EEGClip/results/hydra/single_run/2025-07-16/18-31-43/results/models/EEGClip_100_WhereIsAI/UAE-Large-V1_64.ckpt"  # Update this
TUH_DATA_DIR = preprocess_config.tuh_data_dir
N_RECORDINGS = 2993  # Adjust as needed
BATCH_SIZE = 64
NUM_WORKERS = 16

# Load configuration for preprocessing
SFREQ = 100  # Update based on your config
INPUT_WINDOW_SAMPLES = 1200  # Update based on your config
N_PREDS_PER_INPUT = 519  # Update based on your config
N_MINUTES = 2  # Update based on your config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model from checkpoint
    print("Loading model from checkpoint...")
    model = EEGClipModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location=device
    )
    model.eval()
    model.to(device)
    
    # Load the zero-shot sentence embeddings
    with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
        zc_sentences_emb_dict = json.load(f)
    
    # Get pathological task embeddings
    text_encoder_name = model.hparams.text_encoder_name
    pathological_emb = zc_sentences_emb_dict[text_encoder_name]["pathological"]
    
    # Convert to tensors and project through text projection head
    s0 = torch.Tensor(pathological_emb["s0"]).to(device).unsqueeze(0)
    s1 = torch.Tensor(pathological_emb["s1"]).to(device).unsqueeze(0)
    
    with torch.no_grad():
        s0_proj = model.text_projection(s0).squeeze(0)
        s1_proj = model.text_projection(s1).squeeze(0)
    
    # Create text embedding matrix for zero-shot classification
    text_embeddings = torch.stack([s0_proj, s1_proj]).T  # [dim, 2]
    
    # Load TUH dataset
    print("Loading TUH dataset...")
    dataset = TUHAbnormal(
        path=TUH_DATA_DIR,
        recording_ids=range(N_RECORDINGS),
        target_name="report",
        preload=False,
        add_physician_reports=True,
        n_jobs=NUM_WORKERS,
    )
    
    # Text preprocessing
    dataset.set_description(
        text_preprocessing(dataset.description, processed_categories="all"),
        overwrite=True,
    )
    
    # EEG preprocessing
    print("Preprocessing EEG data...")
    
    
    # Split dataset (using validation set for evaluation)
    valid_set = dataset.split("train")["False"]
    preprocess(valid_set, preprocess_config.preprocessors)
    # Create windowed dataset
    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * SFREQ,
        stop_offset_samples=60 * SFREQ + N_MINUTES * 60 * SFREQ,
        preload=True,
        window_size_samples=INPUT_WINDOW_SAMPLES,
        window_stride_samples=N_PREDS_PER_INPUT,
        drop_last_window=False,
    )
    
    # Create dataloader
    valid_loader = DataLoader(
        window_valid_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    
    # Perform zero-shot classification
    print("Performing zero-shot classification...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            eeg_batch = batch[0].to(device)
            string_batch = batch[1]
            
            # Extract ground truth labels
            labels = []
            for string in string_batch:
                if "pathological: true" in string.lower():
                    labels.append(1)
                else:
                    labels.append(0)
            
            # Get EEG features
            eeg_features = model.eeg_encoder(eeg_batch)
            eeg_features_proj = model.eeg_projection(eeg_features)
            eeg_features_proj = torch.mean(eeg_features_proj, dim=1)
            
            # Compute similarity scores
            logits = eeg_features_proj @ text_embeddings
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            #if batch_idx % 10 == 0:
            #    print(f"Processed {batch_idx}/{len(valid_loader)} batches")
    
    # Calculate balanced accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    print(f"\nZero-shot Results on Pathological Task:")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Positive samples: {np.sum(all_labels)} ({np.mean(all_labels):.2%})")
    
    # Additional metrics
    from sklearn.metrics import classification_report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Non-pathological', 'Pathological']))

if __name__ == "__main__":
    main()