"""
Script to compute gradients that point to a sentence (non-pathological) in EEGClip model
Adapted from braindecode's amplitude gradient computation
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from skorch.utils import to_numpy, to_tensor
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows, preprocess

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


def compute_amplitude_gradients_eegclip(model, dataset, batch_size, target_text_embedding):
    """
    Compute amplitude gradients for EEGClip model pointing towards target text embedding
    
    Args:
        model: EEGClip model
        dataset: Windowed dataset
        batch_size: Batch size for processing
        target_text_embedding: The text embedding to compute gradients towards (e.g., s0)
    
    Returns:
        Array of amplitude gradients with shape (n_samples, n_channels, n_freqs)
    """
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    all_amp_grads = []
    
    for batch_idx, batch in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}")
        batch_X = batch[0]
        
        # Compute gradients for this batch
        amp_grads = compute_amplitude_gradients_for_batch(
            model, batch_X, target_text_embedding
        )
        all_amp_grads.append(amp_grads)
    
    # Concatenate all gradients
    all_amp_grads = np.concatenate(all_amp_grads, axis=0)
    return all_amp_grads


def compute_amplitude_gradients_for_batch(model, X, target_text_embedding):
    """
    Compute amplitude gradients for a single batch
    
    Args:
        model: EEGClip model
        X: Input EEG data (batch_size, n_channels, n_samples)
        target_text_embedding: Target text embedding to maximize similarity with
    
    Returns:
        Amplitude gradients (batch_size, n_channels, n_freqs)
    """
    device = next(model.parameters()).device
    
    # Compute FFT
    ffted = np.fft.rfft(X.numpy(), axis=2)
    amps = np.abs(ffted)
    phases = np.angle(ffted)
    
    # Convert to tensors with gradient tracking
    amps_th = to_tensor(amps.astype(np.float32), device=device).requires_grad_(True)
    phases_th = to_tensor(phases.astype(np.float32), device=device).requires_grad_(True)
    
    # Reconstruct complex FFT coefficients
    fft_coefs = amps_th.unsqueeze(-1) * torch.stack(
        (torch.cos(phases_th), torch.sin(phases_th)), dim=-1
    )
    
    # Convert back to time domain
    complex_fft_coefs = torch.view_as_complex(fft_coefs)
    iffted = torch.fft.irfft(complex_fft_coefs, n=X.shape[2], dim=2)
    
    # Forward pass through EEGClip
    eeg_features = model.eeg_encoder(iffted)
    eeg_features_proj = model.eeg_projection(eeg_features)
    eeg_features_proj = torch.mean(eeg_features_proj, dim=1)  # Average over time
    
    # Normalize features (as done in CLIP)
    eeg_features_norm = eeg_features_proj / eeg_features_proj.norm(dim=-1, keepdim=True)
    target_norm = target_text_embedding / target_text_embedding.norm(dim=-1, keepdim=True)
    
    # Compute similarity with target text embedding
    similarity = torch.sum(eeg_features_norm * target_norm.unsqueeze(0), dim=1)
    
    # Compute gradients for each sample in the batch
    batch_size = X.shape[0]
    amp_grads = np.zeros_like(amps)
    
    for i in range(batch_size):
        # Backward pass for each sample
        if i > 0:
            amps_th.grad.zero_()
        
        similarity[i].backward(retain_graph=(i < batch_size - 1))
        amp_grads[i] = to_numpy(amps_th.grad[i].clone())
    
    return amp_grads


def main(label_name, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model from checkpoint
    print("Loading model from checkpoint...")
    model = EEGClipModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location=device
    )
    model.eval()
    model.to(device)
    
    # Freeze all parameters except those we're computing gradients for
    for param in model.parameters():
        param.requires_grad = False
    
    # Load the zero-shot sentence embeddings
    print("Loading zero-shot embeddings...")
    with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
        zc_sentences_emb_dict = json.load(f)
    
    # Get s (non-pathological) embedding
    text_encoder_name = model.hparams.text_encoder_name
    pathological_emb = zc_sentences_emb_dict[text_encoder_name][label_name]
    

    
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
    
    
    # Use validation set
    valid_set = dataset.split("train")["False"]
    print("Preprocessing EEG data...")
    preprocess(valid_set, preprocess_config.preprocessors)
    
    # Create windowed dataset
    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * SFREQ,
        stop_offset_samples=60 * SFREQ + N_MINUTES * 60 * SFREQ,
        preload=True,  # Set to True for gradient computation
        window_size_samples=INPUT_WINDOW_SAMPLES,
        window_stride_samples=N_PREDS_PER_INPUT,
        drop_last_window=False,
    )
    
    print(f"Computing amplitude gradients for {len(window_valid_set)} windows...")
    # Convert to tensor and project through text projection head
    for sentence in "s2 s3 s4 s5 s6 s7 s8 s9 s10 s11".split():
        print(f"Processing sentence: {sentence}")
        s = torch.Tensor(pathological_emb[sentence]).to(device).unsqueeze(0)
        
        with torch.no_grad():
            s_proj = model.text_projection(s).squeeze(0)
        
        print(f"Target text embedding shape: {s_proj.shape}")
        # Compute amplitude gradients
        amp_grads = compute_amplitude_gradients_eegclip(
            model, 
            window_valid_set, 
            BATCH_SIZE, 
            s_proj
        )
        
        print(f"Amplitude gradients shape: {amp_grads.shape}")
        
        # Save results
        np.save("amplitude_gradients_"+label_name+"_"+sentence+".npy", amp_grads)
        print("Saved amplitude gradients to amplitude_gradients_"+label_name+"_"+sentence+".npy")
        
        # Compute some statistics
        print("\nGradient statistics:")
        print(f"Mean absolute gradient: {np.mean(np.abs(amp_grads)):.6f}")
        print(f"Max absolute gradient: {np.max(np.abs(amp_grads)):.6f}")
        print(f"Std of gradients: {np.std(amp_grads):.6f}")
        
        # Find frequency bands with highest average gradients
        mean_grads_per_freq = np.mean(np.abs(amp_grads), axis=(0, 1))
        freqs = np.fft.rfftfreq(INPUT_WINDOW_SAMPLES, 1/SFREQ)
        
        top_freq_indices = np.argsort(mean_grads_per_freq)[-10:]
        print("\nTop 10 frequencies with highest average gradients:")
        for idx in reversed(top_freq_indices):
            print(f"  {freqs[idx]:.2f} Hz: {mean_grads_per_freq[idx]:.6f}")
        


if __name__ == "__main__":
    main("additional_sentences")