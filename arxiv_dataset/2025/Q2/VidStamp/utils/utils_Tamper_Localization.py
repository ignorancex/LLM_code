import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

def tamper_video(frames, swap_pairs=None, drop_indices=None, insertions=None):
    frames = frames.clone()
    N = frames.shape[0]
    frame_list = [frames[i] for i in range(N)]
    frame_sequence = list(range(N))  # initial mapping

    # Apply swap
    if swap_pairs:
        for i, j in swap_pairs:
            frame_list[i], frame_list[j] = frame_list[j], frame_list[i]
            frame_sequence[i], frame_sequence[j] = frame_sequence[j], frame_sequence[i]

    # Apply drop
    if drop_indices:
        for idx in sorted(drop_indices, reverse=True):
            del frame_list[idx]
            del frame_sequence[idx]

    # Apply insertions
    if insertions:
        for pos, new_frame in sorted(insertions, key=lambda x: x[0]):
            frame_list.insert(pos, new_frame)
            frame_sequence.insert(pos, -1)  # -1 to denote inserted frame

    tampered_frames = torch.stack(frame_list)
    return tampered_frames, frame_sequence


def temporal_tamper_localization(template_keys, tampered_keys, frame_sequence, threshold=0.9):
    predicted_sequence = []
    for i, tampered_key in enumerate(tampered_keys):
        # Compare with all 16 keys
        similarities = (template_keys == tampered_key).sum(dim=1) / tampered_key.shape[0]  # Hamming similarity
        best_match_idx = torch.argmax(similarities).item()
        best_match_acc = similarities[best_match_idx].item()

        true_index = frame_sequence[i]
        
        if best_match_acc < threshold:
            predicted_sequence.append(-1)
        else:
            predicted_sequence.append(best_match_idx)
    
    results = np.array(predicted_sequence) == np.array(frame_sequence)
    accuracy = results.sum() / len(results)

    return accuracy
