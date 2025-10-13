from pydoc import text
import torch
import tqdm
import argparse
import json
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import balanced_accuracy_score
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.preprocessing import (
    create_fixed_length_windows,
    preprocess,
)

from EEGClip.clip_models import EEGClipModel
from transformers import AutoModel, AutoTokenizer
import configs.preprocess_config as preprocess_config
from EEGClip.text_preprocessing import text_preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an EEG classifier on the TUH EEG dataset."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="pathological",
        help="classification task name (pathological, age, gender, report-related tasks ....",
    )
    parser.add_argument(
        "--n_rec",
        type=int,
        default=2993,
        help="Number of recordings to load from TUH EEG dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers to use for data loading.",
    )
    args = parser.parse_args()

    task_name = (
        args.task_name
    ) 

    tuh_data_dir = preprocess_config.tuh_data_dir
    num_workers = args.num_workers
    if task_name in ["pathological", "age", "gender"]:
        target_name = task_name
    n_recordings_to_load = args.n_rec
    mapping = None


    batch_size = 64
    dataset = TUHAbnormal(
            path=tuh_data_dir,
            recording_ids=range(
                n_recordings_to_load
            ),  # loads the n chronologically first recordings
            target_name=target_name,  # age, gender, pathology
            preload=False,
            add_physician_reports=False,
            n_jobs=args.num_workers,
        )
    
    dataset.set_description(
        text_preprocessing(dataset.description, processed_categories="all"),
        overwrite=True,
    )

    

    valid_set = dataset.split("train")["False"]

    preprocess(valid_set, preprocess_config.preprocessors)
    
    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200
    n_preds_per_input = (
        519  # get_output_shape(eeg_classifier_model, n_chans, input_window_samples)[2]
    )
    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        mapping=mapping,
    )

    window_valid_set.transform = lambda x: x * 1e6

    valid_loader = torch.utils.data.DataLoader(
        window_valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"Loaded {len(valid_loader.dataset)} recordings for validation.")

    print("Loading EEGClip model...")

    eegclipmodel = EEGClipModel.load_from_checkpoint(
            "/nfs/norasys/notebooks/camaret/neuro_ai/EEGClip/results/hydra/single_run/2025-07-16/18-31-43/results/models/EEGClip_100_WhereIsAI/UAE-Large-V1_64.ckpt",
            #preprocess_config.model_paths["eegclip_bert"],
            #strict=False,
        )
    
    # put model to gpu
    eegclipmodel.cuda()


    EEGEncoder = torch.nn.Sequential(
        eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
    )

    for param in EEGEncoder.parameters():
            param.requires_grad = False
            
    print(EEGEncoder)

    text_projection = eegclipmodel.text_projection
    text_projection.eval()


    text_encoder_name = "WhereIsAI/UAE-Large-V1"
    #tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    #language_model = AutoModel.from_pretrained(text_encoder_name)

    label_name = "pathological"
    with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
                zc_sentences_emb_dict = json.load(f)
    emb_dict = zc_sentences_emb_dict[text_encoder_name][label_name]
    s0_embed, s1_embed = (
                        torch.tensor(emb_dict["s0"]),
                        torch.tensor(emb_dict["s1"]),
                    )
    
    s0_embed, s1_embed = s0_embed.cuda(), s1_embed.cuda()
    
    s0_embed = text_projection(s0_embed.unsqueeze(0)).squeeze(0)
    s1_embed = text_projection(s1_embed.unsqueeze(0)).squeeze(0)

    s0_embed = s0_embed.detach().cpu().numpy()
    s1_embed = s1_embed.detach().cpu().numpy()


    embeddings = []
    labels = []

    for batch in tqdm.tqdm(valid_loader):
        eeg, label, id = batch
        print("eeg shape :", eeg.shape) # batch_size, n_chans, n_samples
        eeg = eeg.cuda()
        eeg = EEGEncoder(eeg)
        print("eeg shape after encoder :", eeg.shape) # batch_size, n_pred, n_embed
        eeg = torch.mean(eeg, dim=1)
        embeddings.append(eeg.detach().cpu().numpy())
        labels.append(label)


    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=-1)

    distance_classifier = []
    for r in embeddings:
        print(r.shape, s0_embed.shape, s1_embed.shape)
        d0 = distance.cosine(r, s0_embed)
        d1 = distance.cosine(r, s1_embed)

        
        if d0 < d1:
            distance_classifier.append(0)
        else:
            distance_classifier.append(1)

    print("label balance :", np.mean(distance_classifier))

    # compare to the actual labels
    print("Accuracy: ", balanced_accuracy_score(labels, distance_classifier))