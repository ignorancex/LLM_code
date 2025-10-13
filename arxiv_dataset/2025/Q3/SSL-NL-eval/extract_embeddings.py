"""
Example usage:
python extract_embeddings.py --model_id="amsterdamNLP/Wav2Vec2-NL" --segments="phone"
"""

import pandas as pd
import soundfile as sf
import numpy as np
import torch
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SaveOutput:
    def __init__(self):
        self.outputs = defaultdict()
    
    def __call__(self, name):
        def hook(module, module_in, module_out):
            self.outputs[name] = module_out.detach()
        return hook
    
    def clear(self):
        self.outputs = defaultdict()

def get_segment_embeddings(model, feature_extractor, annotations, SSL_NL_dir):
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
    
    embeddings = {
        layer: [] 
        for layer in ['CNN', 'embeds'] + [f'T{i}' for i in range(1, model.config.num_hidden_layers + 1)]
    }

    for r, row in tqdm(annotations.iterrows()):

        audio_filepath = row['audio_filepath']
        audio, sr = sf.read(SSL_NL_dir / audio_filepath)
        
        start_frame = int(np.floor(float(row['start_time']) / time_offset))
        end_frame = int(np.ceil(float(row['end_time']) / time_offset))
        
        # register hooks
        save_output = SaveOutput()
        if type(model) == Wav2Vec2Model:
            last_conv_layer = model.feature_extractor.conv_layers[-1]
            last_conv_layer.activation.register_forward_hook(save_output('CNN'))
            model.encoder.layer_norm.register_forward_hook(save_output('embeds'))
            for i, enc_layer in enumerate(model.encoder.layers):
                enc_layer.final_layer_norm.register_forward_hook(save_output(f'T{i+1}'))
                
        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(device)
        model.eval()
        with torch.no_grad():
            model(inputs, output_hidden_states=True)

        for layer in save_output.outputs.keys():
            # for the CNN output
            if layer.startswith('C'):
                _, projected_CNN_output = model.feature_projection(save_output.outputs[layer].transpose(1,2))
                embeddings[layer].append(projected_CNN_output[:, start_frame:end_frame, :].mean(axis=1).detach().cpu().numpy().flatten())
            
            # for the Transformer embeds + layer representations
            else:
                embeddings[layer].append(save_output.outputs[layer][:, start_frame:end_frame, :].mean(axis=1).detach().cpu().numpy().flatten())

    for l in embeddings.keys():
        embeddings[l] = np.vstack(embeddings[l])

    return embeddings

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_id",
        required=True,
        type=str,
        help='Huggingface ID for the Wav2Vec2 model to extract embeddings from (e.g. "amsterdamNLP/Wav2Vec2-NL"), or path to local file (e.g. "models/nonspeech_model")',
    )
    parser.add_argument(
        "--segments",
        required=True,
        type=str,
        help='which segments to extract embeddings for, i.e. which annotations to load (one of ["phone", "word-clustering", "word-rsa"])'
    )
    parser.add_argument(
        "--SSL_NL_dir",
        type=str,
        default="SSL-NL",
        help='path to the SSL-NL directory'
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        default="embeddings",
        help='directory to save the extracted embeddings to'
    )
    args, unk_args = parser.parse_known_args()

    SSL_NL_dir = Path(args.SSL_NL_dir)
    emb_dir = Path(args.emb_dir)
    emb_dir.mkdir(exist_ok=True, parents=True)
    
    segment_annotations = pd.read_csv(SSL_NL_dir / f'annotations/{args.segments}_annotations.csv')

    model = Wav2Vec2Model.from_pretrained(args.model_id)
    model.to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_id)

    print(f'Extracting {args.segments} embeddings from {args.model_id} ({len(segment_annotations)} segments)...')
    segment_embeddings = get_segment_embeddings(model, feature_extractor, segment_annotations, SSL_NL_dir)

    save_path = emb_dir / f'{args.model_id.replace("/", "_")}_{args.segments}_embs.pkl'
    pickle.dump(segment_embeddings, open(save_path, 'wb'))
    print(f'Done! Saved embeddings to {save_path}')