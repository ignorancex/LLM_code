import fasttext
from huggingface_hub import hf_hub_download
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

model_path = hf_hub_download(repo_id="facebook/fasttext-nl-vectors", filename="model.bin")
fasttext_model = fasttext.load_model(model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--annotations_file",
        type=str,
        default='SSL-NL/annotations/word-rsa_annotations.csv',
        help='filepath to the word-rsa annotations file'
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        default="embeddings",
        help='directory to save the embeddings to'
    )
    args, unk_args = parser.parse_known_args()

    word_rsa_data = pd.read_csv(args.annotations_file)
    emb_dir = Path(args.emb_dir)
    emb_dir.mkdir(exist_ok=True, parents=True)

    fasttext_embs = {
        'MLS': np.stack(
        [fasttext_model[w] for w in word_rsa_data[word_rsa_data['subset'] == 'MLS']['word'].values]
        ),
        'IFADV': np.stack(
        [fasttext_model[w] for w in word_rsa_data[word_rsa_data['subset'] == 'IFADV']['word'].values]
        ),
    }

    save_path = emb_dir / f'fasttext_word-rsa_embs.pkl'
    pickle.dump(fasttext_embs, open(save_path, 'wb'))
    print(f'Done! Saved embeddings to {save_path}')