import os
import sys
import torch
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "LanguageBind"))
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="path to csv file containing mp4 paths, wav paths, and prompt")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
    }
    model = LanguageBind(clip_type=clip_type)
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'lb203/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt)
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    flist = pd.read_csv(args.csv_path, header=None)

    cnt = 0
    avg_score_ta = 0.0
    avg_score_tv = 0.0
    avg_score_av = 0.0
    for idx in range(flist.shape[0]):

        mp4_path = flist.iloc[idx][0]
        wav_path = flist.iloc[idx][1]
        prompt = flist.iloc[idx][2]
        if not (os.path.isfile(wav_path) and os.path.isfile(mp4_path)):
            continue

        inputs = {
            'video': to_device(modality_transform['video'](mp4_path), device),
            'audio': to_device(modality_transform['audio'](wav_path), device),
        }
        inputs['language'] = to_device(tokenizer(prompt, max_length=77, padding='max_length',
                                                truncation=True, return_tensors='pt'), device)

        with torch.no_grad():
            embeddings = model(inputs)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        avg_score_ta += cos(embeddings["language"], embeddings["audio"]).detach().cpu().numpy()
        avg_score_tv += cos(embeddings["language"], embeddings["video"]).detach().cpu().numpy()
        avg_score_av += cos(embeddings["video"], embeddings["audio"]).detach().cpu().numpy()
        cnt += 1

    avg_score_ta /= cnt
    avg_score_tv /= cnt
    avg_score_av /= cnt
    print(f"TEXT-AUDIO SCORE: {avg_score_ta}")
    print(f"TEXT-VIDEO SCORE: {avg_score_tv}")
    print(f"AUDIO-VIDEO SCORE: {avg_score_av}")
    print(f"Number of processed data: {cnt}")


if __name__ == "__main__":
    main()