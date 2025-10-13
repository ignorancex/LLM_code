import os
import sys
import torch
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ImageBind"))
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="path to csv file containing mp4 paths, wav paths, and prompt")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

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
            ModalityType.TEXT: data.load_and_transform_text([prompt], device),
            ModalityType.VISION: data.load_and_transform_video_data([mp4_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([wav_path], device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        avg_score_ta += cos(embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO]).detach().cpu().numpy()
        avg_score_tv += cos(embeddings[ModalityType.TEXT], embeddings[ModalityType.VISION]).detach().cpu().numpy()
        avg_score_av += cos(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]).detach().cpu().numpy()
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