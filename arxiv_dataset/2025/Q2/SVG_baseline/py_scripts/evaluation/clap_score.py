import os
import argparse
import librosa
import torch
from transformers import AutoProcessor, ClapModel


parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", type=str, help="path to audio directory")
args = parser.parse_args()

audio_dir = args.audio_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
model.to(device=device)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

filenames = os.listdir(audio_dir)
cnt = 0
avg_score = 0.0
for fname in filenames:
    if fname[-4:] != ".wav":
        continue

    audio_path = os.path.join(audio_dir, fname)
    input_audio, sr = librosa.load(audio_path, sr=48000)
    input_text = audio_path[:-4].split("_")[-1].replace("-", " ")

    inputs = processor(text=input_text, audios=input_audio, return_tensors="pt", padding=True, sampling_rate=48000)
    inputs.to(device=device)
    outputs = model(**inputs)
    score = cos(outputs.text_embeds, outputs.audio_embeds).detach().cpu().numpy()

    avg_score += score
    cnt += 1

avg_score /= cnt
print(f"Average CLAP score: {avg_score}")
