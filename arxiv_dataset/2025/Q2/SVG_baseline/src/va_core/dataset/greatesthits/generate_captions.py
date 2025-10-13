import os
from transformers import AutoProcessor, Blip2ForConditionalGeneration

import torch
import decord

video_path = "/groups/gce50978/dataset/GreatestHits/vis-data-256"
text_prefix = "hitting objects with a drumstick"
mp4_dir = "data"

# BLIP-2
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for mode in ["train", "test"]:
    with open(os.path.join(video_path, mode + ".txt"), "r") as f:
        lines = f.readlines()

    savelines = ""
    for line in lines:
        line = line.strip()
        filename = os.path.join(video_path, line + "_denoised.mp4")
        with open(filename, 'rb') as f:
            vr = decord.VideoReader(f, ctx=decord.cpu(0))

        # simple prompted
        prompt = "a person is holding a drumstick in a scene with"
        inputs = processor(vr[0].asnumpy(), text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        generated_text = "hitting " + generated_text + " with a drumstick"
        print(f"{line}: {generated_text}")

        # save results
        savelines += mp4_dir + "/" + line + "_denoised.mp4"
        savelines += ",\"" + generated_text + "\""
        savelines += "\n"

    with open("./flist_" + mode + ".txt", "w") as f:
        f.write(savelines)
