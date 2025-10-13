import json
import configs.preprocess_config as preprocess_config

with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
            zc_sentences_emb_dict = json.load(f)

for text_encoder_name, emb_dict in zc_sentences_emb_dict.items():
    print(f"Text Encoder: {text_encoder_name}")
    for label_name, embs in emb_dict.items():
        print(f"  Label: {label_name}")
        for key, value in embs.items():
            if isinstance(value, list):
                print(f"    {key}: {len(value)} elements")
            else:
                print(f"    {key}: {value}")
