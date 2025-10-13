from get_model import get_moe
import jax
import jax.numpy as jnp
import json
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

def normalize(arr):
    norms = jnp.linalg.norm(arr, axis=1, keepdims=True)
    return arr/norms

moe = get_moe("16k_16")

model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
target_layer = 20

results = {"fr": ([], [], [], []), "es": ([], [], [], []), "de": ([], [], [], []), "en": ([], [], [], [])}
with open("translations.json", "r", encoding="utf-8") as file:
    translations = json.load(file)

for translation in translations:
    en = tokenizer(translation["en"], return_tensors="pt")
    fr = tokenizer(translation["fr"], return_tensors="pt")
    es = tokenizer(translation["es"], return_tensors="pt")
    de = tokenizer(translation["de"], return_tensors="pt")
    with torch.no_grad():
        en_embed = base_model(**en)
        fr_embed = base_model(**fr)
        es_embed = base_model(**es)
        de_embed = base_model(**de)

    embeds = []
    for lang, embed, input in [("fr", fr_embed, fr), ("es", es_embed, es), ("de", de_embed, de), ("en", en_embed, en)]:
        last_token_index = (input.attention_mask[0] == 1).sum() - 1
        last_token_embedding = embed.hidden_states[target_layer][0, last_token_index].numpy().reshape(1, -1)
        embeds.append(normalize(last_token_embedding))

    embeds = jnp.concatenate(embeds, axis=0)
    top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values = jax.vmap(moe.encode)(embeds)
    moe_top_k = top_k_indices.tolist()
    moe_top_k_values = top_k_values.tolist()
    for i, lang in enumerate(["fr", "es", "de", "en"]):
        results[lang][0].append(moe_top_k[i])
        results[lang][1].append(moe_top_k_values[i])
pickle.dump(results, open("translation_results.pkl", "wb"))
