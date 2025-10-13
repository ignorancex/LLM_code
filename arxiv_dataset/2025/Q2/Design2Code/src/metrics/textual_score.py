import torch
import open_clip
import pickle
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def calculate_clip_textual_similarity(text1, text2):
    text1_features = model.encode_text(tokenizer([text1]).to(device))    
    text2_features = model.encode_text(tokenizer([text2]).to(device))

    text1_features /= text1_features.norm(dim=-1, keepdim=True)
    text2_features /= text2_features.norm(dim=-1, keepdim=True)

    similarity = (text1_features @ text2_features.T).item()

    return similarity


if __name__ == "__main__":
    
    reference_path = "./websight_hf/data_20100.pickle"
    reference_descriptions = []
    with open(reference_path, "rb") as file:
        tmp = pickle.load(file)
        for i in range(20000, 20100):
            reference_descriptions.append(tmp[i]['llm_generated_idea'])

    evaluated_path = "results/ws_no_graph/predict.json"
    evaluated_descriptions = []
    with open(evaluated_path, "r") as file:
        tmp = json.load(file)
        for i in range(20000, 20100):
            evaluated_descriptions.append(tmp[str(i)]['description'])
    
    description_similarity = 0
    for i in range(100):
        description_similarity += calculate_clip_textual_similarity(reference_descriptions[i], evaluated_descriptions[i])
    
    description_similarity /= 100
    print(description_similarity)
