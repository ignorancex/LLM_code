import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_embeddings(paragraphs, model_name_or_path='jinaai/jina-embeddings-v3', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True)
    model = model.to(device)
    
    embeddings = []
    batch_size = 10
    for i in tqdm(range(0, len(paragraphs), batch_size)):
        batch_paragraphs = paragraphs[i:i+batch_size]
        if isinstance(batch_paragraphs, str):
            batch_paragraphs = [batch_paragraphs]
        inputs = tokenizer(batch_paragraphs, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled_output = outputs.last_hidden_state.mean(dim=1).squeeze()  # Mean pooling

        if len(batch_paragraphs) == 1:
            pooled_output = pooled_output.unsqueeze(0)

        embeddings.extend(pooled_output.cpu().tolist())


    return embeddings