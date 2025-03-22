import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
from typing import List, Union

sys.path.append("encoders/contriever")
sys.path.append(".")
from encoders.contriever.src.contriever import Contriever

# Load the model and tokenizer
model = Contriever.from_pretrained("facebook/contriever")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

model = model.cuda()

batch_size = 1000

def encode_text(text: Union[str, List[str]]):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        inputs = inputs.to(model.device)
        embeddings = model(**inputs)
    
    return embeddings.cpu().numpy()

def encode_queries(queries: List[str], batch_size: int = 64) -> torch.Tensor:
    all_embeddings = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
        batch_queries = queries[i:i+batch_size]
        
        # Using the encode_text function for encoding
        embeddings = encode_text(batch_queries)
        
        # Collect results and move to CPU to prevent GPU memory overflow
        all_embeddings.append(torch.tensor(embeddings))  # Convert to tensor

    # Concatenate all embeddings and return
    return torch.cat(all_embeddings, dim=0) if all_embeddings else None

if __name__ == "__main__":
    queries = []
    with open('data/MS-MARCO/queries.all.tsv', 'r') as f:
        for line in f:
            qid, query = line.split('\t')
            queries.append(query)

    # This will save the embeddings in batches, with each batch saved separately
    embeddings = encode_queries(queries, batch_size=batch_size)
    
    # Move embeddings to CPU before saving
    embeddings = embeddings.cpu() if embeddings is not None else None
    
    # Save the embeddings after all batches
    save_path = f"/shared/eng/pj20/lc/DeepRetrieval/outputs/msmarco_query_embeddings.pt"
    torch.save(embeddings, save_path)