import torch
from labml import monit
from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer
class ChunkEmbeddings:
    def __init__(self, device: torch.device, normalize_emb_bool=False, gpt2_bool=True,embedding_type="multi-qa-mpnet-base-dot-v1"):
        self.device = device
        self.normalize_emb_bool = normalize_emb_bool
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if embedding_type == "Alibaba-NLP/gte-base-en-v1.5":
            self.model = SentenceTransformer(embedding_type, trust_remote_code=True)
        else:
            self.model = SentenceTransformer(embedding_type)
        self.model.to(device)
    def __call__(self, chunks):
        with torch.no_grad():
            detokenized_chunks = self.gpt2_tokenizer.batch_decode(chunks, skip_special_tokens=True)
            unnormalized_emb = self.model.encode(detokenized_chunks)
            unnormalized_emb = torch.from_numpy(unnormalized_emb)
            if self.normalize_emb_bool:
                normalized_emb = torch.nn.functional.normalize(unnormalized_emb)
                return normalized_emb
            else:
                return unnormalized_emb
if __name__ == '__main__':
    pass
