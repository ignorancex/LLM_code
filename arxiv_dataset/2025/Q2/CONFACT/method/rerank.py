import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import pickle as pkl
import statistics
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import numpy as np
import os
from config import parse_args
import sys
sys.path.append("..")
from utils import extract_domain, load_pkl

    
# Define the MediaCredibilityPredictor class
class MediaCredibilityPredictor(nn.Module):
    def __init__(self, model_name: str = 'google/bigbird-roberta-large', dropout_rate: float = 0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = 1024  # BigBird-RoBERTa hidden size

        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # use last_hidden_state as representation
        last_hidden_state = outputs.last_hidden_state
        
        cls_embedding = last_hidden_state[:, 0]
        
        pooled_output = self.dropout(cls_embedding)
        prediction = self.regressor(pooled_output)   
        return prediction

def load_model(model_path: str, device: torch.device):
    model = MediaCredibilityPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-large')
    return model, tokenizer


def normalize(scores: List[float]) -> np.ndarray:
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


# Define the MediaCredibilityPredictor class
class MediaCredibilityPredictor(nn.Module):
    def __init__(self, model_name: str = 'google/bigbird-roberta-large', dropout_rate: float = 0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = 1024  # BigBird-RoBERTa hidden size

        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(cls_embedding)
        prediction = self.regressor(pooled_output)
        return prediction

# Load the tokenizer and model
def load_model(model_path: str, device: torch.device):
    model = MediaCredibilityPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-large')
    return model, tokenizer

# Normalize scores
def normalize(scores: List[float]) -> np.ndarray:
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

# Rerank evidence
def rerank_evidence(
    retrieved_results: List[Dict[str, Any]],
    credibility_data: Dict[str, Dict[str, Any]],
    mbfc_credibility_data: Dict[str, Dict[str, Any]],
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 100,
    media_data: str = 'all'
) -> List[Dict[str, Any]]:

    for qn_entity in tqdm(retrieved_results, desc="Processing queries"):
        media_descriptions = []
        bm25_scores = []
        has_mbfc = []

        # Prepare media descriptions and BM25 scores
        for sentence_entity in qn_entity["top_k"]:
            bm25_scores.append(sentence_entity['score'])
            domain = extract_domain(sentence_entity['original_link'])

            if media_data == 'all' or domain in mbfc_credibility_data:
                has_mbfc.append(1 if domain in mbfc_credibility_data else 0)
                media_descriptions.append(credibility_data[domain]['details'])
            else:
                has_mbfc.append(0)

        # Predict credibility scores in batches
        credibility_scores = []
        for i in range(0, len(media_descriptions), batch_size):
            batch_descriptions = media_descriptions[i:i + batch_size]
            encodings = tokenizer(
                batch_descriptions,
                max_length=1024,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            with torch.no_grad():
                batch_scores = model(input_ids, attention_mask).squeeze().cpu().numpy()
                if batch_scores.ndim == 0:
                    batch_scores = [batch_scores]
                credibility_scores.extend(batch_scores)

        # Normalize scores
        norm_bm25 = normalize(bm25_scores)
        norm_credibility = normalize(credibility_scores) if credibility_scores else []
        median_credibility_score = statistics.median(norm_credibility) if len(norm_credibility) > 0 else 0

        norm_credibility = list(norm_credibility)
        # Assign credibility scores
        credibility = [
            norm_credibility.pop(0) if flag else median_credibility_score
            for flag in has_mbfc
        ]
         
        # Update and sort evidence for the current query
        for e_index, sentence_entity in enumerate(qn_entity["top_k"]):
            sentence_entity["BM25"] = norm_bm25[e_index]
            sentence_entity["Credibility"] = credibility[e_index]

    return retrieved_results


def soft_rerank(retrived_results, beta = 0.8):
    for question in retrived_results:
        for evidence in question['top_k']:
            evidence['final_score'] = evidence['BM25'] + beta * evidence['Credibility']
        # Sort the evidence by final score in descending order
        question['top_k'] = sorted(question['top_k'], key=lambda x: x['final_score'], reverse=True)

    return retrived_results

def hard_rerank(retrived_results, beta = 0.8, gamma = 0.3):
    for question in retrived_results:
        for evidence in question['top_k']:
            credibility_score = 0 if evidence['Credibility'] < gamma else 1
            evidence['final_score'] = evidence['BM25'] + beta * credibility_score
        # Sort the evidence by final score in descending order
        question['top_k'] = sorted(question['top_k'], key=lambda x: x['final_score'], reverse=True)

    return retrived_results


def main():
    args = parse_args()

    # Load data and model
    all_credibility_data = load_pkl("../data/dataset/all_media_data.pkl")
    mbfc_credibility_data = load_pkl("../data/dataset/mbfc_media_data.pkl")
    retrieved_results = load_pkl(f'../results/top{args.n}_retrieved_{args.type}.pkl')

    model_path = args.rerank_model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(model_path, device)

    # Rerank evidence
    reranked_results = rerank_evidence(retrieved_results, all_credibility_data, mbfc_credibility_data, model, tokenizer, device, batch_size=256, media_data = args.media_data)

    results_folder = f'../results/results_{args.media_data}_media'
    if os.path.exists(results_folder):
        os.mkdir(results_folder)

    results_file = os.path.join(results_folder, f'top{args.n}_rerank_scored_{args.type}.pkl')
    with open(results_file, "wb") as f:
        pkl.dump(reranked_results, f)
    
if __name__ == "__main__":
    main()
 