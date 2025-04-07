import torch

from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reasoner:
    '''
    '''
    def __init__(self, lm: str, learning_rate: float, lexicon: dict, weight_decay: float = 1e-3, device: str = 'cpu') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.model = AutoModelForSequenceClassification.from_pretrained(lm)
        self.optimizer = AdamW(self.model.parameters(), lr = learning_rate, eps=1e-8, weight_decay=weight_decay)
        self.device = device
        
        self.model.to(self.device)
        
        self.lexicon = lexicon
        self.losses = []
        self.accuracies = []
        
    def freeze_parameters(self):
        # if 'axxl' in MODEL:
        #     for param in self.model.albert.embeddings.parameters():
        #         param.requires_grad = False
        # elif 'rl' in MODEL:
        #     for param in self.model.roberta.embeddings.parameters():
        #         param.requires_grad = False
        # elif 'bl' in MODEL:
        #     for param in model.bert.embeddings.parameters():
        #         param.requires_grad = False
        raise NotImplementedError
        
    def adapt(self, premise: dict, labels: torch.Tensor, max_epochs: int = 20, constraint: str = 'accuracy') -> None:
        self.model.train()
        
        premise = premise.to(self.device)
        labels = labels.to(self.device)
        
        for epoch in range(max_epochs):
            adapted = self.model(**premise, labels = labels)
            loss = adapted.loss
            logits = adapted.logits
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.detach().item())
            
            if self.device != 'cpu':
                log_probs = (logits - logits.logsumexp(1).unsqueeze(1)).cpu()
            else:
                log_probs = (logits - logits.logsumexp(1).unsqueeze(1))

            accuracy = accuracy_score(labels.tolist(), log_probs.argmax(1).tolist())
            self.accuracies.append(accuracy)
            if constraint == 'accuracy':
                if accuracy == 1.0:
                    self.stopping_epoch = epoch+1
                    self.loss = loss
                    break

        self.model.eval()
    
    def prepare_stimuli(self, concepts: list, property_phrase: str) -> dict:
        
        sentences = [f'{self.lexicon[c].article} {property_phrase}.' for c in concepts]
        encoded = self.tokenizer(sentences, return_tensors = 'pt', padding = True)
        
        return encoded
    
    def generalize(self, generalization: dict) -> torch.Tensor:
        self.model.eval()
        generalization = generalization.to(self.device)
        with torch.no_grad():
            generalized = self.model(**generalization)
            logits = generalized.logits.detach()
            if self.device != 'cpu':
                log_probs = (logits - logits.logsumexp(1).unsqueeze(1)).cpu()
            else:
                log_probs = (logits - logits.logsumexp(1).unsqueeze(1))
            
        return log_probs