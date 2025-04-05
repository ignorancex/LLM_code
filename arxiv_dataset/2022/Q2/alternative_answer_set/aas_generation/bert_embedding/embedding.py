from bert_serving.client import BertClient
import numpy as np
import torch
import json
import sys
import argparse
from tqdm.auto import tqdm

class Embed(object):
    def __init__(self, embedding_file):
        self.embedding_file = embedding_file
        self.items = torch.load(embedding_file)
        self.item_idx = list(self.items.keys())
    
    def __len__(self):
        return len(self.item_idx )

    def idx_to_vec(self):
        all_vecs = []
        for item_idx in self.item_idx:
            item = self.items[item_idx]
            all_vecs.append(item['vec'])
            
        return np.array(all_vecs).squeeze(1)
    
    def get_vecs_by_ans(self, ans):
        for item in self.items:
            if self.items[item]['ans'] == ans:
                return self.items[item]['vec']
        print('ans not in the embedding file')
        return None

    def idx_to_ans(self, idx):
        return self.items[idx]['ans']
    

    def get_vecs_by_idx(self, idx):
        item = self.items[idx]
        return item['vec']


    def cos_similarity(self, W, x):


        cos = np.dot(W, x.reshape(-1,)) / (
            np.sqrt((W*W).sum(1) + 1e-9) * np.sqrt((x * x).sum(1)))
        
        return cos
        

    def get_topk_similarities(self, query_ans, k):
        cos_word = {}
        cos = self.cos_similarity(self.idx_to_vec(),
                        self.get_vecs_by_ans(query_ans))
        topk = np.argsort(cos)[::-1][:k+1] 
        for topk_ in topk[:]:  # Remove input words
            # print('cosine sim=%.3f: %s' % (cos[topk_], (self.idx_to_ans(int(self.item_idx[topk_])))))
            # cos_word.append([cos[topk_],self.idx_to_ans(int(self.item_idx[topk_]))])
            score =' %.3f' % (cos[topk_])
            cos_word[self.idx_to_ans(int(self.item_idx[topk_]))] = score
        return cos_word
    

    def get_theta_similarities(self, query_ans, threshold = 0.5):
        cos_word = []
        cos = self.cos_similarity(self.idx_to_vec(),
                        self.get_vecs_by_ans(query_ans))
        cos_index = np.argsort(cos)[::-1] 
        for cos_ in cos_index:  # Remove input words
            if cos[cos_] > threshold:
                print('cosine sim=%.3f: %s' % (cos[cos_], (self.idx_to_ans(int(self.item_idx[cos_])))))
        return cos_word


    def get_similarity(self, query_ans1, query_ans2):
        cos = self.cos_similarity(self.get_vecs_by_ans(query_ans1),
                        self.get_vecs_by_ans(query_ans2))

        print('cosine sim=%.3f' % (cos))
        return cos 


def get_bert_embedding(ans):
    client = BertClient()
    vector = client.encode([ans])
    return vector
    

def store_vecs_to_file(args):
    ans_file = args.label_file
    all_ans_vec = {}
    with open(ans_file, 'r') as f:
        answers = json.load(f)
        for ans_idx, ans in tqdm(enumerate(answers), total=len(answers)):
            all_ans_vec[ans_idx] = {
                'ans':ans,
                'vec': get_bert_embedding(ans)
            }
        torch.save(all_ans_vec, args.save_path)

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='Get the BERT embeddings of each label')
    parser.add_argument('--label_file',  type=str, default='data/gqa/trainval_label2ans.json',
                        help='the path of label file of dataset')
    parser.add_argument('--save_path', 
                        type = str, 
                        default = "bert_embedding/gqa/trainval_ans2bertemb.pth",
                        help='the path to save the embeddings of each label')
    args = parser.parse_args()
    store_vecs_to_file(args)