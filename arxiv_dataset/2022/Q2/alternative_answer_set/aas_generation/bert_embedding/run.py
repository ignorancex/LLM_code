from embedding import Embed
import json
import argparse
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the BERT embeddings of each label')
    parser.add_argument('--embedding_file',  type=str, default='bert_embedding/gqa/trainval_ans2bertemb.pth',
                        help='the path of label file of dataset')
    parser.add_argument('--save_file',  type=str, default='bert_embedding/gqa/bert_aas.json',
                        help='the path to save the aas of each label')
    
    args = parser.parse_args()
    ans_emb = Embed(args.embedding_file)
    ans2syn_bert={}
    for ans_idx in tqdm(ans_emb.item_idx):
        ans = ans_emb.idx_to_ans(ans_idx)
        syn = ans_emb.get_topk_similarities(ans, k=15)
        ans2syn_bert[ans] = list(syn.keys())
    with open(args.save_file,'w') as f:
        json.dump(ans2syn_bert, f, indent=2)