import json
from typing import List
from tqdm import tqdm
import argparse

from allennlp.predictors.predictor import Predictor
import allennlp_models.nli
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli-roberta-large-2020.04.30.tar.gz", predictor_name="textual-entailment")
print("predictor registered")
def entailment_score(hypothesis:List, premises:List)->float:
    semantic_score = 0
    assert len(hypothesis) == len(premises), print("the length of hypothesis should be the same")
    for hyp, pre in zip(hypothesis, premises):
        score = predictor.predict(hypothesis=hyp, premise=pre)['probs'][0]
        semantic_score += score 
        
    return semantic_score/len(hypothesis)

def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the entailment score of aas of each label ')
    parser.add_argument('--aas_file',  type=str, default='aas_generation/bert_embedding/gqa/bert_aas.json',
                        help='the path of aas file of each label')
    parser.add_argument('--save_file',  type=str, default='aas_generation/bert_embedding/gqa/bert_aas_score.json',
                        help='the path to save the aas of each label')
    parser.add_argument('--dataset_type',  type=str, default='gqa',
                        help='the type of the dataset, either gqa or vqa.')
    
    args = parser.parse_args()
    with open(args.aas_file) as f:
        label2aas = json.load(f)
        f.close()
    
    if args.dataset_type == "gqa":
        template_file = "grounding/GQA_grounded_questions50.json"
    else:
        template_file = "grounding/VQA_grounded_questions50.json"
    
    label2templates = read_json(template_file)
    
    threshold = 0.5 # set the threshold of nli_score be 0.5
    label2aas_scores = {}
    for label, all_aas in tqdm(label2aas.items()):
        premises = label2templates[label]['template']
        label2aas_scores[label] = {}
        for aas in all_aas:
            hypothesis = [pre.replace(label, aas) for pre in premises]
            nli_score=entailment_score(hypothesis, premises) 
            if nli_score>threshold:
                label2aas_scores[label][aas] = nli_score
        if len(label2aas_scores) == 2:
            break
    json.dump(label2aas_scores, open(args.save_file, 'w'), indent=2)

        
