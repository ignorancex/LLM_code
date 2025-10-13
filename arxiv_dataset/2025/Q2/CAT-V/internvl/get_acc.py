import json
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
import os

# Load the JSON file

results = "./results"
# read all json files in the directory
for file in os.listdir(results):
    if file.endswith(".json"):
        with open(f"{results}/{file}", "r", encoding="utf-8") as file:
            data = json.load(file)

    # Prepare references and hypotheses as dictionaries
    gts = {}  # Ground truth (references)
    res = {}  # Results (hypotheses)a

    for i, item in enumerate(tqdm(data)):
        gts[i] = [item["correct_answer"]]  # Reference list for ID i
        res[i] = [item["model_answer"]]    # Hypothesis for ID i

    # BLEU Score
    def compute_bleu(gts, res):
        bleu_scorer = Bleu(4)  # Compute BLEU-1 to BLEU-4
        score, _ = bleu_scorer.compute_score(gts, res)
        return score

    # METEOR Score
    def compute_meteor(gts, res):
        meteor_scorer = Meteor()
        score, _ = meteor_scorer.compute_score(gts, res)
        return score

    # CIDEr Score
    def compute_cider(gts, res):
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return score

    # Calculate scores
    bleu_score = compute_bleu(gts, res)
    meteor_score = compute_meteor(gts, res)
    cider_score = compute_cider(gts, res)

    # Print results
    print(f"Results for {file}")
    print(f"BLEU Scores: {[round(i*100, 2) for i in bleu_score]}")
    print(f"METEOR Score: {round(meteor_score*100, 2)}")
    print(f"CIDEr Score: {round(cider_score*100, 2)}")
