######################################################################
##  
## This script is a heavily modifed version fo taht provided in winogender-schemas
## https://github.com/rudinger/winogender-schemas
##
######################################################################

import csv
import os
from pathlib import Path
from collections import OrderedDict

from config import GENDERED_IDS

# This script fully instantiates the 120 templates in ../data/templates.tsv
# to generate the 720 sentences in ../data/all_sentences.tsv
# By default this script prints to stdout, and can be run with no arguments:

def load_templates(path):
    fp = open(path, 'r')
    next(fp)  # first line headers
    S = []
    for l in fp:

        line = l.strip().split('\t')
        occupation, answer, sentence = line[0], line[1], line[2]
        S.append((occupation, answer, sentence))
    return S

def generate(occupation, sentence, g_id=None, context=None):
    toks = sentence.split(" ")
    occ_index = toks.index("$OCCUPATION")

    if g_id:
        toks[occ_index] = f"The {g_id} {occupation}"

    else:
        toks[occ_index] = f"The {occupation}"
        
    NOM = "$NOM_PRONOUN"
    POSS = "$POSS_PRONOUN"
    ACC = "$ACC_PRONOUN"
    special_toks = set({NOM, POSS, ACC})
    mask_map = {NOM: "MASK", POSS: "MASK", ACC: "MASK"}
    mask_toks = [x if not x in special_toks else mask_map[x] for x in toks]
    masked_sent = " ".join(mask_toks)
        
    return masked_sent 
# %%


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    rel_path = "winogender_schema_evaluation_set"
    abs_path = os.path.join(script_dir, rel_path)
    Path(abs_path).mkdir(parents=True, exist_ok=True)
    # %%

    S = load_templates(os.path.join(abs_path, "templates_g_id.tsv"))

    # %%
    with open(os.path.join(abs_path, "all_sentences_g_id.tsv"), 'w', newline='') as csvfile:
        sentence_writer = csv.writer(csvfile, delimiter='\t')
        sentence_writer.writerow(['sentid', 'sentence'])
        sentence_dict = OrderedDict()

        for s in S:
            for g_id in GENDERED_IDS:
                occupation, answer, sentence = s

                gendered_professional_sentence = generate(
                    occupation, sentence, g_id)
                gendered_sentid = f"{occupation}_{g_id}_{answer}"
                sentence_dict[gendered_sentid] = gendered_professional_sentence
                sentence_writer.writerow([gendered_sentid, gendered_professional_sentence])
                
            non_gendered_professional_sentence = generate(
                occupation, sentence)
            non_gendered_sentid = f"{occupation}_unspecified_{answer}"
            sentence_dict[non_gendered_sentid] = non_gendered_professional_sentence
            sentence_writer.writerow([non_gendered_sentid, non_gendered_professional_sentence])
