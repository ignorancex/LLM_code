######################################################################
##  
## This script is a lightly modifed version fo taht provided in winogender-schemas
## https://github.com/rudinger/winogender-schemas
##
######################################################################

import csv
import os
from pathlib import Path
from collections import OrderedDict

# This script fully instantiates the 120 templates in ../data/templates.tsv
# to generate the 720 sentences in ../data/all_sentences.tsv
# By default this script prints to stdout, and can be run with no arguments:

def load_templates(path):
    fp = open(path, 'r')
    next(fp)  # first line headers
    S = []
    for line in fp:

        line = line.strip().split('\t')
        occupation, other_participant, answer, sentence = line[0], line[1], line[2], line[3]
        S.append((occupation, other_participant, answer, sentence))
    return S

def generate(occupation, other_participant, sentence, second_ref="", context=None):
    toks = sentence.split(" ")
    occ_index = toks.index("$OCCUPATION")
    part_index = toks.index("$PARTICIPANT")
    toks[occ_index] = occupation
    # we are using the instantiated participant, e.g. "client", "patient", "customer",...
    if not second_ref:
        toks[part_index] = other_participant
    elif second_ref != 'someone':
        toks[part_index] = second_ref
    else:
        # we are using the bleached NP "someone" for the other participant
        # first, remove the token that precedes $PARTICIPANT, i.e. "the"
        toks = toks[:part_index-1]+toks[part_index:]
        # recompute participant index (it should be part_index - 1)
        part_index = toks.index("$PARTICIPANT")
        if part_index == 0:
            toks[part_index] = "Someone"
        else:
            toks[part_index] = "someone"
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

    S = load_templates(os.path.join(abs_path, "templates.tsv"))

    # %%
    with open(os.path.join(abs_path, "all_sentences.tsv"), 'w', newline='') as csvfile:
        sentence_writer = csv.writer(csvfile, delimiter='\t')
        sentence_writer.writerow(['sentid', 'sentence'])
        sentence_dict = OrderedDict()

        for s in S:
            occupation, other_participant, answer, sentence = s

            gendered_sentence = generate(
                occupation, other_participant, sentence)
            gendered_sentid = f"{occupation}_{other_participant}_{answer}"
            sentence_dict[gendered_sentid] = gendered_sentence

            someone_sentence = generate(
                occupation, other_participant, sentence, second_ref='someone')
            someone_sentid = f"{occupation}_someone_{answer}"
            sentence_dict[someone_sentid] = someone_sentence

            man_sentence = generate(
                occupation, other_participant, sentence, second_ref='man')
            man_sentid = f"{occupation}_man_{answer}"
            sentence_dict[man_sentid] = man_sentence

            woman_sentence = generate(
                occupation, other_participant, sentence, second_ref='woman')
            woman_sentid = f"{occupation}_woman_{answer}"
            sentence_dict[woman_sentid] = woman_sentence

            sentence_writer.writerow([gendered_sentid, gendered_sentence])
            sentence_writer.writerow([someone_sentid, someone_sentence])
            sentence_writer.writerow([man_sentid, man_sentence])
            sentence_writer.writerow([woman_sentid, woman_sentence])

    # return sentence_dict