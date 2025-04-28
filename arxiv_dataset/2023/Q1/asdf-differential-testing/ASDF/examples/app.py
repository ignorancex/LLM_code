import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, sys
sys.path.append("../")
from crossasr import CrossASR
import utils 

import argparse

DEFAULT_CONFIG = {
    "seed": 2021,
    "tts" : utils.getTTS("google"),
    "recompute": True,
    "num_iteration": 2,
}
DEFAULT_CORPUS_FPATH = "corpus/50-europarl-20000.txt"
DEFAULT_ASRS = ["deepspeech", "wav2letter", "wav2vec2"]
DEFAULT_OUTPUT_DIR = "output/default"
DEFAULT_NUM_TEXTS = 100
MUTATORS = {
    "homophone":"Swaps the erroneous word(s) with a similar sounding alternative word(s)",
    "augmenter":"Inserts a new word adjacent to the erroneous word(s)",
    "plurality":"Swaps the erroneous word(s) with its plural/singular form",
    "tense":"Changes the errroneous sentence to a different tense",
    "deletion":"Deletes a preceding or suceeding word adjacent to the erroneous word(s)"}

if __name__ == "__main__":
    # parse input args from batch file
    corpus_fpath = input("Enter the path to the input corpus: ")
    output_fpath = input("Enter the directory for the results output: ")
    num_texts = input("Enter the number of corpus texts to test: ")
    for asr in DEFAULT_ASRS:
        asr_selected = input("ASR Selection: Test "+asr+"?(Y/n) ")
        if asr_selected.lower() == "y":
            continue
        elif asr_selected.lower() =="n":
            DEFAULT_ASRS.remove(asr)
    print("Mutators Available:")
    for m, d in MUTATORS.items():
        print(m,"-",d)
    mutator_selected = input("Select a transformation method from above to apply onto failed test cases (e.g. plurality): ")

    corpus_fpath = corpus_fpath if corpus_fpath != "" else DEFAULT_CORPUS_FPATH
    asrs = utils.getASRS(DEFAULT_ASRS)
    mutator = utils.getMutator(mutator_selected) if mutator_selected in MUTATORS.keys() else None
    output_dir = "output/" + output_fpath if output_fpath != "" else DEFAULT_OUTPUT_DIR
    num_texts = int(num_texts) if num_texts.isdigit else DEFAULT_NUM_TEXTS

    crossasr = CrossASR(asrs=asrs, mutator=mutator, output_dir=output_dir, **DEFAULT_CONFIG)

    # corpus_fpath = os.path.join(config["output_dir"], config["corpus_fpath"])
    texts = utils.readCorpus(corpus_fpath=corpus_fpath)
    crossasr.processCorpus(texts=texts[:num_texts])
    crossasr.printStatistic()

