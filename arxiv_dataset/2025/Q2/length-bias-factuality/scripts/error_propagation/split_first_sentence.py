"""This file provides a sentence splitter which can be used for splitting the first sentence from a response."""
# === Code from the FActScore repository https://github.com/shmsw25/FActScore/tree/main ===

import re
from nltk.tokenize import sent_tokenize
import numpy as np
import argparse

def split_sentences(text):
    sentences = []
    initials = detect_initials(text)
    
    curr_sentences = sent_tokenize(text)
    curr_sentences_2 = sent_tokenize(text)

    curr_sentences = fix_sentence_splitter(curr_sentences, initials)
    curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

    # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
    assert curr_sentences == curr_sentences_2, (text, curr_sentences, curr_sentences_2)

    sentences += curr_sentences
    
    return sentences

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

# === To fix sentence tokenization ===
def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        # if not found in any sentence, then use the following logic to merge the sentences
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

def main(args):
    # === Test the sentence splitter ===
    sentences = split_sentences(args.text)
    first_sentence = sentences[0] if sentences else ""
    print(f"First sentence: {first_sentence}")
    print(f"Completed text: {args.text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split the first sentence from the response")
    parser.add_argument('--text', type=str, \
        default="Hello, this is a test response. It contains multiple sentences. You can replace this with your own text.")    
    args = parser.parse_args()
    main(args)