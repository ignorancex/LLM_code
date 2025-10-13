import math
import random
import re

import contractions
import en_core_web_sm
import nltk
import numpy as np
import requests
from nltk.corpus import wordnet
from num2words import num2words
# from sentence_transformers import SentenceTransformer, util
from word2number import w2n

import utils
from T5_summarization import T5_summarizer

_SPACY_PIPELINE = en_core_web_sm.load()
# _SBERT = SentenceTransformer('all-MiniLM-L6-v2')
_WIKIDATA_ENTITIES = {
    'FAC': 'Q41176',
    'GPE': 'Q6256',
    'LANGUAGE': 'Q315',
    'LOC': 'Q515',
    'NORP': 'Q231002',
    'ORG': 'Q4830453',
    'PERSON': 'Q5',
}

# DONT CHANGE SEED!
random.seed(10)


def extract_full_comparison_strings(example):
    example_strings = []
    for question in example["questions"]:
        for answer in question["answers"]:
            example_strings.append(question["question"] + " " + answer["answer"])
            if "answer_type" in answer and answer["answer_type"] == "Boolean":
                example_strings[-1] += ". " + answer["boolean_explanation"]

        if len(question["answers"]) == 0:
            example_strings.append(question["question"] + " No answer could be found.")
    return example_strings


# create 'relevance' tests
def robustness_noise_test(dataset: list, rand_dataset: dict = None) -> list:
    """
    Add noise by inserting a random evidence piece from another dataset entry
    @param dataset:
    @param rand_dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        test_sample = entry.copy()
        # random choice
        if not rand_dataset:
            rand_entry = utils.get_random_entry(dataset, entry)
        else:
            rand_entry = random.choice(rand_dataset)
        # add a random QA pair to 'entry'
        rand_question = random.choice(rand_entry["questions"])
        test_sample['questions'].insert(int(len(entry['questions']) / 2), rand_question)
        test_samples_list.append(test_sample)

    return test_samples_list


def coherence_test(dataset: list):
    """
    Test for coherence by shuffling order or evidence sentences
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        test_sample = entry.copy()
        random.shuffle(test_sample['questions'])
        test_samples_list.append(test_sample)

    return test_samples_list


# TODO adjust to find similar sentences using ROUGE/METEOR instead
# def find_most_similar_sentence(claim, evidence):
#     claim_emb = _SBERT.encode([claim], convert_to_tensor=True)
#     evidence_tok = sent_tokenize(" ".join(evidence))
#     evid_emb = _SBERT.encode(evidence_tok, convert_to_tensor=True)
#     cosine_scores = util.cos_sim(claim_emb, evid_emb)
#
#     pairs = []
#     for j in range(0, len(evid_emb)):
#         pairs.append({'evidence': evidence_tok[j], 'score': cosine_scores[0][j]})
#
#     # sort scores in decreasing order
#     pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
#
#     # return sentence with the highest cosine similarity to claim text
#     return [{"question": pairs[0]["evidence"],
#              "answers": [
#                  {
#                      "answer": " ",
#                      "answer_type": "Abstractive",
#                  }
#              ]
#              }]
#     # return None


def coverage_drop_evidence_part_test(dataset: list, type="one_sentence"):
    """
    Test coverage by dropping parts of the evidence
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        test_sample = entry.copy()
        random.shuffle(test_sample['questions'])
        if type == "one_sentence":
            test_sample['questions'].pop()
        elif type == "half":
            for _ in range(int(len(test_sample['questions']) / 2)):
                test_sample['questions'].pop()
        # elif type == "half_similar":
        #     test_sample['questions'] = [find_most_similar_sentence(test_sample["claim"], test_sample['questions'])]
        elif type == "all_but_one":
            test_sample['questions'] = find_most_similar_sentence(test_sample["claim"],
                                                                  extract_full_comparison_strings(test_sample))

        test_samples_list.append(test_sample)

    return test_samples_list


def coverage_drop_answers_test(dataset: list):
    """
    Test coverage by dropping parts of the evidence (for Averitec: answers and keep only questions)
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        test_sample = entry.copy()
        for question in test_sample["questions"]:
            for answer in question["answers"]:
                answer["answer"] = " "
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = " "
        test_samples_list.append(test_sample)

    return test_samples_list


def replace_by_synonyms(text, by_antonymns=False):
    text_replaced = []
    is_noun = lambda pos: 'NN' in pos
    entry_tok = nltk.word_tokenize(text)
    # nouns = [word for (word, pos) in nltk.pos_tag(test_sample_tok) if is_noun(pos)]

    # In test_sample['questions'] replace entities => in questions and answers
    for (word, pos) in nltk.pos_tag(entry_tok):
        if is_noun(pos):
            # print(f"noun: {word}")
            replacement = None
            for syn in wordnet.synsets(word):
                l = syn.lemmas()[0]
                if not by_antonymns and " ".join(l.name().split("_")).lower() != word.lower():
                    replacement = " ".join(l.name().split("_"))
                    # print(f"replaced by: {replacement}\n")
                    break
                elif by_antonymns:
                    if l.antonyms() and " ".join(l.antonyms()[0].name().split("_")).lower() != word.lower():
                        replacement = " ".join(l.antonyms()[0].name().split("_"))
                        break
            if not replacement:
                replacement = word
            # print(f"initial word: {word}, replacement: {replacement}")
            text_replaced.append(replacement)
        else:
            text_replaced.append(word)

    return " ".join(text_replaced)


def invariance_synonym_test(dataset: list):
    """
    Create relevance tests by replacing parts of text by synonyms
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        # questions_replaced = []
        for evidence in entry_copy['questions']:
            evidence["question"] = replace_by_synonyms(evidence["question"])
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = replace_by_synonyms(answer["boolean_explanation"])
                else:
                    answer["answer"] = replace_by_synonyms(answer["answer"])

        test_samples_list.append(entry_copy)

    return test_samples_list


def rand_shuffle_words(text):
    list_tok = text.split(" ")
    random.shuffle(list_tok)
    list_tok = " ".join(list_tok)

    return list_tok


def informativeness_random_word_order_test(dataset: list):
    """
    Create tests by randomly changing order of words in sentences
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        for evidence in entry_copy['questions']:
            evidence["question"] = rand_shuffle_words(evidence["question"])
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = rand_shuffle_words(answer["boolean_explanation"])
                else:
                    answer["answer"] = rand_shuffle_words(answer["answer"])

        test_samples_list.append(entry_copy)

    return test_samples_list


def create_sum_test(dataset: list):
    """
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        evidence = " ".join(extract_full_comparison_strings(entry_copy))
        try:
            evidence = T5_summarizer.summarize(evidence)
        except Exception as e:
            evidence = " "

        evidence_piece = {
            "question": evidence,
            "answers": [{
                "answer": " ",
                "answer_type": "Extractive",
            }]}
        entry_copy["questions"] = [evidence_piece]
        test_samples_list.append(entry_copy)

    return test_samples_list


def change_contraction(text):
    expanded_words = []
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_words)
    return expanded_text


def invariance_contraction_test(dataset: list):
    """
    Create tests by introducing contaction e.g. "we're" where applicable (e.g. instead of "we are")
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        for evidence in entry_copy['questions']:
            evidence["question"] = change_contraction(evidence["question"])
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = change_contraction(answer["boolean_explanation"])
                else:
                    answer["answer"] = change_contraction(answer["answer"])

        test_samples_list.append(entry_copy)

    return test_samples_list


def num_to_word(text):
    adj_text = None
    tok_list = re.findall(r"\d*\.?\d+", text)
    for tok in tok_list:
        tok_word = num2words(float(tok))
        if adj_text:
            adj_text = re.sub(r"\b%s\b" % tok, tok_word, adj_text)
        else:
            adj_text = re.sub(r"\b%s\b" % tok, tok_word, text)

    return adj_text if adj_text else text


def invariance_num2text_test(dataset: list):
    """
    Create tests by replacing numbers as numerals by numbers as text
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        for evidence in entry_copy['questions']:
            evidence["question"] = num_to_word(evidence["question"])
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = num_to_word(answer["boolean_explanation"])
                else:
                    answer["answer"] = num_to_word(answer["answer"])

        test_samples_list.append(entry_copy)

    return test_samples_list


def word_to_num(text):
    adj_text = None
    for tok in text.split():
        try:
            if w2n.word_to_num(tok) and not tok.isdigit():
                tok_num = w2n.word_to_num(tok)
                if adj_text:
                    adj_text = re.sub(r"\b%s\b" % tok, str(tok_num), adj_text)
                else:
                    adj_text = re.sub(r"\b%s\b" % tok, str(tok_num), text)
        except Exception as e:
            pass
    return adj_text if adj_text else text


def invariance_text2num_test(dataset: list):
    """
    Create tests by replacing numbers as text by numbers as numerals
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        for evidence in entry_copy['questions']:
            evidence["question"] = word_to_num(evidence["question"])
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = word_to_num(answer["boolean_explanation"])
                else:
                    answer["answer"] = word_to_num(answer["answer"])

        test_samples_list.append(entry_copy)

    return test_samples_list


def fluency_typos_test(dataset: list, ratio_typos=0.1):
    """
    Create tests by introducing typos in 10% (default) of words
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        for evidence in entry_copy['questions']:
            evidence["question"] = add_typos(evidence["question"], ratio_typos=ratio_typos)
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = add_typos(answer["boolean_explanation"], ratio_typos=ratio_typos)
                else:
                    answer["answer"] = add_typos(answer["answer"], ratio_typos=ratio_typos)

        test_samples_list.append(entry_copy)

    return test_samples_list


# reference: https://github.com/marcotcr/checklist/blob/3edd07c9a84e6c6657333450d4d0e70ecb0c00d9/checklist/perturb.py#L148
def add_typos(text, ratio_typos=0.1, typos=1):
    """Perturbation functions, swaps random characters with their neighbors
    Parameters
    ----------
    tok : str
        input string
    typos : int
        number of typos to add
    Returns
    -------
    list(string)
        perturbed strings
    """
    adj_text = text.split()
    for idx in random.choices(range(len(adj_text)), k=math.ceil(len(adj_text) * ratio_typos)):
        tok = list(adj_text[idx])
        if len(tok) == 1:
            continue
        try:
            swaps = np.random.choice(len(tok) - 1, typos)
            for swap in swaps:
                tmp = tok[swap]
                tok[swap] = tok[swap + 1]
                tok[swap + 1] = tmp
        except Exception as e:
            print(e)
        adj_text[idx] = "".join(tok)

    return " ".join(adj_text)


def get_entity(text):
    doc_text = _SPACY_PIPELINE(text)
    filtered_entities = [(text_proc.text, text_proc.label_) for text_proc in doc_text.ents if
                         text_proc.label_ in _WIKIDATA_ENTITIES.keys()]

    return filtered_entities


def get_wikidata_entity(entity_label):
    """
    Get a wikidata entity belonging to the same class as 'entity_label'
    @param entity_label:
    @return:
    """
    wikidata_entity = _WIKIDATA_ENTITIES[entity_label]

    wd_page_url = f"https://www.wikidata.org/w/api.php?action=query&format=json&list=search&srsearch=haswbstatement:P31={wikidata_entity}&srlimit=max"
    wd_response = requests.get(wd_page_url, stream=True)
    instance = wd_response.json()["query"]["search"]
    entity_id_list = [entry["title"] for entry in instance]
    random.shuffle(entity_id_list)

    instances_name = None
    for entity_id in entity_id_list:
        try:
            wd_instance_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&format=json"
            wd_response = requests.get(wd_instance_url, stream=True)
            instances_name = wd_response.json()["entities"][entity_id]["labels"]["en"]["value"]
            break
        except Exception as e:
            continue

    return instances_name


def find_entity_replacement(text, claim_entities):
    entities = get_entity(text)
    claim_entities_text = [e_text for e_text, e_label in claim_entities]

    for entity, label in entities:
        if entity in claim_entities_text:
            replacement_ent = get_wikidata_entity(label)
            if replacement_ent:
                text = text.replace(entity, replacement_ent)

    return text


def informativeness_entity_swap_test(dataset: list):
    """
    Create tests by changing entities in evidence is same entity in claim
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        # get entities in claim
        claim_entities = get_entity(entry_copy["claim"])
        if not claim_entities or len(claim_entities) == 0:
            test_samples_list.append(entry_copy)
            continue

        for evidence in entry_copy['questions']:
            evidence["question"] = find_entity_replacement(evidence["question"], claim_entities)
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = find_entity_replacement(answer["boolean_explanation"],
                                                                            claim_entities)
                else:
                    answer["answer"] = find_entity_replacement(answer["answer"], claim_entities)

        test_samples_list.append(entry_copy)

    return test_samples_list


def redundancy_duplicate_sentence_test(dataset: list):
    """
    Test performance on redundancy (duplicate evidence pieces)
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        test_sample = entry.copy()
        duplicate = random.choice(test_sample['questions'])
        test_sample['questions'].append(duplicate)
        test_samples_list.append(test_sample)

    return test_samples_list


def duplicate_words(text, ratio):
    adj_text = text.split()
    for idx in random.choices(range(len(adj_text)), k=math.ceil(len(adj_text) * ratio)):
        tok = adj_text[idx]
        tok = tok + " " + tok
        adj_text[idx] = tok

    return " ".join(adj_text)


def redundancy_duplicate_words_test(dataset: list, duplicate_ratio=0.2):
    """
    Test performance on redundancy (duplicate words in evidence sentences)
    @param duplicate_ratio: ratio of words to duplicate
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        for evidence in entry_copy['questions']:
            evidence["question"] = duplicate_words(evidence["question"], duplicate_ratio)
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = duplicate_words(answer["boolean_explanation"], duplicate_ratio)
                else:
                    answer["answer"] = duplicate_words(answer["answer"], duplicate_ratio)

        test_samples_list.append(entry_copy)

    return test_samples_list


def drop_words(text, drop_words_list):
    text_list = [t for t in text.split() if t not in drop_words_list]
    return " ".join(text_list)


def fluency_drop_words_test(dataset: list):
    """
    Test performance on fluency by dropping certain word types (articles, etc.)
    @param duplicate_ratio: ratio of words to duplicate
    @param dataset:
    @return: list
    """
    test_samples_list = []
    stopwords_list = set(nltk.corpus.stopwords.words('english'))

    for entry in dataset:
        entry_copy = entry.copy()
        for evidence in entry_copy['questions']:
            evidence["question"] = drop_words(evidence["question"], stopwords_list)
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = drop_words(answer["boolean_explanation"], stopwords_list)
                else:
                    answer["answer"] = drop_words(answer["answer"], stopwords_list)

        test_samples_list.append(entry_copy)

    return test_samples_list


def number_replacement(text, num_list):
    tok_list = re.findall(r"\d*\.?\d+", text)
    for tok in tok_list:
        if tok in num_list:
            tok_digit = float(tok)
            tok_digit_replace = random.uniform(tok_digit - tok_digit * 0.1, tok_digit + tok_digit * 0.1)
            text = text.replace(tok, str(tok_digit_replace))
    return text


def informativeness_number_change_test(dataset: list):
    """
    Create tests by changing numbers in evidence to a different number if this same number occurs in the claim
    @param dataset:
    @return: list
    """
    test_samples_list = []

    for entry in dataset:
        entry_copy = entry.copy()
        # get numbers in claim
        claim_num = re.findall(r"\d*\.?\d+", entry_copy["claim"])
        if not claim_num or len(claim_num) == 0:
            test_samples_list.append(entry_copy)
            continue

        for evidence in entry_copy['questions']:
            evidence["question"] = number_replacement(evidence["question"], claim_num)
            for i, answer in enumerate(evidence["answers"]):
                if answer["answer_type"] == "Boolean":
                    answer["boolean_explanation"] = number_replacement(answer["boolean_explanation"], claim_num)
                else:
                    answer["answer"] = number_replacement(answer["answer"], claim_num)

        test_samples_list.append(entry_copy)

    return test_samples_list
