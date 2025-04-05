import numpy as np

from bert_as_classifier.shared.model import get_friendly_node_label, get_gold_candidate_one_hot
from data_structures import is_root_node, is_padding_node, ROOT_word, PAD_word
from shared.bert_layer import convert_examples_to_inputs, InputExample
from shared.numeric_features import get_numeric_features


def get_model_inputs(tokenizer, sentence_list, child_parent_candidates, TE_label_set, TE_label_vocab,
                     max_sequence_length, disable_handcrafted_features):
    inputs = []
    node_count = len(child_parent_candidates) + 1  # Number of children plus root node

    # Batch inputs by child
    for candidates_for_child in child_parent_candidates:
        examples = candidates_for_child_to_examples(candidates_for_child, TE_label_set, sentence_list)
        bert_inputs, words_to_tokens_maps = convert_examples_to_inputs(tokenizer, examples, max_sequence_length)

        child_inputs = {
            'input_ids': bert_inputs['input_ids'],
            'input_masks': bert_inputs['input_masks'],
            'segment_ids': bert_inputs['segment_ids']
        }
        if not disable_handcrafted_features:
            numeric_features = np.array(
                [get_numeric_features(parent, child, node_count, len(sentence_list), TE_label_vocab)
                 for parent, child, gold_label in candidates_for_child])
            child_inputs['numeric_features'] = numeric_features
        inputs.append(child_inputs)

    return inputs


def candidates_for_child_to_examples(candidates_for_child, TE_label_set, sentence_list):
    child_examples = []
    for parent, child, label in candidates_for_child:
        # Make fake sentences for BERT
        if is_root_node(parent):
            parent_sentence = [ROOT_word]
        elif is_padding_node(parent):
            parent_sentence = [PAD_word]
        else:
            parent_sentence = sentence_list[parent.snt_index_in_doc]
        child_sentence = sentence_list[child.snt_index_in_doc]

        parent_label = get_friendly_node_label(parent, TE_label_set)
        child_label = get_friendly_node_label(child, TE_label_set)

        first_bert_sentence = parent.words.split() + [parent_label, ':'] + parent_sentence
        second_bert_sentence = child.words.split() + [child_label, ':'] + child_sentence
        child_examples.append(InputExample(first_bert_sentence, second_bert_sentence))
    return child_examples


def data_to_inputs_and_gold(data, bert_tokenizer, labeled, TE_label_vocab, TE_label_set, edge_label_set,
                            max_sequence_length, disable_handcrafted_features):
    # TODO I think I can actually create a np.array/list of ALL candidates, then batch by max_candidate_count, and
    #  avoid fit_generator entirely (but, need to create dict of this since model takes dict)
    #  N.B. need to be careful of the shuffle option on fit_generator
    inputs_and_gold = []
    for document in data:
        sentence_list, child_parent_candidates = document
        model_inputs = get_model_inputs(bert_tokenizer, sentence_list, child_parent_candidates, TE_label_set,
                                        TE_label_vocab, max_sequence_length, disable_handcrafted_features)
        gold_one_hots = [get_gold_candidate_one_hot(candidates_for_child, labeled, edge_label_set)
                         for candidates_for_child in child_parent_candidates]
        doc_inputs_and_gold = [(model_input, gold_one_hot)  # One batch per candidate
                               for model_input, gold_one_hot in zip(model_inputs, gold_one_hots)]
        inputs_and_gold.extend(doc_inputs_and_gold)

    return inputs_and_gold
