import numpy as np

from bert_as_classifier.shared.model import get_friendly_node_label, get_gold_candidate_one_hot
from data_structures import is_root_node, is_padding_node, ROOT_word, PAD_word
from shared.bert_layer import convert_examples_to_inputs, InputExample
from shared.numeric_features import get_numeric_features


def get_model_inputs(tokenizer, sentence_list, child_parent_candidates, TE_label_set, TE_label_vocab,
                     max_sequence_length, max_sentence_span, disable_handcrafted_features):
    inputs = []
    node_count = len(child_parent_candidates) + 1  # Number of children plus root node

    # Batch inputs by child
    for candidates_for_child in child_parent_candidates:
        examples = candidates_for_child_to_examples(candidates_for_child, TE_label_set, sentence_list,
                                                    max_sentence_span)
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


def candidates_for_child_to_examples(candidates_for_child, TE_label_set, sentence_list, max_sentence_span):
    child_examples = []
    for parent, child, label in candidates_for_child:
        # Make fake sentences for BERT
        if is_root_node(parent):
            # Can't use sentence_list[0:child.snt_index_in_doc+1] because would be too long for long documents
            sentence_span = [[ROOT_word], sentence_list[child.snt_index_in_doc]]
        elif parent.is_DCT:
            # Can't use sentence_list[0:child.snt_index_in_doc+1] because would be too long for long documents
            sentence_span = [sentence_list[parent.snt_index_in_doc], sentence_list[child.snt_index_in_doc]]
        elif is_padding_node(parent):
            sentence_span = [[PAD_word], sentence_list[child.snt_index_in_doc]]
        else:
            first_snt_index = min(parent.snt_index_in_doc, child.snt_index_in_doc)
            last_snt_index = max(parent.snt_index_in_doc, child.snt_index_in_doc)
            if last_snt_index + 1 - first_snt_index <= max_sentence_span:
                sentence_span = sentence_list[first_snt_index:last_snt_index + 1]
            else:
                # This is still too long to fit into BERT
                sentence_span = [sentence_list[first_snt_index], sentence_list[last_snt_index]]

        parent_label = get_friendly_node_label(parent, TE_label_set)
        child_label = get_friendly_node_label(child, TE_label_set)

        first_bert_sentence = parent.words.split() + [',', parent_label, ':'] + child.words.split() + [',', child_label]
        second_bert_sentence = flatten_sentences(sentence_span)
        child_examples.append(InputExample(first_bert_sentence, second_bert_sentence))
    return child_examples


def flatten_sentences(sentence_span):
    return [word for sentence in sentence_span for word in sentence]


def data_to_inputs_and_gold(data, bert_tokenizer, labeled, TE_label_vocab, TE_label_set, edge_label_set,
                            max_sequence_length, max_sentence_span, disable_handcrafted_features):
    # TODO I think I can actually create a np.array/list of ALL candidates, then batch by max_candidate_count, and
    #  avoid fit_generator entirely (but, need to create dict of this since model takes dict)
    #  N.B. need to be careful of the shuffle option on fit_generator
    inputs_and_gold = []
    for document in data:
        sentence_list, child_parent_candidates = document
        model_inputs = get_model_inputs(bert_tokenizer, sentence_list, child_parent_candidates, TE_label_set,
                                        TE_label_vocab, max_sequence_length, max_sentence_span,
                                        disable_handcrafted_features)
        gold_one_hots = [get_gold_candidate_one_hot(candidates_for_child, labeled, edge_label_set)
                         for candidates_for_child in child_parent_candidates]
        doc_inputs_and_gold = [(model_input, gold_one_hot)  # One batch per candidate
                               for model_input, gold_one_hot in zip(model_inputs, gold_one_hots)]
        inputs_and_gold.extend(doc_inputs_and_gold)

    return inputs_and_gold
