import tensorflow as tf

from data_structures import UNK_label, PAD_label


def TEEmbeddingLayer(size_TE_label_vocab, size_TE_label_embed, TE_label_set):
    if TE_label_set == 'none':
        raise NotImplementedError('Can\'t embed TE labels if there are no TE labels')

    return tf.keras.layers.Embedding(size_TE_label_vocab, size_TE_label_embed, input_shape=(None,))


def get_TE_labels_for_doc(sentence_count, sorted_nodes, word_in_doc_to_tokens_map, TE_label_vocab,
                          TE_label_set, max_sequence_length):
    # Start by assuming all tokens are padding
    token_count = sentence_count * max_sequence_length
    TE_labels = [PAD_label] * token_count

    # For all actual words, use UNK_label
    for word_start_index, word_end_index in word_in_doc_to_tokens_map:
        for k in range(word_start_index, word_end_index + 1):
            TE_labels[k] = UNK_label

    # For all words which are nodes, use the TE_label of the node
    if TE_label_set != 'none':
        for node in sorted_nodes:
            if node.start_word_index_in_doc >= 0:
                # Node has real words (not a meta node or a padding node)
                start_index = word_in_doc_to_tokens_map[node.start_word_index_in_doc][0]
                end_index = word_in_doc_to_tokens_map[node.end_word_index_in_doc][1]

                for k in range(start_index, end_index + 1):
                    if TE_label_set == 'timex_event':
                        TE_labels[k] = node.TE_label
                    else:
                        TE_labels[k] = node.full_label

    TE_label_ids = []
    for TE_label in TE_labels:
        if TE_label in TE_label_vocab:
            TE_label_ids.append(TE_label_vocab.get(TE_label))
        else:
            print('WARNING: Label {} not in TE_label_vocab, using {}'.format(TE_label, UNK_label))
            TE_label_ids.append(TE_label_vocab.get(UNK_label))

    return TE_label_ids
