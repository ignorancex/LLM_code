import numpy as np
import tensorflow as tf

from data_structures import UNK_label, UNK_word


class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, size_word_vocab, size_TE_label_vocab, size_word_embed, size_TE_label_embed, TE_label_set,
                 **kwargs):
        super(EmbeddingLayer, self).__init__(input_shape=(None, 2), **kwargs)

        self.size_word_vocab = size_word_vocab
        self.size_TE_label_vocab = size_TE_label_vocab
        self.size_word_embed = size_word_embed
        self.size_TE_label_embed = size_TE_label_embed
        self.TE_label_set = TE_label_set

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.word_embeddings = tf.keras.layers.Embedding(self.size_word_vocab, self.size_word_embed,
                                                         input_shape=(None,))
        self.TE_label_embeddings = tf.keras.layers.Embedding(self.size_TE_label_vocab, self.size_TE_label_embed,
                                                             input_shape=(None,))

        super(EmbeddingLayer, self).build(input_shape)

    def get_config(self):
        config = super(EmbeddingLayer, self).get_config()
        config.update({
            'size_word_vocab': self.size_word_vocab,
            'size_TE_label_vocab': self.size_TE_label_vocab,
            'size_word_embed': self.size_word_embed,
            'size_TE_label_embed': self.size_TE_label_embed,
            'TE_label_set': self.TE_label_set,
        })
        return config

    def call(self, inputs, **kwargs):
        word_ids = inputs[:, :, 0]
        TE_label_ids = inputs[:, :, 1]

        if self.TE_label_set == 'none':
            # Return just word embeddings with shape (len(word_ids), size_word_embed)
            return self.word_embeddings(word_ids)

        else:
            # Get word and label embeddings, and concatenate them into a new "embedding"
            embedded_words = self.word_embeddings(word_ids)
            embedded_labels = self.TE_label_embeddings(TE_label_ids)

            # For each word/row, concatenate word and label embeddings
            return tf.concat([embedded_words, embedded_labels], axis=2)

    def compute_output_shape(self, input_shape):
        if self.TE_label_set == 'none':
            # [:2] matches slicing [:, :, 0] above
            return self.word_embeddings.compute_output_shape(input_shape[:2])
        else:
            # [:2] matches slicing [:, :, 0] above, to get (batch_size, n) as input to embeddings
            # Then self.*_embeddings returns shape (batch_size, n, emb_length) for n words so [2] gets emb_length
            word_emb_length = self.word_embeddings.compute_output_shape(input_shape[:2])[2]
            label_emb_length = self.TE_label_embeddings.compute_output_shape(input_shape[:2])[2]
            embedding_length = word_emb_length + label_emb_length
            return tf.TensorShape((input_shape[0], input_shape[1], embedding_length))


def get_words_and_labels_for_doc(sentence_list, child_parent_candidates, word_vocab, TE_label_vocab, TE_label_set):
    # Build word list
    words = []
    TE_labels = []
    for sentence in sentence_list:
        for word in sentence:
            words.append(word)
            TE_labels.append(UNK_label)

    # Build label list
    if TE_label_set != 'none':
        for candidates_for_child in child_parent_candidates:
            # All these candidates share the same child, so we can just get the child from the first one
            first_candidate = candidates_for_child[0]
            parent, child, edge_label = first_candidate

            if child.start_word_index_in_doc >= 0:  # Ignore padding and meta nodes (they're not in sentence_list)
                for k in range(child.start_word_index_in_doc, child.end_word_index_in_doc + 1):
                    if TE_label_set == 'timex_event':
                        TE_labels[k] = child.TE_label
                    else:
                        TE_labels[k] = child.full_label

    word_ids = [word_vocab.get(word, word_vocab[UNK_word]) for word in words]

    TE_label_ids = []
    for TE_label in TE_labels:
        if TE_label in TE_label_vocab:
            TE_label_ids.append(TE_label_vocab.get(TE_label))
        else:
            print('WARNING: Label {} not in TE_label_vocab, using {}'.format(TE_label, UNK_label))
            TE_label_ids.append(TE_label_vocab.get(UNK_label))

    return np.stack([word_ids, TE_label_ids], axis=1)  # Make it one array to make it easy to pass around
