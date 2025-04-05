import numpy as np
import tensorflow as tf

from data_structures import UNK_word

EMBEDDING_DIM = 100
GLOVE_PATH = f'frozen_glove/glove.6B.{EMBEDDING_DIM}d.txt'


def load_glove(vocab_only=False):
    if vocab_only:
        print('Loading GloVe vocabulary...')
    else:
        print('Loading GloVe embeddings...')

    word_vocab = dict()
    embeddings_list = []
    with open(GLOVE_PATH, encoding='utf-8') as glove_file:
        for i, line in enumerate(glove_file):
            values = line.split()
            word = values[0]
            word_vocab[word] = i
            if not vocab_only:
                embedding = np.array(values[1:], dtype='float32')
                embeddings_list.append(embedding)

    # Add zero embedding for UNK
    word_vocab[UNK_word] = len(word_vocab)
    if not vocab_only:
        embeddings_list.append(np.zeros((EMBEDDING_DIM,)))

    embeddings = np.array(embeddings_list)

    return word_vocab, embeddings


class GloVeEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, size_TE_label_vocab, size_TE_label_embed, TE_label_set, **kwargs):
        super(GloVeEmbeddingLayer, self).__init__(input_shape=(None, 2), **kwargs)

        self.size_TE_label_vocab = size_TE_label_vocab
        self.size_TE_label_embed = size_TE_label_embed
        self.TE_label_set = TE_label_set

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        # Load GloVe embeddings here so we don't have to save them when we save the layer/model
        word_vocab, embeddings = load_glove()
        self.word_embeddings = tf.keras.layers.Embedding(len(word_vocab), embeddings.shape[1],
                                                         weights=[embeddings], trainable=False,
                                                         input_shape=(None,))
        self.TE_label_embeddings = tf.keras.layers.Embedding(self.size_TE_label_vocab, self.size_TE_label_embed,
                                                             input_shape=(None,))

        super(GloVeEmbeddingLayer, self).build(input_shape)

    def get_config(self):
        config = super(GloVeEmbeddingLayer, self).get_config()
        config.update({
            'size_TE_label_vocab': self.size_TE_label_vocab,
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
