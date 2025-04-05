import gc
import json

import tensorflow as tf

from base_bilstm.embedding import EmbeddingLayer
from base_bilstm.model import Model, get_model_inputs, data_to_inputs_and_gold
from shared.attention import AttentionLayer, NodeToWordsLayer
from shared.classifier import Classifier
from shared.features import FeatureLayer

if tf.test.is_gpu_available():
    # Only show errors when running on GPU (no deprecation warnings)
    tf.logging.set_verbosity(tf.logging.ERROR)
else:
    # Show errors and warnings but no info
    tf.logging.set_verbosity(tf.logging.WARN)


class BilstmClassifier(Classifier):
    def __init__(self, word_vocab, size_word_embed, size_lstm, size_feed_forward,
                 TE_label_set, size_TE_label_embed, edge_label_set, max_words_per_node, max_candidate_count,
                 disable_handcrafted_features):
        super().__init__(TE_label_set, edge_label_set)

        # Keep parameters so we can write them to a save file
        self.save_config = {'size_word_embed': size_word_embed, 'size_lstm': size_lstm,
                            'size_feed_forward': size_feed_forward, 'TE_label_set': TE_label_set,
                            'size_TE_label_embed': size_TE_label_embed, 'edge_label_set': edge_label_set,
                            'max_words_per_node': max_words_per_node, 'max_candidate_count': max_candidate_count,
                            'disable_handcrafted_features': disable_handcrafted_features}

        self.word_vocab = word_vocab  # Dictionary of strings to word IDs

        self.max_words_per_node = max_words_per_node
        self.disable_handcrafted_features = disable_handcrafted_features

        self.model = Model(len(self.word_vocab), len(self.TE_label_vocab), size_word_embed, size_lstm,
                           size_feed_forward, TE_label_set, size_TE_label_embed, self.size_edge_label,
                           max_words_per_node, max_candidate_count, disable_handcrafted_features)

    def compile_model(self, learning_rate):
        # Compile model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss=tf.keras.losses.categorical_crossentropy,
                           metrics=[tf.keras.metrics.categorical_accuracy])

        # Print model summary
        self.model.summary()

    def data_to_inputs_and_gold(self, gold_data, silver_data, dev_data, labeled):
        gold_inputs_labels = data_to_inputs_and_gold(gold_data, labeled, self.word_vocab, self.TE_label_vocab,
                                                     self.TE_label_set, self.edge_label_set, self.max_words_per_node,
                                                     self.disable_handcrafted_features)
        silver_inputs_labels = None
        if silver_data:
            silver_inputs_labels = data_to_inputs_and_gold(gold_data, labeled, self.word_vocab, self.TE_label_vocab,
                                                           self.TE_label_set, self.edge_label_set,
                                                           self.max_words_per_node,
                                                           self.disable_handcrafted_features)
        dev_inputs_labels = data_to_inputs_and_gold(dev_data, labeled, self.word_vocab, self.TE_label_vocab,
                                                    self.TE_label_set, self.edge_label_set, self.max_words_per_node,
                                                    self.disable_handcrafted_features)

        return gold_inputs_labels, silver_inputs_labels, dev_inputs_labels

    def predict(self, sentence_list, child_parent_candidates, labeled):
        """
        Given a document (tuple of sentence list and child/parent candidate list),
        predict the parent of each child. For each child, return the scores for all possible parents, sorted descending,
        as pairs of the predicted tuple and its score.
        """

        model_inputs = get_model_inputs(sentence_list, child_parent_candidates, self.word_vocab,
                                        self.TE_label_vocab, self.TE_label_set, self.max_words_per_node,
                                        self.disable_handcrafted_features)

        # Remove batch dimension with [0]
        scores_by_child = self.model.predict(model_inputs, batch_size=1, verbose=self.verbose)[0]

        return self.scores_to_predictions(scores_by_child, child_parent_candidates, labeled)

    def save_model(self, output_file):
        self.model.save(output_file + '.h5')
        with open(output_file + '.config', 'w', encoding='utf-8') as file:
            json.dump(self.save_config, file)
        self.save_vocab(output_file + '.vocab')

    def load_model(self, model_file):
        # Manage memory
        tf.keras.backend.clear_session()
        del self.model
        gc.collect()

        # Load model
        self.model = tf.keras.models.load_model(model_file + '.h5', custom_objects={
            'EmbeddingLayer': EmbeddingLayer,
            'AttentionLayer': AttentionLayer,
            'NodeToWordsLayer': NodeToWordsLayer,
            'FeatureLayer': FeatureLayer,
        })

    def save_vocab(self, vocab_file):
        # Store vocabulary of this run
        with open(vocab_file, 'w', encoding='utf-8') as file:
            json.dump(self.word_vocab, file)

    @classmethod
    def load(cls, model_file):
        with open(model_file + '.vocab', 'r', encoding='utf-8') as vocab_file, \
                open(model_file + '.config', 'r', encoding='utf-8') as config_file:
            word_vocab = json.load(vocab_file)
            config = json.load(config_file)

            classifier = cls(word_vocab, config['size_word_embed'], config['size_lstm'], config['size_feed_forward'],
                             config['TE_label_set'], config['size_TE_label_embed'], config['edge_label_set'],
                             config['max_words_per_node'], config['max_candidate_count'],
                             config['disable_handcrafted_features'])

            classifier.load_model(model_file)

            return classifier
