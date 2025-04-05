import gc
import json

import tensorflow as tf

from frozen_bert.model import Model, get_model_inputs, data_to_inputs_and_gold
from shared.attention import AttentionLayer, NodeToWordsLayer
from shared.bert_layer import create_tokenizer_from_hub_module, BertLayer, BatchedBertLayer
from shared.classifier import Classifier
from shared.features import FeatureLayer

if tf.test.is_gpu_available():
    # Only show errors when running on GPU (no deprecation warnings)
    tf.logging.set_verbosity(tf.logging.ERROR)
else:
    # Show errors and warnings but no info
    tf.logging.set_verbosity(tf.logging.WARN)


class BertBilstmClassifier(Classifier):
    def __init__(self, TE_label_set, size_TE_label_embed, size_lstm, size_feed_forward, edge_label_set,
                 max_sequence_length, max_words_per_node, max_candidate_count, disable_handcrafted_features):
        super().__init__(TE_label_set, edge_label_set)

        # Keep parameters so we can write them to a save file
        self.save_config = {'TE_label_set': TE_label_set, 'size_TE_label_embed': size_TE_label_embed,
                            'size_lstm': size_lstm, 'size_feed_forward': size_feed_forward,
                            'edge_label_set': edge_label_set, 'max_sequence_length': max_sequence_length,
                            'max_words_per_node': max_words_per_node, 'max_candidate_count': max_candidate_count,
                            'disable_handcrafted_features': disable_handcrafted_features}

        self.max_sequence_length = max_sequence_length
        self.max_words_per_node = max_words_per_node
        self.disable_handcrafted_features = disable_handcrafted_features

        print('Building model...')
        self.model = Model(len(self.TE_label_vocab), size_TE_label_embed, TE_label_set, size_lstm, size_feed_forward,
                           self.size_edge_label, max_sequence_length, max_words_per_node, max_candidate_count,
                           disable_handcrafted_features)

    def compile_model(self, learning_rate):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss=tf.keras.losses.categorical_crossentropy,
                           metrics=[tf.keras.metrics.categorical_accuracy])
        # Ensure BERT variables initialized
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())

    def data_to_inputs_and_gold(self, gold_data, silver_data, dev_data, labeled):
        bert_tokenizer = create_tokenizer_from_hub_module()
        gold_inputs_labels = data_to_inputs_and_gold(gold_data, bert_tokenizer, labeled, self.TE_label_vocab,
                                                     self.TE_label_set, self.edge_label_set, self.max_words_per_node,
                                                     self.max_sequence_length, self.disable_handcrafted_features)
        silver_inputs_labels = None
        if silver_data:
            silver_inputs_labels = data_to_inputs_and_gold(silver_data, bert_tokenizer, labeled, self.TE_label_vocab,
                                                           self.TE_label_set, self.edge_label_set,
                                                           self.max_words_per_node, self.max_sequence_length,
                                                           self.disable_handcrafted_features)
        dev_inputs_labels = data_to_inputs_and_gold(dev_data, bert_tokenizer, labeled, self.TE_label_vocab,
                                                    self.TE_label_set, self.edge_label_set, self.max_words_per_node,
                                                    self.max_sequence_length, self.disable_handcrafted_features)

        return gold_inputs_labels, silver_inputs_labels, dev_inputs_labels

    def predict(self, sentence_list, child_parent_candidates, labeled):
        """
        Given a document (tuple of sentence list and child/parent candidate list),
        predict the parent of each child. For each child, return the scores for all possible parents, sorted descending,
        as pairs of the predicted tuple and its score.
        """

        bert_tokenizer = create_tokenizer_from_hub_module()
        model_inputs = get_model_inputs(bert_tokenizer, sentence_list, child_parent_candidates,
                                        self.TE_label_vocab, self.TE_label_set, self.max_words_per_node,
                                        self.max_sequence_length, self.disable_handcrafted_features)

        # Remove batch dimension with [0]
        scores_by_child = self.model.predict(model_inputs, batch_size=1, verbose=self.verbose)[0]

        return self.scores_to_predictions(scores_by_child, child_parent_candidates, labeled)

    def save_model(self, output_file):
        self.model.save(output_file + '.h5')
        with open(output_file + '.config', 'w', encoding='utf-8') as file:
            json.dump(self.save_config, file)

    def load_model(self, model_file):
        # Manage memory
        tf.keras.backend.clear_session()
        del self.model
        gc.collect()

        # Load model
        self.model = tf.keras.models.load_model(model_file + '.h5', custom_objects={
            'BertLayer': BertLayer,
            'BatchedBertLayer': BatchedBertLayer,
            'AttentionLayer': AttentionLayer,
            'NodeToWordsLayer': NodeToWordsLayer,
            'FeatureLayer': FeatureLayer,
        })

    @classmethod
    def load(cls, model_file):
        with open(model_file + '.config', 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

            classifier = cls(config['TE_label_set'], config['size_TE_label_embed'],
                             config['size_lstm'], config['size_feed_forward'],
                             config['edge_label_set'], config['max_sequence_length'],
                             config['max_words_per_node'], config['max_candidate_count'],
                             config['disable_handcrafted_features'])

            classifier.load_model(model_file)
            return classifier
