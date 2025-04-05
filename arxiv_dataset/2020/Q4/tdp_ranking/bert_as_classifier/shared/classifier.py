import gc
import json

import numpy as np
import tensorflow as tf

from bert_as_classifier.shared.model import Model
from shared.bert_layer import BertLayer
from shared.classifier import Classifier
from shared.input_label_sequence import InputSequence


# noinspection PyAbstractClass
class BertBaseClassifier(Classifier):
    def __init__(self, TE_label_set, edge_label_set, max_sequence_length, max_candidate_count,
                 disable_handcrafted_features):
        super().__init__(TE_label_set, edge_label_set)

        # Keep parameters so we can write them to a save file
        self.save_config = {'TE_label_set': TE_label_set, 'edge_label_set': edge_label_set,
                            'max_sequence_length': max_sequence_length, 'max_candidate_count': max_candidate_count,
                            'disable_handcrafted_features': disable_handcrafted_features}

        self.max_sequence_length = max_sequence_length
        self.max_candidate_count = max_candidate_count
        self.disable_handcrafted_features = disable_handcrafted_features

        self.model = Model(max_sequence_length, self.size_edge_label, disable_handcrafted_features)

    def compile_model(self, learning_rate):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss=self.loss,
                           metrics=[self.accuracy])
        # Ensure BERT variables initialized
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())

    def loss(self, y_true, y_pred):
        # We actually only have a true one-hot vector per batch/child, so flatten the predictions and the labels
        flat_y_true = tf.reshape(y_true, shape=(-1,))
        flat_y_pred = tf.reshape(y_pred, shape=(-1,))
        return tf.keras.losses.categorical_crossentropy(flat_y_true, flat_y_pred, from_logits=True)

    def accuracy(self, y_true, y_pred):
        # We actually only have a true one-hot vector per batch/child, so flatten the predictions and the labels
        flat_y_true = tf.reshape(y_true, shape=(-1,))
        flat_y_pred = tf.reshape(y_pred, shape=(-1,))
        return tf.keras.metrics.categorical_accuracy(flat_y_true, flat_y_pred)

    def data_to_model_inputs(self, sentence_list, child_parent_candidates):
        raise NotImplementedError('Must be implemented by subclass')

    def predict(self, sentence_list, child_parent_candidates, labeled):
        """
        Given a document (tuple of sentence list and child/parent candidate list),
        predict the parent of each child. For each child, return the scores for all possible parents, sorted descending,
        as pairs of the predicted tuple and its score.
        """

        model_inputs = self.data_to_model_inputs(sentence_list, child_parent_candidates)
        input_sequence = InputSequence(model_inputs)

        candidate_scores = self.model.predict_generator(input_sequence, steps=len(input_sequence),
                                                        verbose=self.verbose)

        # Group by child
        scores_by_child = np.reshape(candidate_scores, (-1, self.max_candidate_count * self.size_edge_label))

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
            'loss': self.loss,
            'accuracy': self.accuracy
        })


