import math
import random
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from data_structures import EDGE_LABEL_LIST_UNLABELED, EDGE_LABEL_LIST_TIMEML, EDGE_LABEL_LIST_FULL, \
    LABEL_VOCAB_TIMEX_EVENT, LABEL_VOCAB_TIMEML, LABEL_VOCAB_FULL
from shared.input_label_sequence import InputLabelSequence

if tf.test.is_gpu_available():
    # Only show errors when running on GPU (no deprecation warnings)
    tf.logging.set_verbosity(tf.logging.ERROR)
else:
    # Show errors and warnings but no info
    tf.logging.set_verbosity(tf.logging.WARN)


class Classifier:
    def __init__(self, TE_label_set, edge_label_set):
        if edge_label_set == 'unlabeled':
            self.edge_label_set = EDGE_LABEL_LIST_UNLABELED
        elif edge_label_set == 'time_ml':
            self.edge_label_set = EDGE_LABEL_LIST_TIMEML
        else:
            self.edge_label_set = EDGE_LABEL_LIST_FULL

        self.size_edge_label = len(self.edge_label_set)

        self.TE_label_set = TE_label_set
        print('Using TE label set: {}, edge label set: {}'.format(TE_label_set, edge_label_set))

        self.TE_label_vocab = self.get_TE_label_vocab(TE_label_set)

        self.verbose = 2 if tf.test.is_gpu_available() else 1  # Hide progress bar on GPU

        self.model = None  # Initialise in subclass

    def get_TE_label_vocab(self, TE_label_set):
        if TE_label_set == 'none':
            return {}
        elif TE_label_set == 'timex_event':
            vocab = LABEL_VOCAB_TIMEX_EVENT
        elif TE_label_set == 'time_ml':
            vocab = LABEL_VOCAB_TIMEML
        else:
            vocab = LABEL_VOCAB_FULL

        return vocab

    def train(self, gold_training_data, silver_training_data, dev_data, output_file, labeled,
              epochs, early_stopping_warmup, early_stopping_threshold,
              blending_initialization, blending_epochs, blend_factor, learning_rate):
        start = timer()

        print('Compiling model and initializing variables...')
        self.compile_model(learning_rate)

        print('Preparing data...')
        gold_inputs_labels, silver_inputs_labels, dev_inputs_labels = self.data_to_inputs_and_gold(gold_training_data,
                                                                                                   silver_training_data,
                                                                                                   dev_data, labeled)
        self.train_model(gold_inputs_labels, silver_inputs_labels, dev_inputs_labels, epochs,
                         early_stopping_warmup, early_stopping_threshold,
                         blending_initialization, blending_epochs, blend_factor,
                         len(gold_training_data), len(silver_training_data) if silver_training_data else 0)

        end = timer()
        print('\nTrained model in {}s'.format(end - start), end='\n\n')

        self.save_model(output_file)

    def compile_model(self, learning_rate):
        raise NotImplementedError('Must override in subclass')

    def data_to_inputs_and_gold(self, gold_data, silver_data, dev_data, labeled):
        raise NotImplementedError('Must override in subclass')

    def train_model(self, gold_inputs_labels, silver_inputs_labels, dev_inputs_labels, epochs,
                    early_stopping_warmup, early_stopping_threshold,
                    blending_initialization, blending_epochs, blend_factor,
                    gold_document_count, silver_document_count):

        dev_sequence = InputLabelSequence(dev_inputs_labels)
        gold_sequence = InputLabelSequence(gold_inputs_labels)

        elapsed_epochs = 0

        # Handle blending
        if silver_inputs_labels:
            silver_sequence = InputLabelSequence(silver_inputs_labels)

            print('{} initialization epochs: Using {} silver documents'
                  .format(blending_initialization, silver_document_count))
            self.model.fit_generator(silver_sequence, steps_per_epoch=len(silver_inputs_labels),
                                     epochs=blending_initialization,
                                     validation_data=dev_sequence, validation_steps=len(dev_inputs_labels),
                                     shuffle=True, verbose=self.verbose)
            elapsed_epochs += blending_initialization

            for i in range(blending_epochs):
                # Print gold/silver ratio inside blend_data
                blended_inputs_labels = self.blend_data(gold_inputs_labels, silver_inputs_labels, i, blend_factor)
                blended_sequence = InputLabelSequence(blended_inputs_labels)
                self.model.fit_generator(blended_sequence, steps_per_epoch=len(blended_inputs_labels),
                                         epochs=1,
                                         validation_data=dev_sequence, validation_steps=len(dev_inputs_labels),
                                         shuffle=True, verbose=self.verbose)
            elapsed_epochs += blending_epochs

        # Handle additional early stopping (on top of blending, or if there was no blending)
        if elapsed_epochs < early_stopping_warmup:
            no_early_stopping_epochs = early_stopping_warmup - elapsed_epochs

            print('{} warm-up epochs: Using {} gold documents'
                  .format(no_early_stopping_epochs, gold_document_count))
            self.model.fit_generator(gold_sequence, steps_per_epoch=len(gold_inputs_labels),
                                     epochs=no_early_stopping_epochs,
                                     validation_data=dev_sequence, validation_steps=len(dev_inputs_labels),
                                     shuffle=True, verbose=self.verbose)
            elapsed_epochs += no_early_stopping_epochs

        # Perform remaining epochs with early stopping
        gold_epochs = epochs - elapsed_epochs
        if gold_epochs > 0:
            print('{} remaining epochs: Using {} gold documents'.format(gold_epochs, gold_document_count))
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=early_stopping_threshold, monitor='val_loss',
                                                          restore_best_weights=True)]
            self.model.fit_generator(gold_sequence, steps_per_epoch=len(gold_inputs_labels),
                                     epochs=gold_epochs,
                                     validation_data=dev_sequence, validation_steps=len(dev_inputs_labels),
                                     callbacks=callbacks,
                                     shuffle=True, verbose=self.verbose)

    def blend_data(self, gold_inputs_labels, silver_inputs_labels, blending_epoch_index,
                   blend_factor):
        """
        Blend gold and silver data with initialization period (silver only), blending period (gold and decreasing
        silver according to blend factor) and gold-only period

        N.B. this assumes that the number of documents and the number of inputs/labels is the same, i.e. each batch
        is one document, and the document is not split into multiple batches. Otherwise, it's possible that only
        part of a document will get included while blending.
        """

        # Blend gold and silver data, gradually decreasing the amount of silver data
        shuffled_silver_inputs_labels = random.sample(silver_inputs_labels, len(silver_inputs_labels))
        silver_ratio = math.pow(blend_factor, blending_epoch_index)
        silver_quantity = math.ceil(silver_ratio * len(silver_inputs_labels))
        # Keras will shuffle the gold and the silver later
        blended_inputs_labels = gold_inputs_labels + shuffled_silver_inputs_labels[:silver_quantity]
        print('Blending epoch {}: Using {} gold and {} silver documents'
              .format(blending_epoch_index, len(gold_inputs_labels), silver_quantity))

        return blended_inputs_labels

    def scores_to_predictions(self, scores_by_child, child_parent_candidates, labeled):
        sorted_predictions_by_child = []

        for scores, candidates_for_child in zip(scores_by_child, child_parent_candidates):
            labelled_scores = np.stack([np.arange(0, scores.shape[0]), scores], axis=1)
            # Sort by score, then reverse with [::-1]
            sorted_labelled_scores = labelled_scores[labelled_scores[:, 1].argsort()][::-1]

            sorted_predictions = []
            for label_and_score in sorted_labelled_scores:
                edge_label_index = int(label_and_score[0])
                score = label_and_score[1]
                prediction = self.edge_label_idx_to_prediction(edge_label_index, candidates_for_child, labeled)
                sorted_predictions.append((prediction, score))

            sorted_predictions_by_child.append(sorted_predictions)

        return sorted_predictions_by_child

    def edge_label_idx_to_prediction(self, candidate_edge_label_index, candidates, labeled):
        """Converts an index of an edge label within a list of candidate tuples for a document to an actual edge label.
        That is, the first candidate tuple occupies indices 0 to size_edge_label, enumerating each possible edge label,
        the second candidate tuple occupies indices size_edge_label+1 to 2*size_edge_label, etc.
        """
        document_index = int(candidate_edge_label_index / self.size_edge_label)
        label_index = candidate_edge_label_index % self.size_edge_label
        label = self.edge_label_set[label_index] if labeled else 'EDGE'

        candidate = candidates[document_index]
        parent, child, _ = candidate
        prediction = (parent, child, label)

        return prediction

    def save_model(self, output_file):
        raise NotImplementedError('Must override in subclass')
