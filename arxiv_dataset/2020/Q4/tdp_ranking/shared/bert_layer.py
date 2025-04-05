"""
Adapted from https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py
and https://github.com/strongio/keras-bert/blob/master/keras-bert.ipynb
"""
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
# noinspection PyProtectedMember
from bert.run_classifier import _truncate_seq_pair
from bert.tokenization import FullTokenizer

bert_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
# Tell TF Hub to use cached version - n.B. if changing the bert_path, must update hash in path to vocab_file
# See https://medium.com/@xianbao.qian/how-to-run-tf-hub-locally-without-internet-connection-4506b850a915
cache_dir = '/tmp' # SET YOUR DESIRED CACHE DIRECTORY HERE
os.environ["TFHUB_CACHE_DIR"] = cache_dir
vocab_file = cache_dir + '/5a395eafef2a37bd9fc55d7f6ae676d2a134a838/assets/vocab.txt'
do_lower_case = True

bert_output_size = 768


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, words_a, words_b=None):
        """Constructs a InputExample.
        Args:
          words_a: list of strings. The words of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          words_b: (Optional) list of strings. The words of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.words_a = words_a
        self.words_b = words_b


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_output, max_seq_length, n_fine_tune_layers=1, **kwargs):
        self.max_seq_length = max_seq_length
        self.n_fine_tune_layers = n_fine_tune_layers
        self.output_size = bert_output_size
        self.bert_path = bert_path

        if bert_output not in ['sequence_output', 'pooled_output']:
            raise ValueError('Invalid BERT output')
        self.bert_output = bert_output

        if 'trainable' in kwargs:
            self.trainable = kwargs['trainable']
            kwargs.pop('trainable')
        else:
            self.trainable = n_fine_tune_layers > 0

        super(BertLayer, self).__init__(trainable=self.trainable, **kwargs)

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.bert = hub.Module(self.bert_path, trainable=self.trainable, name=f'{self.name}_module')

        self.set_trainable_variables()

        super(BertLayer, self).build(input_shape)

    def set_trainable_variables(self):
        # Remove unused layers
        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars if '/cls/' not in var.name and '/pooler/' not in var.name]
        trainable_layers = []

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f'encoder/layer_{str(11 - i)}')

        # Update trainable vars to contain only the specified layers
        trainable_vars = [var for var in trainable_vars if any([layer in var.name for layer in trainable_layers])]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

    def get_config(self):
        config = super(BertLayer, self).get_config()
        config.update({
            'bert_output': self.bert_output,
            'max_seq_length': self.max_seq_length,
            'n_fine_tune_layers': self.n_fine_tune_layers,
        })
        return config

    def call(self, inputs, **kwargs):
        inputs = [tf.cast(x, dtype='int32') for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)

        result = self.bert(inputs=bert_inputs, signature='tokens', as_dict=True)
        # Result has keys 'sequence_output' and 'pooled_output'
        # sequence_output is a [batch_size, sequence_length, hidden_size] Tensor, one hidden_size vector per input
        # pooled_output is a [batch_size, hidden_size] Tensor
        # n.B. layers deeper than last layer are not accessible with the TF Hub module
        return result[self.bert_output]

    def compute_output_shape(self, input_shape):
        input_ids_shape = input_shape[0]  # input_shape is a list because this takes a list of inputs
        batch_size = input_ids_shape[0]
        if self.bert_output == 'sequence_output':
            return tf.TensorShape([batch_size, self.max_seq_length, self.output_size])
        else:
            return tf.TensorShape([batch_size, self.output_size])


class BatchedBertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_output, max_seq_length, n_fine_tune_layers=1, **kwargs):
        super(BatchedBertLayer, self).__init__(**kwargs)

        self.bert_output = bert_output
        self.max_seq_length = max_seq_length
        self.n_fine_tune_layers = n_fine_tune_layers

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.bert_layer = BertLayer(self.bert_output, self.max_seq_length, self.n_fine_tune_layers)

        super(BatchedBertLayer, self).build(input_shape)

    def get_config(self):
        config = super(BatchedBertLayer, self).get_config()
        config.update({
            'bert_output': self.bert_output,
            'max_seq_length': self.max_seq_length,
            'n_fine_tune_layers': self.n_fine_tune_layers,
        })
        return config

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs[0])[0]
        flattened_inputs = [tf.reshape(x, shape=(-1, self.max_seq_length)) for x in inputs]

        bert_outputs = self.bert_layer(flattened_inputs)

        bert_output_dims = list(self.compute_bert_output_shape([inp.shape for inp in flattened_inputs]))[1:]
        batched_shape = [batch_size, -1] + bert_output_dims
        batched_outputs = tf.reshape(bert_outputs, shape=batched_shape)
        return batched_outputs

    def compute_output_shape(self, input_shape):
        input_ids_shape = input_shape[0]
        batch_size = input_ids_shape[0]  # input_shape is a list because this takes a list of inputs
        inputs_per_batch = input_ids_shape[1]
        bert_output_shape = self.compute_bert_output_shape(input_shape)
        return tf.TensorShape([batch_size, inputs_per_batch] + list(bert_output_shape)[1:])

    def compute_bert_output_shape(self, input_shape):
        bert_input_shape = [shape[1:] for shape in input_shape]
        return self.bert_layer.compute_output_shape(bert_input_shape)


def create_tokenizer_from_hub_module():
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length):
    """Converts a single `InputExample` into a single tuple of inputs."""

    tokens_a, tokens_b, words_to_tokens_map = example_to_tokens(tokenizer, example, max_seq_length)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where 'type_ids' are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the 'sentence vector'. Note that this only makes sense because
    # the entire model is fine-tuned.

    tokens = []
    segment_ids = []
    tokens.append('[CLS]')
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, words_to_tokens_map


def example_to_tokens(tokenizer, example, max_seq_length):
    # Create map of [start_token_index, end_token_index] for each word in words_a followed by words_b
    # where end_token_index is inclusive (i.e. slice the output with [start_token_index, end_token_index + 1])
    words_to_tokens_map_a = []
    tokens_a = []

    for word in example.words_a:
        word_tokens = tokenizer.tokenize(word)
        start_index = len(tokens_a)
        words_to_tokens_map_a.append([start_index, start_index + len(word_tokens) - 1])
        tokens_a.extend(word_tokens)

    words_to_tokens_map_b = []
    tokens_b = []

    if example.words_b:
        for word in example.words_b:
            word_tokens = tokenizer.tokenize(word)
            start_index = len(tokens_b)
            words_to_tokens_map_b.append([start_index, start_index + len(word_tokens) - 1])
            tokens_b.extend(word_tokens)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with '- 3'
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with '- 2'
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    words_to_tokens_map = []
    a_offset = 1  # Add 1 for [CLS]
    for token_map in words_to_tokens_map_a:
        start_index = token_map[0] + a_offset if token_map[0] < len(tokens_a) else -1
        end_index = token_map[1] + a_offset if token_map[1] < len(tokens_a) else -1
        words_to_tokens_map.append([start_index, end_index])

    b_offset = len(tokens_a) + 2  # Add 2 for [CLS] and [SEP]
    for token_map in words_to_tokens_map_b:
        start_index = token_map[0] + b_offset if token_map[0] < len(tokens_b) else -1
        end_index = token_map[1] + b_offset if token_map[1] < len(tokens_b) else -1
        words_to_tokens_map.append([start_index, end_index])

    return tokens_a, tokens_b, words_to_tokens_map


def convert_examples_to_inputs(tokenizer, examples, max_seq_length):
    """Convert a set of `InputExample`s to lists of inputs, grouped by type into a tuple."""

    input_ids, input_masks, segment_ids, words_to_tokens_maps = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, words_to_tokens_map = convert_single_example(tokenizer, example,
                                                                                       max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        words_to_tokens_maps.append(words_to_tokens_map)

    bert_inputs = {
        'input_ids': np.array(input_ids),
        'input_masks': np.array(input_masks),
        'segment_ids': np.array(segment_ids)
    }
    return bert_inputs, words_to_tokens_maps
