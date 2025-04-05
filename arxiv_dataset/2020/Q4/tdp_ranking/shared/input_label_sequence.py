import tensorflow as tf


class InputLabelSequence(tf.keras.utils.Sequence):
    """
    Return model inputs and gold predictions one batch at a time (meaning that batches do not need to contain the same
    number of inputs).
    self.inputs should be a list of dictionaries of numpy arrays.
    """

    def __init__(self, inputs_labels):
        self.inputs = [inputs for inputs, labels in inputs_labels]
        self.labels = [labels for inputs, labels in inputs_labels]
        self.input_length = len(self.inputs)

    def __len__(self):
        return self.input_length

    def __getitem__(self, idx):
        # Return one batch's inputs and labels
        return self.inputs[idx], self.labels[idx]


class InputSequence(tf.keras.utils.Sequence):
    """
    Return model inputs one batch at a time (meaning that batches do not need to contain the same number of inputs).
    self.inputs should be a list of dictionaries of numpy arrays.
    """

    def __init__(self, inputs):
        self.inputs = inputs
        self.input_length = len(self.inputs)

    def __len__(self):
        return self.input_length

    def __getitem__(self, idx):
        # Return one batch's inputs and labels
        return self.inputs[idx]
