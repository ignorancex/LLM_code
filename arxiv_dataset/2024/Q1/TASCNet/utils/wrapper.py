from typing import *
from tensorflow.keras import models
from functools import reduce
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow as tf


class OptimizerWrapper(models.Model):
    def __init__(
        self,
        model: models.Model,
        filter_pos_episodes: Dict[int, List[Tuple]] = None,
        filter_neg_episodes: Dict[int, List[Tuple]] = None,
        max_measure: Optional[Callable] = None,
        min_measure: Optional[Callable] = None,
        pos_lambda: float = 1.0,
        neg_lambda: float = 1.0,
        **kwargs
    ):
        """
        Parameters:
        -----------
        model: A Keras model object to optimize
        filter_pos_episodes: A dictionary with conv layer index as key and list of paired up filter
                            indices as value. Filter indices to bring together
        filter_neg_episodes: Same as filter_pos_episodes but the filters to separate even further apart.
        max_measure: measure function for filter_pos_episodes. Left None will default to cosine similarity.
        min_measure: measure function for filter_neg_episodes. Left None will default to cosine similarity.
        """
        super(OptimizerWrapper, self).__init__(**kwargs)
        self.model = model
        self.neg_episodes = filter_neg_episodes
        self.outputs = []
        try:
            assert filter_pos_episodes
            self.pos_episodes = filter_pos_episodes
        except:
            raise TypeError("Argument `filter_pos_episodes` missing.")

        self.max_measure = max_measure if max_measure else self.cosine_similarity
        if self.neg_episodes:
            self.min_measure = min_measure if min_measure else self.cosine_similarity

        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda

    def call(self, data):
        return self.model(data)

    def cosine_similarity(self, filter1, filter2):
        f1 = K.reshape(filter1, (-1, 1))
        f2 = K.reshape(filter2, (-1, 1))

        f1_norm = K.sqrt(K.sum(f1**2))
        f2_norm = K.sqrt(K.sum(f2**2))

        numerator = K.sum(f1 * f2)

        sim = numerator / (f1_norm * f2_norm)
        return sim

    def get_total_sim(self, episodes, measure):
        sim_measures = []
        for key in episodes.keys():
            temp = []
            for pair in episodes[key]:
                temp.append(
                    measure(
                        self.model.layers[key].weights[0][:, :, :, pair[0]],
                        self.model.layers[key].weights[0][:, :, :, pair[1]],
                    )
                )

            temp = reduce(lambda x, y: x + y, temp)
            sim_measures.append(temp)

        regularizer = reduce(lambda x, y: x + y, sim_measures)
        return regularizer

    def train_step(self, data):
        input_, labels, _ = data

        with tf.GradientTape(persistent=True) as tape:
            pred_labels = self.call(input_)
            step_loss = self.compiled_loss(labels, pred_labels)

            # add filter pos regularizer
            if self.pos_episodes:
                step_loss += self.pos_lambda * K.exp(
                    -self.get_total_sim(self.pos_episodes, self.max_measure)
                )

            # add filter neg regularizer
            if self.neg_episodes:
                step_loss += self.neg_lambda * K.exp(
                    self.get_total_sim(self.neg_episodes, self.min_measure)
                )

        grads = tape.gradient(step_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(labels, pred_labels)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        input_, labels = data

        pred_labels = self.call(input_)
        step_loss = self.compiled_loss(labels, pred_labels)

        # add filter pos regularizer
        if self.pos_episodes:
            step_loss += self.pos_lambda * K.exp(
                -self.get_total_sim(self.pos_episodes, self.max_measure)
            )

        # add filter neg regularizer
        if self.neg_episodes:
            step_loss += self.neg_lambda * K.exp(
                self.get_total_sim(self.neg_episodes, self.min_measure)
            )

        self.compiled_metrics.update_state(labels, pred_labels)

        return {m.name: m.result() for m in self.metrics}
