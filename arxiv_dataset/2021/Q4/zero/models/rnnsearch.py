# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from func import linear
from models import model
from utils import util, dtype
from rnns import rnn


def encoder(source, params):
    mask = dtype.tf_to_float(tf.cast(source, tf.bool))
    hidden_size = params.hidden_size

    source, mask = util.remove_invalid_seq(source, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "src_embedding"
    src_emb = tf.get_variable(embed_name,
                              [params.src_vocab.size(), params.embed_size])
    src_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(src_emb, source)
    inputs = tf.nn.bias_add(inputs, src_bias)

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("encoder"):
        # forward rnn
        with tf.variable_scope('forward'):
            outputs = rnn.rnn(params.cell, inputs, hidden_size, mask=mask,
                              ln=params.layer_norm, sm=params.swap_memory)
            output_fw, state_fw = outputs[1]
        # backward rnn
        with tf.variable_scope('backward'):
            if not params.caencoder:
                outputs = rnn.rnn(params.cell, tf.reverse(inputs, [1]),
                                  hidden_size, mask=tf.reverse(mask, [1]),
                                  ln=params.layer_norm, sm=params.swap_memory)
                output_bw, state_bw = outputs[1]
            else:
                outputs = rnn.cond_rnn(params.cell, tf.reverse(inputs, [1]),
                                       tf.reverse(output_fw, [1]), hidden_size,
                                       mask=tf.reverse(mask, [1]),
                                       ln=params.layer_norm,
                                       sm=params.swap_memory,
                                       num_heads=params.num_heads,
                                       one2one=True)
                output_bw, state_bw = outputs[1]

            output_bw = tf.reverse(output_bw, [1])

    if not params.caencoder:
        source_encodes = tf.concat([output_fw, output_bw], -1)
        source_feature = tf.concat([state_fw, state_bw], -1)
    else:
        source_encodes = output_bw
        source_feature = state_bw

    with tf.variable_scope("decoder_initializer"):
        decoder_init = rnn.get_cell(
            params.cell, hidden_size, ln=params.layer_norm
        ).get_init_state(x=source_feature)
    decoder_init = tf.tanh(decoder_init)

    return {
        "encodes": source_encodes,
        "decoder_initializer": decoder_init,
        "mask": mask
    }


def decoder(target, state, params):
    mask = dtype.tf_to_float(tf.cast(target, tf.bool))
    hidden_size = params.hidden_size

    is_training = ('decoder' not in state)

    if is_training:
        target, mask = util.remove_invalid_seq(target, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "tgt_embedding"
    tgt_emb = tf.get_variable(embed_name,
                              [params.tgt_vocab.size(), params.embed_size])
    tgt_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(tgt_emb, target)
    inputs = tf.nn.bias_add(inputs, tgt_bias)

    # shift
    if is_training:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
        inputs = inputs[:, :-1, :]
    else:
        inputs = tf.cond(tf.reduce_all(tf.equal(target, params.tgt_vocab.pad())),
                         lambda: tf.zeros_like(inputs),
                         lambda: inputs)
        mask = tf.ones_like(mask)

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("decoder"):
        init_state = state["decoder_initializer"]
        if not is_training:
            init_state = state["decoder"]["state"]
        returns = rnn.cond_rnn(params.cell, inputs, state["encodes"], hidden_size,
                               init_state=init_state, mask=mask,
                               mem_mask=state["mask"], ln=params.layer_norm,
                               sm=params.swap_memory, one2one=False)
        (_, hidden_state), (outputs, _), contexts, attentions = returns

    feature = linear([outputs, contexts, inputs], params.embed_size,
                     ln=params.layer_norm, scope="pre_logits")
    if 'dev_decode' in state:
        feature = feature[:, -1, :]

    feature = tf.tanh(feature)
    feature = util.valid_apply_dropout(feature, params.dropout)

    embed_name = "tgt_embedding" if params.shared_target_softmax_embedding \
        else "softmax_embedding"
    embed_name = "embedding" if params.shared_source_target_embedding \
        else embed_name
    softmax_emb = tf.get_variable(embed_name,
                                  [params.tgt_vocab.size(), params.embed_size])
    feature = tf.reshape(feature, [-1, params.embed_size])
    logits = tf.matmul(feature, softmax_emb, False, True)

    logits = tf.cast(logits, tf.float32)

    soft_label, normalizer = util.label_smooth(
        target,
        util.shape_list(logits)[-1],
        factor=params.label_smooth)
    centropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=soft_label
    )
    centropy -= normalizer
    centropy = tf.reshape(centropy, tf.shape(target))

    mask = tf.cast(mask, tf.float32)
    per_sample_loss = tf.reduce_sum(centropy * mask, -1) / tf.reduce_sum(mask, -1)
    loss = tf.reduce_mean(per_sample_loss)

    # these mask tricks mainly used to deal with zero shapes, such as [0, 1]
    loss = tf.cond(tf.equal(tf.shape(target)[0], 0),
                   lambda: tf.constant(0, dtype=tf.float32),
                   lambda: loss)

    if not is_training:
        state['decoder']['state'] = hidden_state

    return loss, logits, state, per_sample_loss


def train_fn(features, params, initializer=None):
    with tf.variable_scope(params.scope_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
        state = encoder(features['source'], params)
        loss, logits, state, _ = decoder(features['target'], state, params)

        return {
            "loss": loss
        }


def score_fn(features, params, initializer=None):
    params = copy.copy(params)
    params = util.closing_dropout(params)
    params.label_smooth = 0.0
    with tf.variable_scope(params.scope_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
        state = encoder(features['source'], params)
        _, _, _, scores = decoder(features['target'], state, params)

        return {
            "score": scores
        }


def infer_fn(params):
    params = copy.copy(params)
    params = util.closing_dropout(params)

    def encoding_fn(source):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            state = encoder(source, params)
            state["decoder"] = {
                "state": state["decoder_initializer"]
            }
            return state

    def decoding_fn(target, state, time):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            if params.search_mode == "cache":
                step_loss, step_logits, step_state, _ = decoder(
                    target, state, params)
            else:
                estate = encoder(state, params)
                estate['dev_decode'] = True
                _, step_logits, _, _ = decoder(target, estate, params)
                step_state = state

            return step_logits, step_state

    return encoding_fn, decoding_fn


# register the model, with a unique name
model.model_register("rnnsearch", train_fn, score_fn, infer_fn)
