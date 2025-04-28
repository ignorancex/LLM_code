"""
List the keys available for each task / expected from each task in this t5 multitask generic implementation
"""

# DATA
SAMPLE_ID = "data.sample_id"

# expected outputs foreach task - data pipeline
ENCODER_INPUTS_TOKENS = "data.encoder_input_token_ids"  # encoder input token ids
ENCODER_INPUTS_STR = "data.query.encoder_input"  # the original string representation of encoder input - used for debug
ENCODER_INPUTS_ATTENTION_MASK = "data.encoder_input_attention_mask"  # attention mask of the tokenized encoder input (output of the tokenizer)
DECODER_INPUTS_STR = "data.query.decoder_input"  # the original string representation of decoder input - used for debug
DECODER_INPUTS_TOKENS = "data.decoder_input_token_ids"  # decoder input token ids (decoder start token followed by labels token ids)
DECODER_INPUTS_ATTENTION_MASK = "data.decoder_input_attention_mask"  # attention mask of the tokenized decoder input (output of the tokenizer)
LABELS_TOKENS = "data.labels_token_ids"  # labels token ids
LABELS_ATTENTION_MASK = "data.labels_attention_mask"  # attention mask of the tokenized labels (output of the tokenizer)
LABELS_STR = (
    "data.query.labels"  # the original string representation of labels - used for debug
)

# adding custom embeddings
ENCODER_INPUT_ADD_EMBEDDINGS = "data.encoder_input_add_embeddings"  # optional, can be used to add (in additional to token_ids) custom embeddings


# the list of keys each task (data_module) must add to sample_dict for training in encoder only mode
DATA_KEYS_ENCODER = [
    ENCODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    LABELS_TOKENS,
    LABELS_ATTENTION_MASK,
    LABELS_STR,
    SAMPLE_ID,
]

# the list of keys each task (data_module) must add to sample_dict for training in encoder-decoder mode
DATA_KEYS = DATA_KEYS_ENCODER + [DECODER_INPUTS_TOKENS, DECODER_INPUTS_ATTENTION_MASK]

# MODEL
# expected model outputs
LOGITS = "model.out.logits"
SCORES = "model.out.scores"  # model output after softmax
CLS_PRED = "model.out.cls_pred"  # result argmax() to get teacher forcing prediction
CE_LOSS = "model.out.loss"  # cross-entropy loss calculated in t5 model
ENCODER_LAST_HIDDEN_STATE = "model.out.encoder_last_hidden_state"  # encoder head logits

########################################
#### related to scalars inputs/outputs
########################################


# logits of the scalars output prediction head - a single scalar is predicted per input element
# active only in encoder-only mode!
SCALARS_PREDICTION_HEAD_LOGITS = "model.out.scalars_prediction_logits"

ENCODER_INPUTS_SCALARS = "data.encoder_input.scalars"
# a float tensor with the values of scalars. A default value is used for elements that are not scalars. Its length is the number of input elements
ENCODER_INPUTS_SCALARS_VALUES = ENCODER_INPUTS_SCALARS + ".values"
# a boolean tensor with the values of scalars. True/False per element describes what are scalar elements . Its length is the number of input elements
ENCODER_INPUTS_SCALARS_VALID_MASK = ENCODER_INPUTS_SCALARS + ".valid_mask"


LABELS_SCALARS = "data.labels.scalars"
# a float tensor with the values of scalars. A default value is used for elements that are not scalars. Its length is the number of input elements
LABELS_SCALARS_VALUES = LABELS_SCALARS + ".values"
# a boolean tensor with the values of scalars. True/False per element describes what are scalar elements . Its length is the number of input elements
LABELS_SCALARS_VALID_MASK = LABELS_SCALARS + ".valid_mask"
