from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping
from time import time
import numpy as np
from collections import Counter

from tensorflow.python.keras.layers import Embedding, Dropout, GaussianNoise, LSTM, Bidirectional, Dense, Flatten, \
    Activation, BatchNormalization
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.np_utils import to_categorical

from config import TENSOR_BOARD_LOGS, LOGGING_FORMAT, LOGGING_DATE_FORMAT, LSTM_TMP_DIR, LSTM_TMP_HISTOGRAMS
from src.polarity.baseline.utils import label_probabilities, de_vectorize, evaluate_baseline_model
from src.polarity.lstm_baseline.embeddings.EmbeddingsMatrixManager import EmbeddingsMatrixManager
from src.polarity.lstm_baseline.embeddings.EmbeddingsVectorizer import EmbeddingsVectorizer
from src.polarity.lstm_baseline.callbacks import MyTensorBoardCallback, MyWandbCallbackLogger, MyOriginalWandbCallback
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns


from src.polarity.lstm_baseline.nn_config import get_lang_configs
from src.polarity.lstm_baseline.nn_utils import generate_file_name, build_param_string, get_optimizer, get_learning_rate, \
    compute_number_of_steps
from src.polarity.lstm_baseline.sequence_utils import extract_n_gram_seq
from src.polarity.preprocessing.Data_Cleaner import Data_Cleaner, text_processor_normalize
from src.polarity.preprocessing.demo_data import demo_sents

logger = logging.getLogger(__name__)



def build_model_base(we_matrix, n_classes, expected_steps, **kwargs):
    max_sequence_length = kwargs.get("max_seq_len", 256)

    # variables for embedding
    trainable_we = kwargs.get("trainable_word_embeddings", False)
    use_masking = kwargs.get("use_masking", True)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    optimizer_name = kwargs.get("optimizer", 'Adam')
    initial_learing_rate = kwargs.get("lr")
    lr_scheduler_name = kwargs.get("lr_scheduler_name", None)
    rnn_cells = kwargs.get("rnn_cells")
    final_layer_size = kwargs.get("final_layer_size")
    bidirectional = kwargs.get("bidirectional", False)
    use_attention = kwargs.get("use_attention", False)
    use_batch_norm = kwargs.get("use_batch_norm", False)
    weight_decay = kwargs.get("weight_decay", 0.0)
    we_noise = kwargs.get("we_noise", 0.0)
    l2_reg = kwargs.get("l2_reg", 0.0)
    l2_rnn_reg = kwargs.get("l2_rnn_reg", 0.0)
    warmup_steps = kwargs.get("warm_up_steps", 0)


    # Build base model
    model, param_dict = build_rnn_model(we_matrix,
                                        max_sequence_length,
                                        optimizer_name,
                                        n_classes,
                                        expected_steps,
                                        trainable_we=trainable_we,
                                        use_masking=use_masking,
                                        dropout_words=dropout_words,
                                        rnn_layers=2,
                                        dropout_rnn=dropout_rnn,
                                        initial_lr=initial_learing_rate,
                                        lr_scheduler_name=lr_scheduler_name,
                                        rnn_cells=rnn_cells,
                                        bidirectional=bidirectional,
                                        final_layer_size=final_layer_size,
                                        use_attention=use_attention,
                                        use_batch_norm=use_batch_norm,
                                        weight_decay=weight_decay,
                                        we_noise=we_noise,
                                        loss_l2=l2_reg,
                                        rnn_loss_l2=l2_rnn_reg,
                                        warmup_steps=warmup_steps)

    return model, param_dict

# def get_lr_metric(optimizer):
#     def lr(y_true, y_pred):
#         return optimizer.lr
#     return lr

def build_rnn_model(we_matrix, max_sequence_length, optimizer_name, n_classes, expected_num_steps, trainable_we=False, use_masking=True,
                    we_noise=0.0, dropout_words=0.0, rnn_layers=1, unit=LSTM, dropout_rnn=0.0,
                    dropout_rnn_recurrent=0.0, initial_lr=0.0005, lr_scheduler_name=None, weight_decay=0.0, rnn_loss_l2=0., loss_l2=0.,
                    rnn_cells=64, bidirectional=False, final_layer_size=400, dropout_final=0.0, use_attention=False, dropout_attention=0.0,
                    attention_type='additive', attention_activation='none', use_batch_norm=False, warmup_steps=0):
    # print settings of the model
    dict_loc = dict(locals())
    del dict_loc['we_matrix']
    dict_loc['unit'] = str(dict_loc['unit'])

    string_param = build_param_string(dict_loc)
    print(string_param)


    lr = get_learning_rate(initial_lr, lr_scheduler_name, expected_num_steps, warmup_steps=warmup_steps)

    optimizer = get_optimizer(optimizer_name, lr, weight_decay)
    # lr_metric = get_lr_metric(optimizer)

    emb_in = Input(shape=(max_sequence_length,), name='emb-input')
    emb_word = (get_embeddings_layer(we_matrix, max_sequence_length,
                                     trainable_we=trainable_we, use_masking=use_masking))(emb_in)

    if we_noise > 0:
        emb_word = GaussianNoise(we_noise)(emb_word)

    if dropout_words > 0:
        emb_word = Dropout(dropout_words)(emb_word)

    # RNN layers
    for i in range(rnn_layers):
        rs = (rnn_layers > 1 and i < rnn_layers - 1) or use_attention
        emb_word = (get_rnn_layer(unit, rnn_cells, bidirectional,
                                  return_sequences=rs,
                                  recurent_dropout=dropout_rnn_recurrent,
                                  l2_reg=rnn_loss_l2))(emb_word)

        if use_batch_norm:
            emb_word = BatchNormalization()(emb_word)

        if dropout_rnn > 0:
            emb_word = (Dropout(dropout_rnn))(emb_word)

    # Attention after RNN
    if use_attention is True:
        activation = ATTENTIONS_ACTIVATIONS[attention_activation]
        if attention_type == 'additive':
            emb_word = SeqSelfAttention(attention_activation=activation,
                                        name='Attention',
                                        attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD)(emb_word)
        elif attention_type == 'multiplicative':
            emb_word = SeqSelfAttention(attention_activation=activation,
                                        name='Attention',
                                        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(emb_word)

        elif attention_type == 'scaled_dot':
            emb_word = ScaledDotProductAttention(name='Attention')(emb_word)
        else:
            raise NotImplementedError("Unkwonw attention type:" + str(attention_type))

        if dropout_attention > 0:
            emb_word = Dropout(dropout_attention)(emb_word)

        emb_word = Flatten()(emb_word)


    final_layer = emb_word
    if final_layer_size > 0:
        final_layer = Dense(final_layer_size)(emb_word)
        if use_batch_norm is True:
            final_layer = BatchNormalization()(final_layer)

        final_layer = Activation(activation='relu')(final_layer)
        if dropout_final > 0:
            final_layer = Dropout(dropout_final)(final_layer)

    out = Dense(n_classes, activation='softmax',use_bias=False, activity_regularizer=l2(loss_l2))(final_layer)
    model = Model(inputs=[emb_in], outputs=[out])
    model.summary()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model, dict_loc

ATTENTIONS_TYPES = ['additive','multiplicative','scaled_dot']
ATTENTIONS_ACTIVATIONS = {
    'none' : None,
    'sigmoid' : 'sigmoid'
}

def get_rnn_layer(unit, cells=64, bi=False, return_sequences=True, recurent_dropout=0., l2_reg=0.):
    rnn = unit(cells, return_sequences=return_sequences,
               recurrent_dropout=recurent_dropout,
               kernel_regularizer=l2(l2_reg))
    if bi is True:
        return Bidirectional(rnn)
    else:
        return rnn


def get_embeddings_layer(we_matrix, max_len, trainable_we=False, use_masking=True):
    # size of word embeddings, number of words for which embedding vector exist
    vocab_size = we_matrix.shape[0]
    we_dimension = we_matrix.shape[1]

    emb_layer = Embedding(
        input_dim=vocab_size,
        weights=[we_matrix],
        output_dim=we_dimension,
        input_length=max_len,
        trainable=trainable_we,
        mask_zero=use_masking
    )

    return emb_layer


MODELS_GETTER_MAP = {
    'lstm-base': build_model_base
}


def draw_histogram_stats(data_loader, args):
    logger.info("Drawing dataset stats histogram")
    dataset_df = data_loader.load_entire_dataset()

    tokenizer_name = args.tokenizer
    use_stemmer = args.use_stemmer


    use_data_cleaner = args.use_data_cleaner
    data_cleaner = None
    if use_data_cleaner is True:
        data_cleaner = Data_Cleaner(text_processor=text_processor_normalize, lower_case=False,
                                    elong_words=True, elong_punct=True)

    vectorizer = EmbeddingsVectorizer(word_map=None,
                                      tokenizer_name=args.tokenizer,
                                      use_stemmer=args.use_stemmer,
                                      data_cleaner=data_cleaner,
                                      we_matrix=None,
                                      language=None,
                                      max_length=None)

    token_lens = []
    for txt in dataset_df.text:
        tokens = vectorizer.tokenize_to_list(txt)
        token_lens.append(len(tokens))

    max_len = max(token_lens)
    avg_len = np.mean(token_lens)
    cnt = Counter(token_lens)
    logger.info(f"Max tokens len:{max_len}")
    logger.info(f"Avg tokens len:{avg_len}")
    # sort by key
    cnt = sorted(cnt.items())
    print("Sentence len - Counts")

    dataset_name = args.dataset_name
    if dataset_name == 'combined':
        tmp = '-'.join(args.combined_datasets)
        dataset_name = dataset_name + '-' + tmp

    prefix = str(dataset_name) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) \
             + "-tok_" + str(tokenizer_name) + '-'

    histogram_txt_file = os.path.join(LSTM_TMP_HISTOGRAMS, prefix + 'tokens_histogram.txt')

    with open(histogram_txt_file, mode='w', encoding='utf-8') as f:
        f.write("Average len:{:.4f}".format(avg_len) + '\n')
        f.write("Max len:" + str(max_len) + '\n')
        f.write('length - count' + '\n')
        for (length, count) in cnt:
            # print()
            f.write(str(length) + ' - ' + str(count) + '\n')

    tokens_histogram_path = os.path.join(LSTM_TMP_HISTOGRAMS, prefix + 'tokens_histogram.png')
    logger.info(f"Tokens histogram image saved to:{tokens_histogram_path}")
    plt.figure()  # it resets the plot
    figure = sns.distplot(token_lens).get_figure()
    plt.xlim([0, 512])
    plt.xlabel(f"Token count, max len:{max_len} avg len:{avg_len}")
    figure.savefig(tokens_histogram_path, dpi=400)
    plt.figure()
    logger.info("Dataset histogram saved")


def build_and_run(x_train, y_train, x_dev, y_dev, n_classes, args):
    batch_size = args.batch_size
    epochs = args.epoch_count
    train_size = len(x_train)

    X_train, X_dev, _, we_matrix, vectorizer, cached_x_vectors_test_path = extract_word_embeddings(x_train, x_dev, None, args)
    Y_train = to_categorical(y_train, num_classes=n_classes, dtype=int)

    Y_dev = None
    if args.use_only_train_data is False:
        Y_dev = to_categorical(y_dev, num_classes=n_classes, dtype=int)

    expected_num_steps = compute_number_of_steps(train_size, batch_size, epochs)

    dict_args = vars(args)
    model_getter = MODELS_GETTER_MAP[args.model_name]
    model_getter_attrs = (we_matrix, n_classes, expected_num_steps)

    callback_list, tensorboard_log_dir = init_callbacks(dict_args, args.use_early_stopping, args.enable_wandb,
                                                        args.use_only_train_data)

    t0 = time()
    # build model
    model, param_dict = model_getter(*model_getter_attrs, **dict_args)
    model = start_training(model, X_train, Y_train, X_dev, Y_dev,
                           batch_size, epochs, callback_list)

    # evaluate on test data
    f1_dev = accuracy_dev = precision_dev = recall_dev = 0
    if args.use_only_train_data is False:
        f1_dev, accuracy_dev, precision_dev, recall_dev, _, _ = \
            evaluate_baseline_model(model, X_dev, y_dev,
                                    average='macro', grid=False,
                                    categorical_output=True, prob_output=True)

    # time measure
    train_time = time() - t0

    # return predictions and test
    return f1_dev, accuracy_dev, precision_dev, recall_dev,  model, tensorboard_log_dir,\
           train_time, callback_list, vectorizer, param_dict, cached_x_vectors_test_path



def init_callbacks(config, use_early_stop=False, use_wandb=False, use_only_train_data=False):
    file_name = generate_file_name(config)
    tensorboard_log_dir = os.path.join(TENSOR_BOARD_LOGS, file_name)
    os.makedirs(tensorboard_log_dir)
    # tensorBoardCB = keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=0, batch_size=batch_size,
    #                                             write_graph=True, write_grads=True,
    #                                             write_images=False, embeddings_freq=0, embeddings_layer_names=None,
    #                                             embeddings_metadata=None)
    # tensorBoardCB = MyTensorBoardCallback(log_dir=tensorboard_log_dir, batch_size=batch_size)

    callbacks_list = []
    if use_early_stop:
        early_stop = EarlyStopping(monitor='val_loss', patience=0, mode='min', verbose=1,
                                   restore_best_weights=True)
        callbacks_list.append(early_stop)

    if use_wandb:
        # wandb_callback = WandbCallback()
        # wandb_callback = MyWandbCallbackLogger()
        # callbacks_list.append(wandb_callback)
        # callbacks_list.append(WandbCallback(log_gradients=True,
        #                                     log_weights=True))
        try:
            callbacks_list.append(MyOriginalWandbCallback(log_gradients=False,
                                                          log_weights=False,
                                                          save_model=False,
                                                          log_batch_frequency=1,
                                                          use_only_train_data=use_only_train_data))
        except Exception as e:
            logger.error("Error When creating WANDB callback with exception e:" + str(e))


    return callbacks_list, tensorboard_log_dir


def start_training(model, x_train, y_train, x_validate, y_validate, batch_size,
                   epoch_count, callback_list):
    print("Training model...")
    validation = None
    if x_validate is not None and y_validate is not None:
        validation = (x_validate, y_validate)
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epoch_count,
              validation_data=validation,
              shuffle=True,
              callbacks=callback_list,
              verbose=1)
    return model


def extract_sequences(X, vectorizers_dict, max_seq_len):
    # ngram settings
    extracted_X_dict = extract_n_gram_seq(X, vectorizers_dict, max_sentence_len=max_seq_len)

    inputs_ngram_X = []

    # get extracted sequences
    for ngram_order, vectorizer in vectorizers_dict.items():
        inputs_ngram_X.append(extracted_X_dict.get(ngram_order))

    # input_X_train = []
    input_X = []

    input_X.extend(inputs_ngram_X)

    return input_X,


def extract_word_embeddings(X_train, X_dev, X_test, args):
    lang = 'cs'
    max_words = args.max_words
    tokenizer_name = args.tokenizer
    use_stemmer = args.use_stemmer
    use_data_cleaner = args.use_data_cleaner
    max_seq_len = args.max_seq_len

    data_cleaner = None
    if use_data_cleaner is True:
        data_cleaner = Data_Cleaner(text_processor=text_processor_normalize, lower_case=False,
                                    elong_words=True, elong_punct=True)
        for text in demo_sents:
            print(70 * "-")
            print("Original:" + text)
            print("Cleaned :" + data_cleaner.clean_text(text, 0))

        pass

    dataset_name = args.dataset_name
    if dataset_name == 'combined':
        tmp = '-'.join(args.combined_datasets)
        dataset_name = dataset_name + '-' + tmp

    embeddings_lang_dir, cached_embeddings_path, \
    cached_we_matrix_path, cached_wordmap_path, \
    cached_x_vectors_train_path, cached_y_vectors_train_path, \
    cached_x_vectors_dev_path, cached_y_vectors_dev_path, \
    cached_x_vectors_test_path, cached_y_vectors_test_path, \
    cached_part_prefix = get_lang_configs(args.embeddings_file, dataset_name, lang, max_words, tokenizer_name,
                                          use_stemmer, use_data_cleaner, max_seq_len, args.use_only_train_data)

    max_seq_len = args.max_seq_len
    embeddings_size = args.embeddings_size
    embeddings_path = os.path.join(embeddings_lang_dir, args.embeddings_file)

    we_matrix, wordmap = EmbeddingsMatrixManager(embeddings_filename=embeddings_path,
                                                 cached_embeddings_filename=cached_embeddings_path,
                                                 we_embedding_matrix_filename=cached_we_matrix_path,
                                                 wordmap_filename=cached_wordmap_path,
                                                 dimension=embeddings_size,
                                                 max_words=max_words,
                                                 use_gzip=False).get_we_matrix()

    vectorizer = EmbeddingsVectorizer(word_map=wordmap,
                                      tokenizer_name=tokenizer_name,
                                      use_stemmer=use_stemmer,
                                      data_cleaner=data_cleaner,
                                      we_matrix=we_matrix,
                                      language=lang,
                                      max_length=max_seq_len)
    x_vectors_train = None
    if X_train is not None:
        x_vectors_train = vectorizer.vectorize(X_train, cache_file_x=cached_x_vectors_train_path)

    x_vectors_dev = None
    if X_dev is not None:
        x_vectors_dev = vectorizer.vectorize(X_dev, cache_file_x=cached_x_vectors_dev_path)

    x_vectors_test = None
    if X_test is not None:
        x_vectors_test = vectorizer.vectorize(X_test, cache_file_x=cached_x_vectors_test_path)

    vectorizer.print_OOV()
    # vectorizer.print_OOV_words()
    vectorizer.print_stem_words_cnt()
    # vectorizer.print_stem_words()
    return x_vectors_train, x_vectors_dev, x_vectors_test, we_matrix, vectorizer, cached_x_vectors_test_path


