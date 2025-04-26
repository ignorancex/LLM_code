import os

from pathlib import Path

from config import EMB_DIR, LSTM_TMP_DIR, TMP_SPLIT_DIR


OPTIMIZERS_CHOICES = ['Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adadelta', 'SGD']

# nltk.download('punkt', quiet=True)
#
TOKENIZER_CHOICES = ['white_space', 'toktok', 'word_tokenizer']

# possible learning rate schedulers
LR_SCHEDULER_CHOICES = ['none','exp','poly','cosine']


def get_lang_configs(embeddings_file, dataset_name, lang, max_words, tokenizer_name, use_stemmer, use_data_cleaner,
                     max_seq_len, use_only_train_data):
    dataset_conf = Data_Config(embeddings_file, dataset_name + str('-train'), dataset_name + str('-dev'),
                               dataset_name + str('-test'), max_words, max_seq_len, tokenizer_name, use_stemmer, use_data_cleaner,
                               lang=lang, use_only_train_data=use_only_train_data)

    embeddings_lang_dir = dataset_conf.embeddings_lang_dir
    cached_embeddings_path = dataset_conf.cached_embeddings_path
    cached_we_matrix_path = dataset_conf.cached_we_matrix_path
    cached_wordmap_path = dataset_conf.cached_wordmap_path

    cached_x_vectors_train_path = dataset_conf.cached_x_vectors_train_path
    cached_y_vectors_train_path = dataset_conf.cached_y_vectors_train_path
    cached_x_vectors_dev_path = dataset_conf.cached_x_vectors_dev_path
    cached_y_vectors_dev_path = dataset_conf.cached_y_vectors_dev_path
    cached_x_vectors_test_path = dataset_conf.cached_x_vectors_test_path
    cached_y_vectors_test_path = dataset_conf.cached_y_vectors_test_path
    cached_part_prefix = dataset_conf.cached_part_prefix

    return embeddings_lang_dir, cached_embeddings_path, \
           cached_we_matrix_path, cached_wordmap_path, \
           cached_x_vectors_train_path, cached_y_vectors_train_path, \
           cached_x_vectors_dev_path, cached_y_vectors_dev_path, \
           cached_x_vectors_test_path, cached_y_vectors_test_path, \
           cached_part_prefix


class Data_Config(object):
    def __init__(self, embeddings_file, train_combination, dev_combination, test_combination,
                 max_words, max_seq_len, tokenizer_name, use_stemmer, use_data_cleaner, lang='cs',
                 use_only_train_data=False):
        self.lang = lang
        embeddings_file = embeddings_file.replace('/', '-')
        self.embeddings_lang_dir = os.path.join(EMB_DIR, lang)
        Path(self.embeddings_lang_dir).mkdir(parents=True, exist_ok=True)

        self.cached_part_prefix = os.path.join(TMP_SPLIT_DIR,
                                               embeddings_file + '-' + train_combination + '-' + lang + '_TRAIN-' +str(use_only_train_data) +  '-max_seq_len-' + str(max_seq_len) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) + "-max_w-" + str(max_words))

        self.cached_embeddings_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + lang + "-max_w-" + str(max_words) + '-emb.bin')
        self.cached_we_matrix_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + lang + "-max_w-" + str(max_words) + '-we_matrix.bin')
        self.cached_wordmap_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + lang + "-max_w-" + str(max_words) + '-wordmap.bin')
        self.cached_x_vectors_train_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + train_combination + '-' + lang + '_TRAIN-' +str(use_only_train_data) + '-max_seq_len-' + str(max_seq_len) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) + "-max_w-" + str(max_words) + "-tok_" + str(tokenizer_name) + '-cached_x.bin')
        self.cached_y_vectors_train_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + train_combination + '-' + lang + '_TRAIN-' +str(use_only_train_data) + '-max_seq_len-' + str(max_seq_len) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) + "-max_w-" + str(max_words) + "-tok_" + str(tokenizer_name) + '-cached_y.bin')
        self.cached_x_vectors_dev_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + dev_combination + '-' + lang + '_TRAIN-' +str(use_only_train_data) + '-max_seq_len-' + str(max_seq_len) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) + "-max_w-" + str(max_words) + "-tok_" + str(tokenizer_name) + '-cached_x.bin')
        self.cached_y_vectors_dev_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + dev_combination + '-' + lang + '_TRAIN-' +str(use_only_train_data) + '-max_seq_len-' + str(max_seq_len) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) + "-max_w-" + str(max_words) + "-tok_" + str(tokenizer_name) + '-cached_y.bin')
        self.cached_x_vectors_test_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + test_combination + '-' + lang + '_TRAIN-' +str(use_only_train_data) + '-max_seq_len-' + str(max_seq_len) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) + "-max_w-" + str(max_words) + "-tok_" + str(tokenizer_name) + '-cached_test_x.bin')
        self.cached_y_vectors_test_path = os.path.join(LSTM_TMP_DIR, embeddings_file + '-' + test_combination + '-' + lang + '_TRAIN-' +str(use_only_train_data) + '-max_seq_len-' + str(max_seq_len) + '-use_clean-' + str(use_data_cleaner) + '-stem-' + str(use_stemmer) + "-max_w-" + str(max_words) + "-tok_" + str(tokenizer_name) + '-cached_text_y.bin')


    @property
    def lang(self):
        return self._lang

    @lang.setter
    def lang(self, value):
        self._lang = value

    @property
    def embeddings_lang_dir(self):
        return self._embeddings_lang_dir

    @embeddings_lang_dir.setter
    def embeddings_lang_dir(self, value):
        self._embeddings_lang_dir = value

    @property
    def cached_part_prefix(self):
        return self._cached_part_prefix

    @cached_part_prefix.setter
    def cached_part_prefix(self, value):
        self._cached_part_prefix = value

    @property
    def cached_embeddings_path(self):
        return self._cached_embeddings_path

    @cached_embeddings_path.setter
    def cached_embeddings_path(self, value):
        self._cached_embeddings_path = value

    @property
    def cached_we_matrix_path(self):
        return self._cached_we_matrix_path

    @cached_we_matrix_path.setter
    def cached_we_matrix_path(self, value):
        self._cached_we_matrix_path = value

    @property
    def cached_wordmap_path(self):
        return self._cached_wordmap_path

    @cached_wordmap_path.setter
    def cached_wordmap_path(self, value):
        self._cached_wordmap_path = value

    @property
    def cached_x_vectors_train_path(self):
        return self._cached_x_vectors_train_path

    @property
    def cached_y_vectors_train_path(self):
        return self._cached_y_vectors_train_path

    @property
    def cached_x_vectors_dev_path(self):
        return self._cached_x_vectors_dev_path

    @property
    def cached_y_vectors_dev_path(self):
        return self._cached_y_vectors_dev_path

    @property
    def cached_x_vectors_test_path(self):
        return self._cached_x_vectors_test_path

    @property
    def cached_y_vectors_test_path(self):
        return self._cached_y_vectors_test_path

    @cached_x_vectors_train_path.setter
    def cached_x_vectors_train_path(self, value):
        self._cached_x_vectors_train_path = value

    @cached_y_vectors_train_path.setter
    def cached_y_vectors_train_path(self, value):
        self._cached_y_vectors_train_path = value

    @cached_x_vectors_dev_path.setter
    def cached_x_vectors_dev_path(self, value):
        self._cached_x_vectors_dev_path = value

    @cached_y_vectors_dev_path.setter
    def cached_y_vectors_dev_path(self, value):
        self._cached_y_vectors_dev_path = value

    @cached_x_vectors_test_path.setter
    def cached_x_vectors_test_path(self, value):
        self._cached_x_vectors_test_path = value

    @cached_y_vectors_test_path.setter
    def cached_y_vectors_test_path(self, value):
        self._cached_y_vectors_test_path = value