import torch
import glob
import nltk
import numpy as np
from config import ConfigManager

from data_utils import tokenize, truncate_and_pad, read_data, filter_data, read_synthetic_data

class DatasetParser:
    def __init__(self, config):
        self.config = config

        nltk.download('punkt')
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        self.max_norm_lev_distance = 0.5

    def read_train_dataset(self):
        train_data_files = glob.glob(self.config.get_train_dataset() + '/*.txt') # Get all files from the configured directory
        sentence_raw, aligned_sentences_raw, gs_sentence_raw, sentence_labels_raw = read_data(train_data_files, self.sentence_tokenizer)

        words, aligned_words, gs_words, labels = filter_data(sentence_raw, aligned_sentences_raw, gs_sentence_raw, sentence_labels_raw, self.max_norm_lev_distance)

        if self.config.get_enable_synthetic():
            s_words, s_aligned_words, s_gs_words, s_labels = self.read_synthetic_dataset()

            words = words + s_words
            aligned_words = aligned_words + s_aligned_words
            gs_words = gs_words + s_gs_words
            labels = labels + s_labels

        return words, aligned_words, gs_words, labels

    def read_synthetic_dataset(self):
        files = glob.glob(self.config.get_synthetic_dataset() + '/*.txt') # Get all files from the configured directory

        sentence_raw, aligned_sentences_raw, gs_sentence_raw, sentence_labels_raw = read_synthetic_data(files)
        words, aligned_sentences, gs_sentences, labels = filter_data(sentence_raw, aligned_sentences_raw, gs_sentence_raw, sentence_labels_raw, self.max_norm_lev_distance)

        return words, aligned_sentences, gs_sentences, labels

    def read_test_dataset(self):
        test_files = glob.glob(self.config.get_test_dataset() + '/*.txt')
        sentence_raw, aligned_sentences_raw, gs_sentence_raw, sentence_labels_raw = read_data(test_files, self.sentence_tokenizer)

        test_words, test_aligned_words, test_gs_words, test_labels = filter_data(sentence_raw, aligned_sentences_raw, gs_sentence_raw, sentence_labels_raw, self.max_norm_lev_distance)

        return test_words, test_aligned_words, test_gs_words, test_labels

    def tokenize_dataset(self, words, labels, max_sequence_length, tokenizer):
        tokenized_texts_and_labels = [tokenize(sentence_words, sentence_labels, tokenizer) for sentence_words, sentence_labels in zip(words, labels)]

        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        tokenized_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        input_ids = np.array([truncate_and_pad(tokenizer.convert_tokens_to_ids(txt), max_sequence_length, tokenizer) for txt in tokenized_texts], dtype='long')
        fixed_labels = np.array([truncate_and_pad(sentence_labels, max_sequence_length, tokenizer) for sentence_labels in tokenized_labels], dtype='long')
        attention_masks = [[int(x != tokenizer.pad_token_id) for x in y] for y in input_ids]

        inputs = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(fixed_labels, dtype=torch.long)
        masks = torch.tensor(attention_masks, dtype=torch.long)

        return inputs, labels, masks, tokenized_texts

def main():
    config = ConfigManager()
    parser = DatasetParser(config)
    x, y, z, t = parser.read_train_dataset()

    print(x[27])
    print(z[27])

if __name__ == '__main__':
    main()