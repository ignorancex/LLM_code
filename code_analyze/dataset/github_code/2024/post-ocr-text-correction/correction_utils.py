from config import ConfigManager
from dataset_parser import DatasetParser
from data_utils import clean_text_all

class DictionaryFeatures:
    def __init__(self, 
                 char_to_id, 
                 max_encoder_sequence_length, 
                 max_decoder_sequence_length, 
                 num_characters):
        self.char_to_id = char_to_id
        self.max_encoder_sequence_length = max_encoder_sequence_length
        self.max_decoder_sequence_length = max_decoder_sequence_length
        self.num_characters = num_characters

class CorrectionUtil:
    def __init__(self, config):
        self.config = config
        self.start_character = '\t'
        self.pad_character = ' '
        self.end_character = '\n'

        self.max_len = 25

        self.parser = DatasetParser(self.config)

    def read_train_correction_data(self):
        chars = set()

        chars.update([self.start_character, self.pad_character, self.end_character])

        input_texts = []
        target_texts = []

        train_words, _, train_gs_words, train_labels = self.parser.read_train_dataset()

        
        for sentence, gs_sentence, label in zip(train_words, train_gs_words, train_labels):
            for w, gs, l in zip(sentence, gs_sentence, label):
                input = clean_text_all(w)
                target = clean_text_all(gs)

                if len(input) > self.max_len or len(target) > self.max_len:
                    continue

                target = self.start_character + target + self.end_character

                input_texts.append(input)
                target_texts.append(target)

                chars = chars.union(set(input)).union(set(target))

        test_words, _, test_gs_words, test_labels = self.parser.read_test_dataset()

        max_encoder_sequence_length = max([len(txt) for txt in input_texts])
        max_decoder_sequence_length = max([len(txt) for txt in target_texts])

        for sentence, gs_sentence, label in zip(test_words, test_gs_words, test_labels):
            for w, gs, l in zip(sentence, gs_sentence, label):
                input = clean_text_all(w)
                target = clean_text_all(gs)

                if len(input) > self.max_len or len(target) > self.max_len:
                  continue

                if len(input) > max_encoder_sequence_length:
                    max_encoder_sequence_length = len(input)

                if len(target) > max_decoder_sequence_length:
                    max_decoder_sequence_length = len(target)
                
                chars = chars.union(set(w)).union(set(gs))

        print(chars)
        print("Number of samples:", len(input_texts))

        characters = sorted(list(chars))
        num_characters = len(characters)

        print("Number of unique tokens:", num_characters)

        print("Max sequence length for inputs:", max_encoder_sequence_length)
        print("Max sequence length for outputs:", max_decoder_sequence_length)

        char_to_id = dict([(char, i) for i, char in enumerate(characters)])

        features = DictionaryFeatures(
            char_to_id=char_to_id,
            max_encoder_sequence_length=max_encoder_sequence_length,
            max_decoder_sequence_length=max_decoder_sequence_length,
            num_characters=num_characters
        )

        return input_texts, target_texts, features

    def read_test_correction_data(self):
        chars = set()

        chars.update([self.start_character, self.pad_character, self.end_character])

        input_texts = []
        target_texts = []

        test_words, _, test_gs_words, test_labels = self.parser.read_test_dataset()

        for sentence, gs_sentence, label in zip(test_words, test_gs_words, test_labels):
            for w, gs, l in zip(sentence, gs_sentence, label):
                input = clean_text_all(w)
                target = clean_text_all(gs)

                if len(input) > self.max_len or len(target) > self.max_len:
                  continue

                target = self.start_character + target + self.end_character

                input_texts.append(input)
                target_texts.append(target)

                chars = chars.union(set(w)).union(set(gs))

        print(chars)
        print("Number of samples:", len(input_texts))

        characters = sorted(list(chars))
        num_characters = len(characters)

        max_encoder_sequence_length = max([len(txt) for txt in input_texts])
        max_decoder_sequence_length = max([len(txt) for txt in target_texts])

        print("Number of unique tokens:", num_characters)

        print("Max sequence length for inputs:", max_encoder_sequence_length)
        print("Max sequence length for outputs:", max_decoder_sequence_length)

        char_to_id = dict([(char, i) for i, char in enumerate(characters)])

        features = DictionaryFeatures(
            char_to_id=char_to_id,
            max_encoder_sequence_length=max_encoder_sequence_length,
            max_decoder_sequence_length=max_decoder_sequence_length,
            num_characters=num_characters
        )

        return input_texts, target_texts, features

def main():
    config = ConfigManager()
    util = CorrectionUtil(config)
    x, y, features = util.read_train_correction_data()

    print(x[4])
    print(y[4])

    print(features.char_to_id)

if __name__ == '__main__':
    main()