# class based on https://github.com/cbaziotis/datastories-semeval2017-task6
import os

import numpy as np
from nltk.tokenize import WhitespaceTokenizer
# from nltk.tokenize.regexp import WhitespaceTokenizer

from src.polarity.lstm_baseline.nn_utils import save_data_pickle, load_data_pickle, get_tokenizer


class EmbeddingsVectorizer:
    def __init__(self,
                 word_map,
                 we_matrix,
                 language,
                 max_length,
                 use_stemmer,
                 normalized_tokens=None,
                 tokenizer_name=None,
                 data_cleaner=None,
                 unk_policy="random"):

        """

        :param word_map:
        :param we_matrix:
        :param language:
        :param max_length:
        :param normalized_tokens: list of tokens <money>, that will be unmasked i.e. "<user>" => "user"
        :param unk_policy: "random","zero"
        """

        # indices to we_matrix
        self.word_map = word_map

        # we matrix
        self.we_matrix = we_matrix

        self.language = language
        if tokenizer_name is None:
            self.tokenizer = WhitespaceTokenizer()
        else:
            self.tokenizer = get_tokenizer(tokenizer_name)

        self.use_stemmer = use_stemmer
        self.succes_stem_words = dict()
        self.succes_stem_cnt = 0

        self.data_cleaner = data_cleaner

        # max sequence length
        self.max_seq_length = max_length

        self.normalized_tokens = normalized_tokens

        self.unk_policy = unk_policy

        self.OOV = 0

        self.total_words = 0

        self.OOV_words = dict()

    def tokenize_to_list(self, text):
        words = [x for x in self.tokenizer.tokenize(text) if len(x) >= 1]
        return words

    # prevede text na sekvenci čísel (odpovídajících slovům)
    def text_to_sequence(self, word_list, add_tokens=True):
        max_len = self.max_seq_length

        words = np.zeros(max_len).astype(int)
        # trim tokens after max length
        sequence = word_list[:max_len]

        if add_tokens:
            index = self.word_map.get('<s>', -1)
            if index != -1:
                words[0] = index
                start_token_added = True
            else:
                # bcs we didnt added word
                start_token_added = False

        for i, token in enumerate(sequence):
            index = i
            if add_tokens and start_token_added:
                index = i + 1

            if index >= max_len:
                # todo mozna dodelat protoze pokud je ten text delsi nez max len tak se tam neprida ukoncovaci znackaa
                break

            self.total_words += 1

            # unmask tokens
            if self.normalized_tokens is not None:
                token = self.unmask_token(token)

            # whether the word is unknown
            unknown_flag = False

            if token in self.word_map:
                words[index] = self.word_map[token]
            else:
                if ',žebranim' in token:
                    print("")
                    pass

                tmp = self.remove_inter(token)
                lower = tmp.lower()
                if tmp in self.word_map:
                    # if start with comma or ends with dot
                    words[index] = self.word_map[tmp]
                elif lower in self.word_map:
                    words[index] = self.word_map[lower]
                elif self.use_stemmer is True:
                    stem_word_ref, stemmed_word = self.try_find(tmp)
                    if stem_word_ref is not None:
                        words[index] = stem_word_ref
                        stem_string = "Stemmed word:" + str(stemmed_word) + " from original:" + tmp
                        # print(stem_string)
                        self.succes_stem_cnt += 1
                        tmp_cnt, original_words_list = self.succes_stem_words.get(stemmed_word, (0, list()))
                        original_words_list.append(tmp)
                        tmp_cnt += 1
                        self.succes_stem_words[stemmed_word] = (tmp_cnt, original_words_list)
                    else:
                        unknown_flag = True
                else:
                    unknown_flag = True

                # if we did not found the word
                if unknown_flag is True:
                    if self.unk_policy == 'random':
                        words[index] = self.word_map["<unk>"]
                        # print(token)
                        self.OOV_words[token] = self.OOV_words.get(token, 0) + 1
                        self.OOV += 1
                    elif self.unk_policy == 'zero':
                        words[index] = 0
                        # print(token)
                        self.OOV_words[token] = self.OOV_words.get(token, 0) + 1
                        self.OOV += 1

        if add_tokens:
            index = min(index + 1, max_len - 1)
            words[index] = self.word_map.get('</s>', 0)

        return words

    def unmask_token(self, token):
        for tmp in self.normalized_tokens:
            if tmp == token:
                token = token[1:-1]
                break
        return token

    # TODO pridat OTAZNIK,
    # TODO ODSTRANIT DIAKRITIU
    def remove_inter(self, token):
        token = token.strip()
        if len(token) > 1:
            if token.endswith('.'):
                token = token[:-1]

            if token.startswith('‚') or token.startswith(',') or token.startswith('-') or token.startswith('¨') \
                    or token.startswith('+'):
                token = token[1:]
            # nakej jinej znak pro carku
            if token.endswith('‚') or token.endswith(',') or token.endswith('-') or token.endswith('¨') \
                    or token.endswith('+'):
                token = token[:-1]
        return token

    def print_OOV(self):
        if self.total_words != 0:
            print('OOV words:  ', self.OOV)
            print('Total words:', self.total_words)
            print('Ratio:      ', (self.OOV / self.total_words))

    def print_OOV_words(self):
        sorted_dict = dict(sorted(self.OOV_words.items(), key=lambda item: item[1], reverse=True))
        for key, value in sorted_dict.items():
            print(key, ' - ', value)

    def print_stem_words(self):
        sorted_dict = dict(sorted(self.succes_stem_words.items(), key=lambda item: (item[1])[0], reverse=True))
        if self.use_stemmer:
            for key, value in sorted_dict.items():
                print(key, ' - ', value)

    def print_stem_words_cnt(self):
        if self.use_stemmer:
            print("Number of stemmed words:" + str(self.succes_stem_cnt))

    def vectorize(self, x_texts, cache_file_x=None):

        if cache_file_x is not None:
            if os.path.exists(cache_file_x):
                x_vectors = load_data_pickle(cache_file_x)
                return x_vectors

        x_vectors = np.zeros(shape=(len(x_texts), self.max_seq_length)).astype(int)

        for i, text in enumerate(x_texts):
            # original_text = text
            if self.data_cleaner is not None:
                text = self.data_cleaner.clean_text(text, 0)

            word_list = self.tokenize_to_list(text)
            x_vector = self.text_to_sequence(word_list)

            x_vectors[i] = x_vector

        if cache_file_x is not None:
            save_data_pickle(x_vectors, cache_file_x)

        return x_vectors

    # Stemming
    # Based on https://github.com/UFAL-DSG/alex/blob/master/alex/utils/czech_stemmer.py

    def try_find(self, word):
        word = word.lower()
        word_ref = None
        if word in self.word_map:
            word_ref = self.word_map[word]
            return word_ref, word

        word = _remove_case(word)
        if word in self.word_map:
            word_ref = self.word_map[word]
            return word_ref, word

        word = _remove_possessives(word)
        if word in self.word_map:
            word_ref = self.word_map[word]
            return word_ref, word

        word = _remove_comparative(word)
        if word in self.word_map:
            word_ref = self.word_map[word]
            return word_ref, word

        word = _remove_diminutive(word)
        if word in self.word_map:
            word_ref = self.word_map[word]
            return word_ref, word

        word = _remove_augmentative(word)
        if word in self.word_map:
            word_ref = self.word_map[word]
            return word_ref, word

        word = _remove_derivational(word)
        if word in self.word_map:
            word_ref = self.word_map[word]
            return word_ref, word

        return word_ref, word


def _remove_augmentative(word):
    if len(word) > 6 and word.endswith(u"ajzn"):
        return word[:-4]
    if len(word) > 5 and word[-3:] in {u"izn", u"isk"}:
        return _palatalise(word[:-2])
    if len(word) > 4 and word.endswith(u"ák"):
        return word[:-2]
    return word


def _remove_derivational(word):
    if len(word) > 8 and word.endswith(u"obinec"):
        return word[:-6]
    if len(word) > 7:
        if word.endswith(u"ionář"):
            return _palatalise(word[:-4])
        if word[-5:] in {u"ovisk", u"ovstv", u"ovišt", u"ovník"}:
            return word[:-5]
    if len(word) > 6:
        if word[-4:] in {u"ásek", u"loun", u"nost", u"teln", u"ovec", u"ovík",
                         u"ovtv", u"ovin", u"štin"}:
            return word[:-4]
        if word[-4:] in {u"enic", u"inec", u"itel"}:
            return _palatalise(word[:-3])
    if len(word) > 5:
        if word.endswith(u"árn"):
            return word[:-3]
        if word[-3:] in {u"ěnk", u"ián", u"ist", u"isk", u"išt", u"itb", u"írn"}:
            return _palatalise(word[:-2])
        if word[-3:] in {u"och", u"ost", u"ovn", u"oun", u"out", u"ouš",
                         u"ušk", u"kyn", u"čan", u"kář", u"néř", u"ník",
                         u"ctv", u"stv"}:
            return word[:-3]
    if len(word) > 4:
        if word[-2:] in {u"áč", u"ač", u"án", u"an", u"ář", u"as"}:
            return word[:-2]
        if word[-2:] in {u"ec", u"en", u"ěn", u"éř", u"íř", u"ic", u"in", u"ín",
                         u"it", u"iv"}:
            return _palatalise(word[:-1])
        if word[-2:] in {u"ob", u"ot", u"ov", u"oň", u"ul", u"yn", u"čk", u"čn",
                         u"dl", u"nk", u"tv", u"tk", u"vk"}:
            return word[:-2]
    if len(word) > 3 and word[-1] in u"cčklnt":
        return word[:-1]
    return word


def _remove_diminutive(word):
    if len(word) > 7 and word.endswith(u"oušek"):
        return word[:-5]
    if len(word) > 6:
        if word[-4:] in {u"eček", u"éček", u"iček", u"íček", u"enek", u"ének",
                         u"inek", u"ínek"}:
            return _palatalise(word[:-3])
        if word[-4:] in {u"áček", u"aček", u"oček", u"uček", u"anek", u"onek",
                         u"unek", u"ánek"}:
            return _palatalise(word[:-4])
    if len(word) > 5:
        if word[-3:] in {u"ečk", u"éčk", u"ičk", u"íčk", u"enk", u"énk",
                         u"ink", u"ínk"}:
            return _palatalise(word[:-3])
        if word[-3:] in {u"áčk", u"ačk", u"očk", u"učk", u"ank", u"onk",
                         u"unk", u"átk", u"ánk", u"ušk"}:
            return word[:-3]
    if len(word) > 4:
        if word[-2:] in {u"ek", u"ék", u"ík", u"ik"}:
            return _palatalise(word[:-1])
        if word[-2:] in {u"ák", u"ak", u"ok", u"uk"}:
            return word[:-1]
    if len(word) > 3 and word[-1] == u"k":
        return word[:-1]
    return word


def _remove_comparative(word):
    if len(word) > 5:
        if word[-3:] in {u"ejš", u"ějš"}:
            return _palatalise(word[:-2])
    return word


def _remove_possessives(word):
    if len(word) > 5:
        if word[-2:] in {u"ov", u"ův"}:
            return word[:-2]
        if word.endswith(u"in"):
            return _palatalise(word[:-1])
    return word


def _remove_case(word):
    if len(word) > 7 and word.endswith(u"atech"):
        return word[:-5]
    if len(word) > 6:
        if word.endswith(u"ětem"):
            return _palatalise(word[:-3])
        if word.endswith(u"atům"):
            return word[:-4]
    if len(word) > 5:
        if word[-3:] in {u"ech", u"ich", u"ích", u"ého", u"ěmi", u"emi", u"ému",
                         u"ete", u"eti", u"iho", u"ího", u"ími", u"imu"}:
            return _palatalise(word[:-2])
        if word[-3:] in {u"ách", u"ata", u"aty", u"ých", u"ama", u"ami",
                         u"ové", u"ovi", u"ými"}:
            return word[:-3]
    if len(word) > 4:
        if word.endswith(u"em"):
            return _palatalise(word[:-1])
        if word[-2:] in {u"es", u"ém", u"ím"}:
            return _palatalise(word[:-2])
        if word[-2:] in {u"ům", u"at", u"ám", u"os", u"us", u"ým", u"mi", u"ou"}:
            return word[:-2]
    if len(word) > 3:
        if word[-1] in u"eiíě":
            return _palatalise(word)
        if word[-1] in u"uyůaoáéý":
            return word[:-1]
    return word


def _palatalise(word):
    if word[-2:] in {u"ci", u"ce", u"či", u"če"}:
        return word[:-2] + u"k"

    if word[-2:] in {u"zi", u"ze", u"ži", u"že"}:
        return word[:-2] + u"h"

    if word[-3:] in {u"čtě", u"čti", u"čtí"}:
        return word[:-3] + u"ck"

    if word[-3:] in {u"ště", u"šti", u"ští"}:
        return word[:-3] + u"sk"
    return word[:-1]
