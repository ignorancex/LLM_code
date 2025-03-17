import configparser
import os
import sys 
import inspect
import re

import classla

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import ConfigManager
    
class Converter:

    def __init__(self, config):
        self.config = config
        
        # classla.download('bg') uncomment for first time run
        self.nlp_tagger = classla.Pipeline('bg', processors='tokenize,pos', dir=self.config.get_stanza_dir())

        self.cons = ["б", "в", "г", "д", "ж", "з", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ"]
        self.vowels = ["а", "ъ", "о", "у", "е", "и"]

        self.verbs_ending_on_ee_or_eq = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/verbs_ending_on_ee_or_eq.txt')

        self.adverbs_ending_on_e = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/adverbs_ending_on_e.txt')

        self.soft_ending_masculine_nouns = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/masculine-nouns-soft-ending.txt')

        self.soft_feminine_ending_words = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/feminine-nouns-ending-on-con.txt')

        self.numerical_form_suffix = "десет"

        self.us_roots = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/us-roots.txt')

        self.us_special_words = ['винаги', 'веднъж']

        self.special_feminine_adjectives = ['бледосин', 'възсин', 'лазурносин', 'морскосин', 'небесносин', 'пастелносин', 'светлосин', 'сивосин']

        self.yat_roots = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/yat-roots.txt')

        self.verb_nouns = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/verb-nouns.txt')

        self.verbs_first_conjugation = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/verbs-first-conjugation.txt')

        self.verbs_second_conjugation = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/verbs-second-conjugation.txt')

        self.verbs_past_tense_3rd_person = self.read_file_into_list(self.config.get_converter_drinovski_resources_dir() + '/verbs-past-tense-3rd-person.txt')

        self.abbreviations = ["лв","др","абв","т","н","пр","сащ","проф","гр","сл","ссср","хр","доц","бкп","сдс","дпс","мн","ч","р",
            "ж","ср","св","стр","подкл","кл","чл","кр","мвр","бан","момн","мон","аец", "заб","бул","ул","вм","вж","пл","церн","бзнс","бнб"]

        self.exclusion_pronouns = {
            "нея": "неѭ",
            "я": "ѭ"
        }

        self.c_len = len(self.cons)
        self.us_len = len(self.us_roots)
        self.yat_len = len(self.yat_roots)

    def read_into_list_split(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        d = []
        for line in lines:
            line = line.rstrip('\n')

            d.extend(line.split())


        return sorted(list(set(d)))
    
    def read_file_into_list(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        output = []
        for w in lines:
            w = w.strip('\n  —')

            if len(w) < 1:
                continue
            
            output.extend(w.split())

        return output

    def split_into_words(self, text):
        words = [x for x in re.split('[ *.+—:;,?\r\t\n{}]+', text) if x != '']
        wLen = len(words)

        return words, wLen

    def save_me(self, file, data_to_save):
        f1 = open(file, "w", encoding="utf-8")

        for entry in data_to_save:
            f1.write(entry + '\n')

        f1.close()

    def convert_word(self, word_to_convert, upos=None, feats=None):
        w1, _ = self.convert_verb_nouns(word_to_convert, word_to_convert)
        w2, _ = self.convert_ending(word_to_convert, w1)
        w3, _ = self.apply_us(word_to_convert, w2, upos, feats)
        w4, _ = self.convert_the(word_to_convert, w3, upos, feats)
        w5, _ = self.apply_yat(word_to_convert, w4, upos, feats)

        return w5

    def clean_up(self, word):
        word = re.sub(r'\n„', '', word)
        word = re.sub(r'\s+', ' ', word)
        word = re.sub(r'[„“\[\])()]', '', word)
        word = word.lower()
        return word.strip(' ,.?!…')

    def convert_text(self, text):
        if len(text) < 1:
            return []

        words, wLen = self.split_into_words(text)
        text = ' '.join(words)
        morph_tagging = self.nlp_tagger(text).to_dict()
        
        if len(morph_tagging) > 0 and len(morph_tagging[0]) > 0:
            morph_tagging_res = morph_tagging[0][0]
        else:
            return []

        upos_list = []
        feats_list = []

        for r in morph_tagging_res:
            if r['upos'] == 'PUNCT' or r['upos'] == 'X':
                continue

            upos_list.append(r['upos'])
            feats_list.append(r['feats'].split('|') if 'feats' in r else [])

        try:
            assert len(upos_list) == len(feats_list)
            assert wLen == len(feats_list)
        except AssertionError:
            print(words)
            print(upos_list)
            print(len(feats_list))
            print(len(words))

            return []

        converted_words = []
        
        for word, upos, feats in zip(words, upos_list, feats_list):
            word = self.clean_up(word)
            if len(word) > 0:
                converted_words.append(self.convert_word(word, upos, feats))

        return converted_words

    # Отглаголни съществителни - Правило 1
    def convert_verb_nouns(self, word_to_convert, current_word):

        if self._includes(self.verb_nouns, word_to_convert):
            return current_word[:len(current_word) - 1] + 'ье', True 

        return current_word, False


    # Краесловни ерове (хитрост, скромност ж.р.) - Правило 2
    def convert_ending(self, word_to_convert, current_word, upos=None, feats=None):
        n = len(word_to_convert)
        last_character = word_to_convert[n - 1]

        if upos == None or feats == None:
            morph_tagging_res = self.nlp_tagger(word_to_convert).to_dict()[0][0][0]
            upos = morph_tagging_res['upos']
            feats = morph_tagging_res['feats'].split('|') if 'feats' in morph_tagging_res else []

        if self._includes(self.cons, last_character):
            if self._includes(self.abbreviations, word_to_convert) or ((word_to_convert[n - 2:] == "ят" or word_to_convert[n - 2:] == "ът") and not self._is_verb(upos)):
                return current_word, False
            
            if self._includes(self.soft_ending_masculine_nouns, word_to_convert) or self._includes(self.soft_feminine_ending_words, word_to_convert)\
                or self._includes(self.special_feminine_adjectives, word_to_convert) or word_to_convert.endswith(self.numerical_form_suffix):
                current_word += 'ь'
            else:
                current_word += "ъ"
            
            return current_word, True

        return current_word, False
    
    # Голям юс (Правило 3)
    def apply_us(self, word_to_convert, current_word, upos=None, feats=None):
        n = len(word_to_convert)
        last_characters = word_to_convert[n - 2:]

        if upos == None or feats == None:
            morph_tagging_res = self.nlp_tagger(word_to_convert).to_dict()[0][0][0]
            upos = morph_tagging_res['upos']
            feats = morph_tagging_res['feats'].split('|') if 'feats' in morph_tagging_res else []

        if self._includes(self.exclusion_pronouns, word_to_convert):
            return self.exclusion_pronouns[word_to_convert], True

        if (self._includes(self.verbs_first_conjugation, word_to_convert) or self._includes(self.verbs_second_conjugation, word_to_convert))\
                and not (last_characters == 'ея' or last_characters == 'ее'):
            if word_to_convert[-2:1] in self.vowels or word_to_convert[-1:] == 'я' or word_to_convert[n - 2:] == 'ят':
                current_word = current_word[::-1].replace('я', 'ѭ', 1)[::-1]
            else:
                current_word = current_word[::-1].replace('а', 'ѫ', 1)[::-1]

            return current_word,  True

        for k in range(len(self.us_special_words)):
            if word_to_convert.find(self.us_special_words[k]) != -1:
                first_y = current_word.find('ъ')
                first_a = current_word.find('а')
                
                replace = 'ъ' if first_y > first_a else 'а'
                current_word = current_word[::-1].replace(replace, 'ѫ', 1)[::-1]

                return current_word, True

        for k in range(self.us_len):
            us_index = word_to_convert.find(self.us_roots[k])

            if us_index != -1:
                current_word = current_word[0:us_index] + current_word[us_index:].replace("ъ","ѫ", 1)
            
                return current_word, True
        
        return self._handle_verb_us(word_to_convert, current_word, feats) if self._is_verb else (current_word, False)

    # Членни форми (Правило 4)
    def convert_the(self, word_to_convert, current_word, upos=None, feats=None):
        n = len(current_word)
        if upos == None or feats == None:
            morph_tagging_res = self.nlp_tagger(word_to_convert).to_dict()[0][0][0]
            upos = morph_tagging_res['upos']
            feats = morph_tagging_res['feats'].split('|') if 'feats' in morph_tagging_res else []
        
        did_convert = False
        last_characters = word_to_convert[n - 2:]

        if self._is_adjective(upos) and last_characters == 'ят':
            current_word, _ = self.convert_ending(word_to_convert[:n - 2], current_word[:n - 2])

            current_word = current_word[:n - 3] + 'ий'
            
            did_convert = True
        elif self._is_noun(upos) and self._is_masculine(feats) and (last_characters == "ят" or last_characters == "ът"):
            current_word, _ = self.convert_ending(word_to_convert[:n - 2], current_word[:n - 2])
            current_word = current_word + 'тъ'

            did_convert = True
        elif self._is_noun(upos) and last_characters == 'та' and self._is_feminine(feats):
            current_word, c_ending = self.convert_ending(word_to_convert[:n - 2], current_word[:n - 2])

            current_word = current_word + 'та'
            did_convert = did_convert or c_ending
        elif self._is_noun(upos) and self._is_plural(feats):
            current_word = current_word[:n - 2] + 'тѣ'
            did_convert = True

        return current_word, did_convert

    # Ят (Правило 5)
    def apply_yat(self, word_to_convert, current_word, upos=None, feats=None):
        if upos == None or feats == None:
            morph_tagging_res = self.nlp_tagger(word_to_convert).to_dict()[0][0][0]
            upos = morph_tagging_res['upos']
            feats = morph_tagging_res['feats'].split('|') if 'feats' in morph_tagging_res else []

        n = len(word_to_convert)
        last_characters = word_to_convert[n - 2:]
        did_convert = False

        if word_to_convert[0:3] == "пре":
            current_word = current_word.replace("е","ѣ", 1)
            did_convert = True

        if self._includes(self.adverbs_ending_on_e, word_to_convert):
            return current_word[:n - 1] + 'ѣ', True

        if word_to_convert == "де":
            return current_word[0] + "ѣ", True

        for k in range(self.yat_len):
            yat_index = word_to_convert.find(self.yat_roots[k])

            if yat_index != -1:
                if k < 209:
                    current_word = current_word[0:yat_index] + current_word[yat_index:].replace("е","ѣ", 1)
                else:
                    current_word = current_word[0:yat_index] + current_word[yat_index:].replace("я","ѣ", 1)

                return current_word, True

        if self._is_verb(upos):
            r, f = self._handle_verb_yat(word_to_convert, current_word, last_characters, feats)

            return r, (f or did_convert)

        return current_word, did_convert

    def _handle_verb_yat(self, word_to_convert, current_word, last_characters, feats):
        if (last_characters == 'ея' or last_characters == 'ее') or self._includes(self.verbs_ending_on_ee_or_eq, word_to_convert):
            return (current_word.replace('ее', 'ѣе', 1) if last_characters == 'ее' else current_word.replace('ея', 'ѣѭ', 1)), True
        
        if self._is_past_tense(feats) or self._is_imperfect_tense(feats) or self._is_imperative_mood(feats):
            first_ya = current_word.find('я')
            first_e = current_word.find('е')
            has_special_prefix_index = current_word.find('без') # представка съдържаща e (само тази е)

            replace = 'е' if first_ya == -1 else ('я' if first_e == -1 else ('я' if first_ya < first_e else 'е'))

            current_word = current_word.replace(replace, "ѣ", 1) if has_special_prefix_index == -1 else current_word[0:3] + current_word[3:].replace(replace, "ѣ", 1)

            return current_word, True

        return current_word, False

    def _handle_verb_us(self, word_to_convert, current_word, feats):
        if (self._is_plural(feats) and self._is_past_tense(feats) and self._is_3rd_person(feats))\
                or self._includes(self.verbs_past_tense_3rd_person, word_to_convert) or self._includes(self.us_special_words, word_to_convert):
            first_y = current_word.find('ъ')
            first_a = current_word.find('а')
            
            replace = 'ъ' if first_y > first_a else 'а'
            current_word = current_word[::-1].replace(replace, 'ѫ', 1)[::-1]

            return current_word, True

        return current_word, False

    def _is_plural(self, feats):
      return 'Number=Plur' in feats

    def _is_past_tense(self, feats):
      return 'Tense=Past' in feats

    def _is_imperfect_tense(self, feats):
      return 'Tense=Imp' in feats

    def _is_3rd_person(self, feats):
      return 'Person=3' in feats

    def _is_imperative_mood(self, feats):
      return 'Mood=Imp' in feats

    def _is_masculine(self, feats):
      return 'Gender=Masc' in feats
    
    def _is_feminine(self, feats):
      return 'Gender=Fem' in feats

    def _is_noun(self, upos):
      return upos == 'NOUN' or upos == 'PROPN'

    def _is_verb(self, upos):
      return upos == 'VERB'

    def _is_adjective(self, upos):
      return upos == 'ADJ'

    def _includes(self, arr, val):
      return val in arr

def main():
    config = ConfigManager()
    converter = Converter(config)

    print(converter.convert_word('мляко'))
    
if __name__ == '__main__':
    main()