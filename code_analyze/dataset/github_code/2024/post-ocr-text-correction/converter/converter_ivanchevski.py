import os
import re
import sys
import nltk
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import ConfigManager

class Converter:

    def __init__(self, config):
        self.config = config

        self.cons = ["б", "в", "г", "д", "ж", "з", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ"]
        self.soft_feminine_ending_words = self.read_file_into_list(self.config.get_converter_ivanchevski_resources_dir() + '/feminine-nouns-ending-on-con.txt')
        
        self.soft_ending_masculine_nouns = self.read_file_into_list(self.config.get_converter_ivanchevski_resources_dir() + '/masculine-nouns-soft-ending.txt')

        self.us_roots = self.read_file_into_list(self.config.get_converter_ivanchevski_resources_dir() + '/us-roots.txt')

        self.yat_roots = self.read_file_into_list(self.config.get_converter_ivanchevski_resources_dir() + '/yat-roots.txt')
        
        self.exclusion_words_1 = ["празник","празнич","сърц","нужни"]
        self.exclusion_words_2 = ["нишк","овошк"]
        self.corrected_exclusion_words = ["праздник","празднич","сърдц","нуждни","нищк","овощк"]


        self.abbreviations = ["лв","др","абв","т","н","пр","сащ","проф","гр","сл","ссср","хр","доц","бкп","сдс","дпс","мн","ч","р","ж",
            "ср","св","стр","подкл","кл","чл","кр","мвр","бан","момн","мон","аец","заб","бул","ул","вм","вж","пл","церн","бзнс","бнб"]
            
        self.c_len = len(self.cons)
        self.us_len = len(self.us_roots)
        self.yat_len = len(self.yat_roots)
        self.ex_len_1 = len(self.exclusion_words_1)
        self.ex_len_2 = len(self.exclusion_words_2)

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

    def split_to_words(self, text):
        words = [x for x in re.split('[ *.+:;,?\r\t\n{}]+', text) if x != '']

        return words, len(words)

    def includes(self, arr, val):
        return val in arr

    def check_ending(self, word_to_convert, current_word):
        """
        word_to_convert - initial word
        current_word - currently converted word to some step
        """
        last_char = word_to_convert[len(word_to_convert) - 1]

        if self.includes(self.cons, last_char):
            if self.includes(self.abbreviations, word_to_convert):
                return current_word, False
            
            if self.includes(self.soft_feminine_ending_words, word_to_convert) or self.includes(self.soft_ending_masculine_nouns, word_to_convert): # Проверяваме за меко завършващите думи
                current_word += 'ь'
                            
                return current_word, True
            
            if word_to_convert[len(word_to_convert) - 2:] == "ят" or word_to_convert[len(word_to_convert) - 2:] == "ът": # Проверяваме за пълен член
                if self.includes(self.soft_ending_masculine_nouns, (word_to_convert[0:len(word_to_convert) - 2])):
                    current_word = current_word[0:len(current_word)-2] + "ьт"

            # Във всички други случаи се добавя Ъ към края
            current_word += "ъ"
            
            return current_word, True

        return current_word, False

    def check_us(self, word_to_convert, current_word):
        """
        word_to_convert - initial word
        current_word - currently converted word to some step
        """
        for k in range(self.us_len):
            us_index = current_word.find(self.us_roots[k])
            if us_index != -1: 
                current_word = current_word[0:us_index] + current_word[us_index:].replace("ъ","ѫ", 1)
                
                return current_word, True

        if word_to_convert == "са":
            current_word = current_word.replace("а","ѫ", 1)
                
            return current_word, True
        
        return current_word, False

    def check_yat(self, word_to_convert, current_word):
        return_value = False
        if word_to_convert[len(word_to_convert) - 2:] == "те":
            
            current_word = current_word[0:len(current_word) - 1] + "ѣ"
            
            return_value = True

        elif word_to_convert[0:3] == "пре":
            current_word = current_word.replace("е","ѣ", 1)

            return_value = True

        for k in range(self.yat_len):
            yat_index = current_word.find(self.yat_roots[k])

            

            if yat_index != -1:
                if k < 209:
                    current_word = current_word[0:yat_index] + current_word[yat_index:].replace("е","ѣ", 1)
                else:
                    current_word = current_word[0:yat_index] + current_word[yat_index:].replace("я","ѣ", 1)

                return current_word, True

        if word_to_convert == "де":
            
            current_word = current_word[0] + "ѣ"
            
            return_value = True
        
        return current_word, return_value

    def check_vs(self, word_to_convert, current_word):
        if word_to_convert == "във" or word_to_convert == "със": # със, във -> съ, въ
            current_word = current_word[0] + "ъ"
            
            return current_word, True

        return current_word, False

    def check_feminine_the(self, word_to_convert, current_word):
        if word_to_convert[len(word_to_convert) - 3:] == "тта" or word_to_convert[len(word_to_convert)-3:] == "щта":
            current_word = current_word[0:len(current_word) - 2] + "ь" + current_word[len(current_word)-2:]
            
            return current_word, True

        return current_word, False


    def check_exclusion_words_1(self, word_to_convert, current_word):
        for k in range(self.ex_len_1):
            ex_index = current_word.find(self.exclusion_words_1[k])

            if ex_index != -1:
                current_word = current_word.replace(self.exclusion_words_1[k], self.corrected_exclusion_words[k], 1)
                
                return current_word, True

        return current_word, False

    """
    Проверява дали думата съдържа корен, който е имал различно изписване 
    """
    def check_exclusion_words_2(self, word_to_convert, current_word):
        for k in range(self.ex_len_2):
            ex_index = current_word.find(self.exclusion_words_2[k])
            if ex_index != -1:
                current_word = current_word.replace(self.exclusion_words_2[k], self.corrected_exclusion_words[k+self.ex_len_1], 1)
                
                return current_word, True

        return current_word, False

    def convert_words(self, words):
        new_words  = []
        for word in words:

            word = word.lower()

            has_changed_1 = False
            has_changed_2 = False
            has_changed_3 = False
            has_changed_4 = False
            has_changed_5 = False
            has_changed_6 = False
            has_changed_7 = False
            
            current_word = word

            current_word, has_changed_1 = self.check_ending(word, current_word)
            current_word, has_changed_2 = self.check_us(word, current_word)
            current_word, has_changed_3 = self.check_yat(word, current_word)
            current_word, has_changed_4 = self.check_vs(word, current_word)
            current_word, has_changed_5 = self.check_feminine_the(word, current_word)
            current_word, has_changed_6 = self.check_exclusion_words_1(word, current_word)
            current_word, has_changed_7 = self.check_exclusion_words_2(word, current_word)

            new_words.append(current_word)

        return ' '.join(new_words)
        

def main():
    config = ConfigManager()
    converter = Converter(config)

    print(converter.convert_words(['мляко', 'със', 'сусам']))
    
if __name__ == '__main__':
    main()