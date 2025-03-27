
def load_word_list(file_path):
    f = open(file_path, "r")
    word_list = str(f.read()).split("\n")
    #print(word_list[0:10])
    return word_list


from icecream import ic
import re
def count_word_list(word_list, str):
    str_words = str.split(' ')
    counts = 0
    romve_chars = '[0-9’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    for i in str_words:
        word = i.lower()
        word = re.sub(romve_chars, '',word)
        for j in word_list:
            if j == word:
                counts+=1
                break
    return counts

def Sarcasm_Detection(str):
    pass



if __name__=='__main__':
    pos_word_list = load_word_list('pos_words.txt')
    neg_word_list = load_word_list('neg_words.txt')
    from datasets import load_from_disk
    dataset = load_from_disk('./yelp_small_test')
    text, label = dataset[5]['text'], dataset[5]['label']
    print(text, label)
    from icecream import ic
    ic(count_word_list(pos_word_list, text))
    ic(count_word_list(neg_word_list, text))

