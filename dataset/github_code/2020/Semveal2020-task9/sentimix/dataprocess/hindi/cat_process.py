import re
import emoji
from nltk.corpus import wordnet
import pandas as pd
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from ekphrasis.classes.tokenizer import SocialTokenizer
import itertools


def replacement(review):
    review = review.lower()
    review = review.replace("# ", "#")
    review = review.replace("@ ", "@")
    review = review.replace(" _ ", "_")
    review = review.replace(" __ ", "")
    review = review.replace("__ ", "__")
    review = review.replace("_ ", "_")
    review = review.replace(' ’ s ', ' is ')
    review = review.replace(' ’ m ', ' am ')
    review = review.replace(' ’ re ', ' are ')
    review = review.replace("’ ll", 'will')
    review = review.replace("i'm", 'i am')
    review = review.replace("you'll", 'you will')
    review = review.replace("don't", 'do not')
    review = review.replace("can't", "can not")
    review = review.replace("it's", "it is")
    review = review.replace("she's", "she is")
    review = review.replace("let's", "let us")
    review = review.replace("i'll", "i will")
    review = review.replace("haven't", "have not")
    review = review.replace("doesn't", "does not")
    review = review.replace("he's", "he is")
    review = review.replace("doesn ’ t", "does not")
    review = review.replace("didn ’ t", "did not")
    review = review.replace("i ’ ve", "i have")
    review = review.replace("we'll", "we will")
    review = review.replace("i ’ d", "i had")
    review = review.replace("won ’ t", "would not")
    review = review.replace("we ’ ve", "we have")
    review = review.replace("you ’ ve", "you are")
    review = review.replace("ain ’ t", "are not")
    review = review.replace("y ’ all", "you and all")
    review = review.replace("couldn ’ t", "could not")
    review = review.replace("haven ’ t", "have not")
    review = review.replace("aren't", "are not")
    review = review.replace("you ’ d", "you had")
    review = review.replace("that's", "that is")
    review = review.replace("wasn't", "was not")
    review = review.replace("he'll", "he will")
    review = review.replace("ma ’ am", 'madam')
    review = review.replace("ma'am ", "madam")
    review = review.replace("they ’ ve", "they have")
    review = review.replace('don ’ t', 'do not')
    review = review.replace('can ’ t', 'can not')
    review = review.replace('isn ’ t', 'is not')
    review = review.replace("b'day", 'birthday')
    review = review.replace("I've", 'I have')
    review = review.replace("didn't", "did not")
    review = review.replace("u're", "you are")
    review = review.replace("What's", 'what is')
    review = review.replace("you're", 'you are')
    review = review.replace("You're", 'you are')
    review = review.replace("I'm", 'I am')
    review = review.replace("isn't", "is not")
    review = review.replace(" ___", "___ ")
    review = review.replace("won't", 'will not')
    review = review.replace('can ’ t', 'can not')
    review = review.replace('I ’ ll ', 'I will')
    review = review.replace("we ’ ll", 'we will')
    review = review.replace("didn ’ t", 'did not')
    review = review.replace(" u ", ' you ')
    review = review.replace("wasn ’ t", 'was not')
    review = review.replace('+', 'and')
    review = review.replace('whats ', 'what is').replace("what's ", 'what is ').replace("i'm ", 'i am')
    review = review.replace("it's ", 'it is ')
    review = review.replace('Iam ', 'I am ').replace(' iam ', ' i am').replace(' dnt ', ' do not ')
    review = review.replace('I ve ', 'I have ').replace('I m ', ' I am ').replace('i m ', 'i am ')
    review = review.replace('Iam ', 'I am ').replace('iam ', 'i am ')
    review = review.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ', ' have ')
    review = review.replace(' F ', ' Fuck ').replace("Ain\'t ", ' are not ').replace(' lv ', ' love ')
    review = review.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ', ' it is ')
    review = review.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')
    return review


def ekphrasis_config(str):
    social_tokenizer = SocialTokenizer(lowercase=False).tokenize
    return social_tokenizer(str)

def removeURL(text):
    """ remove url address  """
    # text = re.sub("((http[s]{0,1}|ftp)//[a-zA-Z0-9\\.\\-]+\\[.]([a-zA-Z]{2,4})(:\\d+)?(/[a-zA-Z0-9\\.\\-~!@#$%^&*+?:_/=<>]*)?)|((www.)|[a-zA-Z0-9\\.\\-]+\\.([a-zA-Z]{2,4})(:\\d+)?(/[a-zA-Z0-9\\.\\-~!@#$%^&*+?:_/=<>]*)?)",'', text)
    clean_sentence = []
    review_text = text.split()
    for i in review_text:
        if i.startswith("htt"):
            continue
        if i.startswith("rt"):
            continue
        else:
            clean_sentence.append(i)
    text = ' '.join(str(i) for i in clean_sentence)
    return text


def replaceuser(text):
    """ Replaces "@user" with "User" """
    text = re.sub('@[^\s]+', 'user', text)
    return text


def removeuser(text):
    """ remove "@user"  """
    text = re.sub(r'@[^\s]+', r'', text)
    return text


def removehashtag(text):
    """ Removes hastag in front of a word """
    text = re.sub('#([^\s]+)', '', text)
    return text


def replacehashtag(text):
    """ Removes hastag in front of a word """
    text = re.sub('#([^\s]+)', 'hashtag', text)
    return text


def removeHashtagInFrontOfWord(text):
    """ Removes hastag in front of a word """
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def removeNumbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '', text)
    return text


def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '', text)
    return text


def replaceMultiStopMark(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", '', text)
    return text


""" Creates a dictionary with slangs and their equivalents and replaces them """
with open('slang.txt', encoding='utf-8') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
                     for line in file if line.strip())
# print(slang_map)
lines_seen = set()


def replaceslang(text):
    list_str = ekphrasis_config(text)
    for index in range(len(list_str)):
        if list_str[index] in slang_map.keys():
            lines_seen.add(list_str[index])
            print(list_str[index])
            list_str[index] = slang_map[list_str[index]]
    text = ' '.join(list_str)
    # for item in slang_map.keys():
    #     # print(item)
    #     text = text.replace(' ' + item + ' ', ' ' + slang_map[item].lower() + ' ')

    return text


""" Replaces contractions from a string to their equivalents """
contraction_patterns = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'),
                        (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                        (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'),
                        (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'),
                        (r'wont', 'will not')]


def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text


# def replaceElongated(word):
#     """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """
#
#     repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
#     repl = r'\1\2\3'
#     if wordnet.synsets(word):
#         return word
#     repl_word = repeat_regexp.sub(repl, word)
#     if repl_word != word:
#         return replaceElongated(repl_word)
#     else:
#         return repl_word

# def removeEmoticons(text):
#     """ Removes emoticons from text """
#     text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
#     return text

def removeSpecialchar(text):
    return re.findall("[a-zA-Z]+", text.lower())


def processemoji(text):
    repeatedChars = ['user', 'hashtag']
    for c in repeatedChars:
        lineSplit = text.split(c)
        while True:
            try:
                lineSplit.remove('')
            except:
                break
        cSpace = ' ' + c + ' '
        text = cSpace.join(lineSplit)

    emoji_repeatedChars = TWEMOJI_LIST
    for emoji_meta in emoji_repeatedChars:
        emoji_lineSplit = text.split(emoji_meta)
        while True:
            try:
                emoji_lineSplit.remove('')
                emoji_lineSplit.remove(' ')
                emoji_lineSplit.remove('  ')
                emoji_lineSplit = [x for x in emoji_lineSplit if x != '']
            except:
                break
        emoji_cSpace = ' ' + TWEMOJI[emoji_meta][0] + ' '
        text = emoji_cSpace.join(emoji_lineSplit)

    for item in LOGOGRAM.keys():
        text = text.replace(' ' + item + ' ', ' ' + LOGOGRAM[item].lower() + ' ')
        # print(item)

    list_str = ekphrasis_config(text)
    for index in range(len(list_str)):

        if list_str[index] in EMOTICONS_TOKEN.keys():
            list_str[index] = EMOTICONS_TOKEN[list_str[index]][1:len(EMOTICONS_TOKEN[list_str[index]]) - 1].lower()
    for index in range(len(list_str)):
        if list_str[index] in LOGOGRAM.keys():
            # print("kkk",list_str[index])
            list_str[index] = LOGOGRAM[list_str[index]].lower()

    string = ' '.join(list_str)
    string = emoji.demojize(string.lower())
    string = re.sub(':\S+?:', '', string)

    return string


def process(train):
    clean_train_reviews = []
    text = ''
    for i, review in enumerate(train["sentence"]):
        y = train['label'][i]
        review = replacement(review)
        review = removeURL(review)
        review = replaceuser(review)
        review = replacehashtag(review)
        review = replaceMultiExclamationMark(review)
        review = replaceMultiQuestionMark(review)
        review = replaceMultiStopMark(review)
        review = replaceslang(review)
        review = processemoji(review)
        # review = removeNumbers(review)
        review = replaceContraction(review)
        # review = replaceElongated(review)
        # review = removeSpecialchar(review)
        review_text = re.sub("[^a-zA-Zn.?!]", " ", review)
        words = review_text.lower().split()
        text = ' '.join(words).lower()
        # review = removeEmoticons(review)
        # text = ' '.join(review).lower()

        text += '\t' + y
        clean_train_reviews.append(text)
    return clean_train_reviews


def processtest(test):
    clean_test_reviews = []
    text = ''
    for i, review in enumerate(test["sentence"]):
        # uid = test['uid'][i]
        review = replacement(review)
        review = removeURL(review)
        review = replaceuser(review)
        review = replacehashtag(review)
        review = replaceMultiExclamationMark(review)
        review = replaceMultiQuestionMark(review)
        review = replaceMultiStopMark(review)
        review = replaceslang(review)
        review = processemoji(review)
        # review = removeNumbers(review)
        review = replaceContraction(review)
        # review = replaceElongated(review)
        # review = removeSpecialchar(review)
        review_text = re.sub("[^a-zA-Zn.?!]", " ", review)
        words = review_text.lower().split()
        text = ' '.join(words).lower()
        # review = removeEmoticons(review)
        # text = ' '.join(review).lower()

        # uid += '\t' + text
        clean_test_reviews.append(text)
    return clean_test_reviews


def write_file(file_name, clean_reviews):
    with open(file_name, 'w', encoding='utf8') as file_obj:
        for i in clean_reviews:
            file_obj.write(i + '\n')


def write_as_test(file_name, texts, sids):
    with open(file_name, 'w', encoding='utf8') as my_file:
        # my_file.write('uid\tsetence\n')
        for i, text in enumerate(texts):
            my_file.write('%s\t%s\n' % (sids[i], text))


if __name__ == '__main__':
    train_file_name = './data/train_user_hashtag_hindi_nochar.tsv'
    dev_file_name = './data/dev_user_hashtag_hindi_nochar.tsv'
    test_file_name = './data/test_user_hashtag_hindi_nochar.tsv'
    train = pd.read_csv("./data/get_train_f.tsv", header=0, delimiter="\t", quoting=3)
    dev = pd.read_csv("./data/get_dev_f.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./data/get_test_f1.txt", header=0, delimiter="\t", quoting=3)
    clean_train_reviews = process(train)
    clean_dev_reviews = process(dev)
    clean_test_sentence = processtest(test)
    write_file(train_file_name, clean_train_reviews)
    write_file(dev_file_name, clean_dev_reviews)
    write_as_test(test_file_name, clean_test_sentence, test["uid"])
