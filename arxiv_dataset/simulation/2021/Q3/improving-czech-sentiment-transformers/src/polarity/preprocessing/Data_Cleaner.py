import re


# Credit and based on https://github.com/cbaziotis/ekphrasis
from bs4 import BeautifulSoup

from src.polarity.preprocessing.MyProcessor import MyTextPreProcessor


class Data_Cleaner(object):

    def __init__(self, use_soup=True, lower_case=True, replace_bom_utf=True,
                 elong_punct=False, repeated_punct=None, text_processor=None,
                 replace_spec_chars=False, replace_spec_chars_arr=None,
                 remove_hashtag_symbol=False, remove_tweet_modifier=False,
                 elong_words=False, verbose=True, language='en'):

        """

        :param use_soup:
        :param lower_case:
        :param replace_bom_utf:
        :param elong_punct:
        :param repeated_punct:
        :param text_processor:
        :param replace_spec_chars:
        :param replace_spec_chars_arr:
        :param remove_hashtag_symbol:
        :param remove_tweet_modifier: if remove tweet begin modifiers like RT, MT, PRT, HT, CC
                                    see https://www.adweek.com/digital/advanced-twitter-terminology-to-get-you-tweeting-like-a-pro/
        :param elong_words:
        :param verbose:
        """

        self.language = language
        self.verbose = verbose
        self.use_soup = use_soup
        self.remove_hashtag_symbol = remove_hashtag_symbol
        self.remove_tweet_modifier = remove_tweet_modifier
        self.lower_case = lower_case
        self.replace_bom_utf = replace_bom_utf
        self.elong_words = elong_words

        # remove repeated punctuation the var "repeated_punct" is used for punctuation
        self.elong_punct = elong_punct
        if repeated_punct is None:
            repeated_punct = ['.', '?', '!', ',']
        self.repeated_punct = repeated_punct
        self.text_processor = text_processor

        # replace special chars
        self.replace_spec_chars = replace_spec_chars
        if replace_spec_chars_arr is None:
            replace_spec_chars_arr = "[„“\\!\_‘?+\"]"
        self.replace_spec_chars_arr = replace_spec_chars_arr

    def set_text_processor(self, text_processor):
        self.text_processor = text_processor


    def clean_text(self, text, langs_len, disable_replace=False):
        """

        :param text:
        :return:
        """

        cleaned = text
        if cleaned.startswith("<--"):
            cleaned = cleaned[3:]

        # clean text, get rid of html and other characters
        if disable_replace is False:
            if self.use_soup is True:
                soup = BeautifulSoup(cleaned, 'lxml')
                cleaned = soup.get_text()

        # deal with utf bom
        if self.replace_bom_utf is True:
            # deal with utf bom
            try:
                cleaned = cleaned.replace(u"\ufffd", "?")
            except:
                cleaned = cleaned

        if self.remove_hashtag_symbol is True:
            # split by space, loope over every token and remove hashtag symbol if is it as first and longer than 1 character
            cleaned = " ".join([word[1:] if word[0] =='#' and len(word) > 1 else word for word in cleaned.split()])

        # normalize to at most 2 repeating chars
        if self.elong_words is True:
            cleaned = re.compile("(.)\\1{2,}").sub(r'\1\1', cleaned)

        if disable_replace is False:
            if self.remove_tweet_modifier is True:
                cleaned = re.compile('RT @|MT @|PRT @|HT @|CC @').sub('@', cleaned, count=1)

        # apply external text processor
        if self.text_processor is not None:
            previous = cleaned
            cleaned = self.text_processor.pre_process_doc(cleaned)
            if type(cleaned) is list:
                cleaned = " ".join(cleaned).strip()

            if disable_replace is True:
                if len(previous.split(" ")) != len(cleaned.split(" ")):
                    cleaned = previous


        # removing of annotated files
        cleaned = cleaned.replace("<elongated>", " ")
        cleaned = cleaned.replace("</elongated>", " ")

        cleaned = cleaned.replace("<repeated>", " ")
        cleaned = cleaned.replace("</repeated>", " ")


        # Convert multiple instances of . ? ! , to single instance
        # okay...sure -> okay . sure
        # okay???sure -> okay ? sure
        # Add whitespace around such punctuation
        # okay!sure -> okay ! sure
        if disable_replace is False:
            if self.elong_punct is True:
                for c in self.repeated_punct:
                    lineSplit = cleaned.split(c)
                    while True:
                        try:
                            lineSplit.remove('')
                        except:
                            break
                    cSpace = ' ' + c + ' '
                    cleaned = cSpace.join(lineSplit)

        # remove special chars
        if disable_replace is False:
            if self.replace_spec_chars is True:
                cleaned = re.sub(self.replace_spec_chars_arr, '', cleaned)

        # remove duplicate spaces
        if disable_replace is False:
            duplicateSpacePattern = re.compile(r'\ +')
            cleaned = re.sub(duplicateSpacePattern, ' ', cleaned)

        # convert to lower case
        if self.lower_case is True:
            cleaned = cleaned.lower()

        # nope use tokenizer inside textprocessor
        # # tokenize
        # if self.tokenizer is None:
        #     words = [x for x in word_tokenize(cleaned, language=self.language, preserve_line=True) if len(x) >= 1]
        # else:
        #     words = self.tokenizer.tokenize(cleaned)
        #
        # # join by space
        # cleaned = " ".join(words).strip()


        if disable_replace is True:
            text_len = len(text.split(" "))
            cleaned_text_len = len(cleaned.split(" "))
            if cleaned_text_len != langs_len:
                raise ValueError("Lengths does not match text_len:" + str(text_len) + " cleaned_text_len:" + str(cleaned_text_len) + " langs_len:" + str(langs_len) + 'for text:\n'
                                 + text + '\n cleaned text:\n' + cleaned)
            if text_len != langs_len:
                raise ValueError("Lengths does not match text_len:" + str(text_len) + " cleaned_text_len:" + str(
                    cleaned_text_len) + " langs_len:" + str(langs_len) + 'for text:\n'
                                 + text + '\n cleaned text:\n' + cleaned)

        return cleaned


# text_processor = TextPreProcessor(
#     # terms that will be normalizedx
#     normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
#
#     fix_html=True,  # fix HTML tokens
#
#     unpack_hashtags=False,  # perform word segmentation on hashtags
#     unpack_contractions=False,  # Unpack contractions (can't -> can not)
#     spell_correct_elong=False,  # spell correction for elongated words
#     spell_correction=False,  # spell correction for elongated words
#
#     # select a tokenizer. You can use SocialTokenizer, or pass your own
#     # the tokenizer, should take as input a string and return a list of tokens
#     # tokenizer=SocialTokenizer(lowercase=False).tokenize,
#     tokenizer=None,
#     # list of dictionaries, for replacing tokens extracted from the text,
#     # with other expressions. You can pass more than one dictionaries.
#     dicts=[emoticons]
# )

text_processor_normalize = MyTextPreProcessor(
    # terms that will be normalizedx
    normalize=['url', 'email', 'percent', 'money', 'user', 'number'],

    fix_html=True,  # fix HTML tokens
    fix_bad_unicode=True,
    unpack_hashtags=False,  # perform word segmentation on hashtags
    unpack_contractions=False,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    spell_correction=True,  # spell correction for elongated words
    omit=[('email', 'email'), ('percent', 'procento'), ('money', 'peníze'),
          ('phone', 'telefon'), ('user', 'uživatel'), ('number', 'číslo'),
          ('time', 'čas'),  ('url', 'url'),  ('date', 'datum'), ('hashtag', 'hashtag')],
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=None,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=None
)