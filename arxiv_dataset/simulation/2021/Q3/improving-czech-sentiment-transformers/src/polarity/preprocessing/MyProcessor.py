import re
from functools import lru_cache

import ftfy

# Credit and based on https://github.com/cbaziotis/ekphrasis
# This is modified version for Czech!!!! Don't use for English

# noinspection PyPackageRequirements
from src.polarity.preprocessing.exmanager import ExManager


class MyTextPreProcessor:
    def __init__(self, **kwargs):
        """
        Kwargs:
            omit (list): choose what tokens that you want to omit from the text.
                possible values: ['email', 'percent', 'money', 'phone', 'user',
                    'time', 'url', 'date', 'hashtag']
                Important Notes:
                            1 - put url at front, if you plan to use it.
                                Messes with the regexes!
                            2 - if you use hashtag then unpack_hashtags will
                                automatically be set to False

            normalize (list): choose what tokens that you want to normalize
                from the text.
                possible values: ['email', 'percent', 'money', 'phone', 'user',
                    'time', 'url', 'date', 'hashtag']
                for example: myaddress@mysite.com will be transformed to <email>
                Important Notes:
                            1 - put url at front, if you plan to use it.
                                Messes with the regexes!
                            2 - if you use hashtag then unpack_hashtags will
                                automatically be set to False

            annotate (list): add special tags to special tokens.
                possible values: ['hashtag', 'allcaps', 'elongated', 'repeated']
                for example: myaddress@mysite.com -> myaddress@mysite.com <email>

            tokenizer (callable): callable function that accepts a string and
                returns a list of strings if no tokenizer is provided then
                the text will be tokenized on whitespace


            corrector (str): define the statistics of what corpus you would
                like to use [english, twitter]

            all_caps_tag (str): how to wrap the capitalized words
                values [single, wrap, every]
                Note: applicable only when `allcaps` is included in annotate[]
                    - single: add a tag after the last capitalized word
                    - wrap: wrap all words with opening and closing tags
                    - every: add a tag after each word

            spell_correct_elong (bool): choose if you want to perform
                spell correction after the normalization of elongated words.
                * significantly affects performance (speed)

            spell_correction (bool): choose if you want to perform
                spell correction to the text
                * significantly affects performance (speed)

            fix_text (bool): choose if you want to fix bad unicode terms and
                html entities.
        """
        self.omit = kwargs.get("omit", {})
        self.backoff = kwargs.get("normalize", {})
        self.include_tags = kwargs.get("annotate", {})
        self.tokenizer = kwargs.get("tokenizer", None)
        self.dicts = kwargs.get("dicts", None)
        self.spell_correct_elong = kwargs.get("spell_correct_elong", False)
        self.fix_text = kwargs.get("fix_bad_unicode", False)
        self.unpack_hashtags = kwargs.get("unpack_hashtags", False)
        self.all_caps_tag = kwargs.get("all_caps_tag", "wrap")
        self.mode = kwargs.get("mode", "normal")


        self.regexes = ExManager().get_compiled()
        if 'hashtag' in self.omit or 'hashtag' in self.backoff:
            print("You can't omit/backoff and unpack hashtags!\n "
                  "unpack_hashtags will be set to False")
            self.unpack_hashtags = False

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @staticmethod
    def add_special_tag(m, tag, mode="single"):

        if isinstance(m, str):
            text = m
        else:
            text = m.group()

        if mode == "single":
            return " {} <{}> ".format(text, tag)
        elif mode == "wrap":
            return " ".join([" <{}> {} </{}> ".format(tag, text, tag)]) + " "
        elif mode == "every":
            tokens = text.split()
            processed = " ".join([" {} <{}> ".format(t, tag)
                                  for t in tokens])
            return " " + processed + " "

    def handle_elongated_match(self, m):
        text = m.group()

        # normalize to at most 2 repeating chars
        text = self.regexes["normalize_elong"].sub(r'\1\1', text)

            # print(m.group(), "-", text)
        if "elongated" in self.include_tags:
            text = self.add_special_tag(text, "elongated")

        return text

    @lru_cache(maxsize=65536)
    def handle_repeated_puncts(self, m):
        """
        return the sorted set so mathes random combinations of puncts
        will be mapped to the same token
        "!??!?!!", "?!!!!?!", "!!?", "!?!?" --> "?!"
        "!...", "...?!" --> ".!"
        :param m:
        :return:
        """
        text = m.group()
        text = "".join(sorted(set(text), reverse=True))

        if "repeated" in self.include_tags:
            text = self.add_special_tag(text, "repeated")

        return text

    @lru_cache(maxsize=65536)
    def handle_generic_match(self, m, tag, mode="every"):
        """

        Args:
            m ():
            tag ():
            mode ():

        Returns:

        """
        text = m.group()
        text = self.add_special_tag(text, tag, mode=mode)

        return text

    @lru_cache(maxsize=65536)
    def handle_emphasis_match(self, m):
        """
        :param m:
        :return:
        """
        text = m.group().replace("*", "")
        if "emphasis" in self.include_tags:
            text = self.add_special_tag(text, "emphasis")

        return text

    @staticmethod
    def dict_replace(wordlist, _dict):
        return [_dict[w] if w in _dict else w for w in wordlist]

    @staticmethod
    def remove_hashtag_allcaps(wordlist):
        in_hashtag = False
        _words = []
        for word in wordlist:

            if word == "<hashtag>":
                in_hashtag = True
            elif word == "</hashtag>":
                in_hashtag = False
            elif word in {"<allcaps>", "</allcaps>"} and in_hashtag:
                continue

            _words.append(word)

        return _words

    def pre_process_doc(self, doc):

        doc = re.sub(r' +', ' ', doc)  # remove repeating spaces

        # ###########################
        # # fix bad unicode
        # ###########################
        # if self.fix_bad_unicode:
        #     doc = textacy.preprocess.fix_bad_unicode(doc)
        #
        # ###########################
        # # fix html leftovers
        # ###########################
        # doc = html.unescape(doc)

        ###########################
        # fix text
        ###########################
        if self.fix_text:
            doc = ftfy.fix_text(doc)

        ###########################
        # BACKOFF & OMIT
        ###########################
        for item in self.backoff:
            # better add an extra space after the match.
            # Just to be safe. extra spaces will be normalized later anyway
            doc = self.regexes[item].sub(lambda m: "" + "<" + item + ">" + "",
                                         doc)
        for (item_en, item_cs) in self.omit:
            doc = doc.replace("<" + item_en + ">", item_cs)

        # Do it twice there were some repeated
        for (item_en, item_cs) in self.omit:
            doc = doc.replace("<" + item_en + ">", item_cs)

        ###########################
        # handle special cases
        ###########################
        if self.mode != "fast":
            if "allcaps" in self.include_tags:
                doc = self.regexes["allcaps"].sub(
                    lambda w: self.handle_generic_match(w, "allcaps",
                                                        mode=self.all_caps_tag),
                    doc)

            if "elongated" in self.include_tags:
                doc = self.regexes["elongated"].sub(
                    lambda w: self.handle_elongated_match(w), doc)

            if "repeated" in self.include_tags:
                doc = self.regexes["repeat_puncts"].sub(
                    lambda w: self.handle_repeated_puncts(w), doc)

            if "emphasis" in self.include_tags:
                doc = self.regexes["emphasis"].sub(
                    lambda w: self.handle_emphasis_match(w), doc)

            if "censored" in self.include_tags:
                doc = self.regexes["censored"].sub(
                    lambda w: self.handle_generic_match(w, "censored"), doc)

        ###########################
        # unpack contractions: i'm -> i am, can't -> can not...
        ###########################


        # omit allcaps if inside hashtags
        doc = re.sub(r' +', ' ', doc)  # remove repeating spaces
        # doc = re.sub(r'<hashtag><allcaps>', '<hashtag>', doc)  # remove repeating spaces
        # doc = doc.replace('<hashtag> <allcaps>', '<hashtag>')
        # doc = doc.replace('</allcaps> </hashtag>', '</hashtag>')

        ###########################
        # Tokenize
        ###########################
        doc = self.remove_hashtag_allcaps(doc.split())
        doc = " ".join(doc)  # normalize whitespace
        if self.tokenizer:
            doc = self.tokenizer(doc)

            # Replace tokens with special dictionaries (slang,emoticons ...)
            # todo: add spell check before!
            if self.dicts:
                for d in self.dicts:
                    doc = self.dict_replace(doc, d)

        return doc

    def pre_process_docs(self, docs, lazy=True):
        from tqdm import tqdm
        for d in tqdm(docs, desc="PreProcessing..."):
            yield self.pre_process_doc(d)
