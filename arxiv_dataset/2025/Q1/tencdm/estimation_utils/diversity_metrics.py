import spacy
import string
import numpy as np
import nltk
import itertools
from typing import Dict, List


class NGramStats:
    """Ngram basic statistics and entropy, working with tokenized & lowercased data (+ variant excluding punctuation):
    - data length (total number of words)
    - mean instance length (number of words)
    - distinct-N (ratio of distinct N-grams / total number of N-grams)
    - vocab_size-N (total number of distinct N-grams)
    - unique-N (number of N-grams that only occur once in the whole data)
    - entropy-N (Shannon entropy over N-grams)
    - cond-entropy-N (language model style conditional entropy -- N-grams conditioned on N-1-grams)
    All these are computed for 1,2,3-grams (conditional entropy only for 2,3).
    Based on:
    https://github.com/evanmiltenburg/NLG-diversity/blob/main/diversity.py
    https://github.com/tuetschek/e2e-stats/blob/master/nlg_dataset_stats.py
    """

    def __init__(self):
        self.tokenizer = spacy.load("en_core_web_sm").tokenizer
        self.PUNCTUATION = set(string.punctuation)
        self.results = dict()
        

    def compute(self, texts: List[str]) -> Dict:
        data = self._list_tokenized_lower_nopunct(texts)

        results = {}
        lengths = [len(inst) for inst in data]
        results[f"total_length"] = np.sum(lengths)
        results[f"mean_pred_length"] = np.mean(lengths)
        results[f"std_pred_length"] = np.std(lengths)

        last_ngram_freqs = None  # for conditional entropy, we need lower-level n-grams

        for N in [1, 2, 3, 4]:
            ngram_freqs = self._ngram_freqs(data, N)
            self.ngram_freqs = ngram_freqs

            num_uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
            total_ngram_num = sum(list(ngram_freqs.values()))

            results[f"distinct-{N}"] = len(ngram_freqs) / total_ngram_num if total_ngram_num > 0 else 0
            results[f"vocab_size-{N}"] = len(ngram_freqs)
            results[f"unique-{N}"] = num_uniq_ngrams
            results[f"entropy-{N}"] = self._entropy(ngram_freqs)

            if last_ngram_freqs:
                results[f"cond_entropy-{N}"] = self._cond_entropy(ngram_freqs, last_ngram_freqs)
            last_ngram_freqs = ngram_freqs

        self.results = results
        self.results["msttr"] = self._MSTTR(data)
        self.results["dublicates"] = self._find_consequent_dublicates(data)
        self.results["diversity"] = self._diversity()

        self.results = results
        return results

    def _ngram_freqs(self, data: List[List[str]], N: int) -> Dict:
        """Return a dict of all ngrams and their freqsuencies"""
        ngram_freqs = {}  # ngrams with frequencies
        for inst in data:
            for ngram in nltk.ngrams(inst, N):
                ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
        return ngram_freqs

    def _entropy(self, ngram_freqs: Dict) -> float:
        """Shannon entropy over ngram frequencies"""
        total_freq = sum(ngram_freqs.values())
        return -sum(
            [
                freq / total_freq * np.log2(freq / total_freq)
                for freq in ngram_freqs.values()
            ]
        )

    def _cond_entropy(self, joint_freqs: Dict, marginal_freqs: Dict) -> float:
        """Conditional/next-word entropy (language model style), using ngrams (joint) and n-1-grams (ctx)."""
        total_num_joint = sum(joint_freqs.values())
        total_num_marg = sum(marginal_freqs.values())
        # H(y|x) = - sum_{x,y} p(x,y) log_2 p(y|x)
        # p(y|x) = p(x,y) / p(x)
        return -sum(
            [
                freq / total_num_joint * np.log2(
                    (freq / total_num_joint) / (marginal_freqs[ngram[:-1]] / total_num_marg))
                for ngram, freq in joint_freqs.items()
            ]
        )

    def _list_tokenized_lower_nopunct(self, texts: List[str]):
        texts = self._tokenize(texts)
        return [
            [w.lower().translate(str.maketrans('', '', string.punctuation)) for w in ref if w not in self.PUNCTUATION]
            for ref in texts]

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        return [[str(token) for token in self.tokenizer(sentence)] for sentence in texts]
    

    def _TTR(self, text: List[str]) -> float:
        tokens = set(text)
        return len(tokens) / len(text)

    def _MSTTR(self, text: List[List[str]], seg_len=100) -> float:
        flat_list = list(itertools.chain.from_iterable(text))
        ttrs = []
        for i in range(0, len(flat_list), seg_len):
            segment = flat_list[i: i + seg_len]
            if len(segment) < seg_len:
                break
            ttrs.append(self._TTR(segment))
        return np.mean(ttrs) if ttrs else np.nan

    def __str__(self):
        keys = sorted(list(self.results.keys()))

        n_sym = 20
        result = ["-" * (n_sym * 2 + 3)]
        for key in keys:
            str_value = f"{self.results[key]:0.3f}"
            result.append(f"|{key}{' ' * (n_sym - len(key))}|{str_value}{' ' * (n_sym - len(str_value))}|")
            result.append("-" * (n_sym * 2 + 3))
        return "\n".join(result)

    def _find_consequent_dublicates(self, data: List[List[str]]) -> int:
        """
        Count number of consequent same words
        :param data:
        :return:
        """
        num_dublicates = 0
        for seq in data:
            prev_word = ""
            for word in seq:
                if word == prev_word:
                    num_dublicates += 1
                prev_word = word

        return num_dublicates
    
    def _diversity(self):
        diversity = 1.
        for N in [2, 3, 4]:
            diversity *= self.results[f"distinct-{N}"]
        return diversity

