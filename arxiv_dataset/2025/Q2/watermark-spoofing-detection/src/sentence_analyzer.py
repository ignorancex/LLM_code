import numpy as np
import os
import pickle as pkl
import scipy.stats as stats
from typing import List, Dict
import pickle as pkl
from watermark_stealing.watermarks.kgw.alternative_prf_schemes import (
    seeding_scheme_lookup,
)


class SentenceAnalyzer:
    def __init__(self, sentence, server, load_mode: bool = False):
        if not load_mode:
            self.sentence = sentence
            _, context_size, _, _ = seeding_scheme_lookup(
                server.watermarks[0].cfg.generation.seeding_scheme
            )
            if server.watermarks[0].cfg.generation.seeding_scheme == "selfhash":
                context_size -= 1
            self.context_size = context_size
            self._get_tokens_color(server)

    def _tokenize_sentence(self, server):
        """Get the tokenized sentence."""

        tokenizer = server.model.tokenizer
        encoded_completion = tokenizer.encode(self.sentence, return_tensors="pt")
        return encoded_completion[0].tolist()[1:]

    def _get_tokenized_sentence(self):
        """Get the tokenized sentence."""
        return self.sentence_tokens

    def _get_tokens_color(self, server):
        """Get the color of the tokens in the sentence."""

        detection_result = server.detect([self.sentence])[0]

        if detection_result is None:
            self.color_mask = None
            self.sentence_tokens = None
            return

        token_mask = detection_result["token_mask"]

        self.color_mask = token_mask
        self.sentence_tokens = self._tokenize_sentence(server)

    def _check_occurence_ngram(self, occurence_dictionnary: Dict):
        key = next(iter(occurence_dictionnary))

        if isinstance(key, tuple):
            n_gram = len(key)
        elif isinstance(key, int):
            n_gram = 1

        return n_gram

    def get_length(self):
        color_mask = np.array(self.color_mask)
        return np.sum(color_mask != -1)

    def relabel_duplicates(self):
        context_size = self.context_size

        tokenized_sentence = self._get_tokenized_sentence()
        seen_ngrams = set()
        color_mask = np.array(self.color_mask)

        context_size += 1  # To include the token itself

        for i in range(len(tokenized_sentence) - context_size):
            ngram = tuple(tokenized_sentence[i : i + context_size])

            if ngram in seen_ngrams:
                color_mask[i + context_size - 1] = -1
            else:
                seen_ngrams.add(ngram)

    def merge(self, sentence_analyzer):
        """Merge the current SentenceAnalyzer with another one."""

        self.sentence += sentence_analyzer.sentence
        self.color_mask = np.concatenate(
            [self.color_mask, sentence_analyzer.color_mask]
        )
        self.sentence_tokens = np.concatenate(
            [self.sentence_tokens, sentence_analyzer.sentence_tokens]
        )

        self.relabel_duplicates()

    def _save(self):
        to_save = {
            "sentence": self.sentence,
            "color_mask": self.color_mask,
            "sentence_tokens": self.sentence_tokens,
        }

        return to_save

    def save(self, path):
        """Save the object to a dictionary."""

        to_save = self._save()

        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        np.save(path, to_save)

    def shallow_clean_sentence(self, token_target: int):
        encoded_sentence = self._get_tokenized_sentence()

        if token_target > len(encoded_sentence) - 3:
            return False
        else:
            mask = np.array(self.color_mask)
            idx = find_index_with_n_ones(mask, token_target)

            if idx == -1:
                return False
            new_encoded_sentence = encoded_sentence[:idx]
            self.color_mask = self.color_mask[:idx]
            self.sentence_tokens = new_encoded_sentence
            return True

    def compute_z_score(self, gamma: float):
        x_i = np.array(self.color_mask)
        s_g = np.sum(x_i == 1)
        T = np.sum(x_i != -1)

        return (s_g - gamma * T) / np.sqrt(T * gamma * (1 - gamma))

    def get_token_occurence_score(self, occurrence_dictionnaries: List[Dict]):
        if not isinstance(occurrence_dictionnaries, list):
            occurrence_dictionnaries = [occurrence_dictionnaries]

        token_occurence_score_array = np.zeros(len(self.color_mask))

        for occurrence_dictionnary in occurrence_dictionnaries:
            encoded_sentence = self._get_tokenized_sentence()
            token_occurence_score = []

            n_gram = self._check_occurence_ngram(occurrence_dictionnary.ngram_counts)

            for i in range(len(encoded_sentence)):
                if i < n_gram - 1:
                    token_occurence_score.append(np.nan)
                else:
                    n_gram_tuple = tuple(encoded_sentence[i - n_gram + 1 : i + 1])
                    token_occurence_score.append(
                        occurrence_dictionnary.get_ngram_count(n_gram_tuple)
                    )

            median_score = np.nanmedian(token_occurence_score)
            token_occurence_score = np.array(token_occurence_score)
            token_occurence_score[np.isnan(token_occurence_score)] = median_score

            token_occurence_score_array += np.array(token_occurence_score)

        return token_occurence_score_array

    def get_token_occurence_context_score(self, ngram, context_length: int = None):
        if context_length is None:
            context_length = self.context_size

        token_occurence = self.get_token_occurence_score(ngram)

        token_occurence = np.roll(token_occurence, context_length)
        return token_occurence

    def compute_correlation(self, ngram, context_length: int = None):
        token_occurence = self.get_token_occurence_context_score(ngram, context_length)
        color_mask = np.array(self.color_mask)
        mask = color_mask != -1
        token_occurence = np.array(token_occurence)[mask]
        color_mask = color_mask[mask]

        length = np.sum(mask)

        return np.arctanh(
            stats.spearmanr(token_occurence, color_mask).correlation
        ), length


def load_sentence_analyzer(path, server, data: dict = None):
    """Load a SentenceAnalyzer object from a json file."""
    if data is None:
        data = np.load(path, allow_pickle=True).item()

    sentence = data["sentence"]
    color_mask = data["color_mask"]

    sentence_analyzer = SentenceAnalyzer(None, None, load_mode=True)
    sentence_analyzer.sentence = sentence
    sentence_analyzer.color_mask = color_mask
    sentence_analyzer.sentence_tokens = sentence_analyzer._tokenize_sentence(server)
    _, context_size, _, _ = seeding_scheme_lookup(
        server.watermarks[0].cfg.generation.seeding_scheme
    )
    if server.watermarks[0].cfg.generation.seeding_scheme == "selfhash":
        context_size -= 1
    sentence_analyzer.context_size = context_size

    return sentence_analyzer


def batch_save_analyzers(
    filename: str, analyzers: List[SentenceAnalyzer],
):
    saves = [analyzer._save() for analyzer in analyzers]

    with open(filename, "wb") as f:
        pkl.dump(saves, f)


def find_index_with_n_ones(a, n):
    count = 0
    for idx, value in enumerate(a):
        if value != -1:
            count += 1
        if count == n:
            return idx + 1
    return -1
