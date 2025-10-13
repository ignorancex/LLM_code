"""Modified from https://github.com/danieldeutsch/qaeval/blob/master/qaeval/answer_selection.py"""

import spacy
import random
from spacy.tokens import Span, Doc
from tqdm import tqdm

from .ae_model import BaseAEModel

random.seed(42)

NP_CHUNKS_STRATEGY = "np-chunks"
MAX_NP_STRATEGY = "max-np"
NER_STRATEGY = "ner"
TEXTRANK_STRATEGY = "textrank"
ALL_STRATEGY = "all"
STRATEGIES = [
    NP_CHUNKS_STRATEGY,
    MAX_NP_STRATEGY,
    NER_STRATEGY,
    TEXTRANK_STRATEGY,
    ALL_STRATEGY,
]

MODELS = {
    "en": "en_core_web_sm",
    "ja": "ja_core_news_sm",
    "zh": "zh_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "ko": "ko_core_news_sm",
    "ru": "ru_core_news_sm",
    "fr": "fr_core_news_sm",
}


class SpacyAE(BaseAEModel):
    def __init__(self, lp: str, strategy: str = "all", extract_from="target"):
        """Task-specific question generation models."""
        self.strategy = strategy
        self.nlp = spacy.load(MODELS[lp])
        if extract_from not in ["target", "source"]:
            raise ValueError(
                "Please specify from where the keyphrases should be extracted: 'source' or 'target'."
            )
        else:
            self.extract_from = extract_from
        if (
            self.strategy == "all" or self.strategy == "textrank"
        ):  # ['textrank', 'biasedtextrank', 'positionrank']
            # Is this used?
            import pytextrank  # noqa

            self.nlp.add_pipe("textrank")

    def _get_np_chunks_answers(self, sentence: Span) -> list[str]:
        chunks = []
        for chunk in sentence.noun_chunks:
            chunks.append(str(chunk))
        return chunks

    def _get_max_np_answers(self, sentence: Span) -> list[str]:
        root = sentence.root
        nodes = [root]
        nps = []

        while len(nodes) > 0:
            node = nodes.pop()

            # If the node is a noun, collect all of the tokens
            # which are descendants of this node
            recurse = True
            if node.pos_ in ["NOUN", "PROPN"]:
                min_index = node.i
                max_index = node.i
                stack = [node]
                while len(stack) > 0:
                    current = stack.pop()
                    min_index = min(min_index, current.i)
                    max_index = max(max_index, current.i)
                    for child in current.children:
                        stack.append(child)

                sent_start_index = sentence[0].i

                # Because of parsing issues, we only take NPs if they are shorter than a given length
                num_tokens = max_index - min_index + 1
                if num_tokens <= 7:
                    recurse = False
                    span = sentence[
                        min_index - sent_start_index : max_index + 1 - sent_start_index
                    ]
                    nps.append(str(span))

            if recurse:
                # Otherwise, process all of this node's children
                for child in node.children:
                    nodes.append(child)

        return nps

    def _get_ner_answers(self, sentence: Span) -> list[str]:
        ners = []
        for entity in sentence.ents:
            if entity.label_ in [
                "PERSON",
                "NORP",
                "FAC",
                "ORG",
                "GPE",
                "LOC",
                "EVENT",
                "WORK_OF_ART",
            ]:
                ners.append(str(entity))
        return ners

    def _get_tr_answers(self, doc: Doc) -> list[str]:
        keyphrases = []
        for phrase in doc._.phrases:
            keyphrases.append(str(phrase.text))
        return keyphrases

    def _get_all_answers(self, sentence: Span) -> list[str]:
        answers = set()
        answers |= set(self._get_np_chunks_answers(sentence))
        answers |= set(self._get_max_np_answers(sentence))
        answers |= set(self._get_ner_answers(sentence))
        return list(answers)

    def _extract_answer_text(self, text: str, num_answers: int | None) -> list[str]:
        """
        Selects a list of noun phrases from the input `text.
        """
        doc = self.nlp(text)
        answers = []

        if self.strategy == TEXTRANK_STRATEGY or self.strategy == ALL_STRATEGY:
            answers.extend(self._get_tr_answers(doc))
        else:
            for sent in doc.sents:
                if self.strategy == NP_CHUNKS_STRATEGY:
                    answers.extend(self._get_np_chunks_answers(sent))
                elif self.strategy == MAX_NP_STRATEGY:
                    answers.extend(self._get_max_np_answers(sent))
                elif self.strategy == NER_STRATEGY:
                    answers.extend(self._get_ner_answers(sent))
                elif self.strategy == ALL_STRATEGY:
                    answers.extend(self._get_all_answers(sent))
                else:
                    raise Exception(f"Unknown strategy: {self.strategy}")
        if num_answers:
            return random.choices(answers, k=min(num_answers, len(answers)))
        else:
            return answers

    def extract_answers(
        self,
        passages: list[str],
        num_answers: int | None = None,
    ) -> list[list[str]]:
        return [self._extract_answer_text(text, num_answers) for text in tqdm(passages)]
