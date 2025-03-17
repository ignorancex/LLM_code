# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class FastLexRankSummarizer:
    """
    Calculate the LexRank score for each sentence in the corpus and return the top sentences using a fast implementation.
    :param corpus: list of sentences
    :param model_path: path to the sentence transformer model used for sentence embeddings
    :param threshold: threshold for the cosine similarity
    :return: list of sentences with the highest LexRank score
    """

    def __init__(
        self,
        model_path: str = "all-MiniLM-L12-v2",
        threshold: float = None,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_path, device=self.device)

    def _get_sentence_embeddings(self, corpus: list[str]) -> np.ndarray:
        """
        Calculate the sentence embeddings for the corpus
        :return: sentence embeddings
        """
        embeddings = self.model.encode(corpus)
        return embeddings

    def get_lexrank_scores(self, corpus: list[str]) -> np.ndarray:
        """
        Calculate the LexRank score for each sentence
        :return: LexRank scores
        """
        embeddings = self._get_sentence_embeddings(corpus)

        # Transpose the similarity matrix
        F = embeddings.T
        # Normalize the similarity matrix
        z = embeddings.sum(axis=0)
        z = z / np.sqrt((z**2).sum(axis=0))
        # Calculate the LexRank scores
        approx_scores = np.dot(z.T, F)
        return approx_scores

    def _get_top_sentences(self, lexrank_scores: np.ndarray, n: int = 3) -> list[str]:
        """
        Return the top sentences with the highest LexRank score
        :param lexrank_scores: LexRank scores
        :param n: number of sentences to return
        :return: list of sentences with the highest LexRank score
        """
        top_sentences = np.argsort(lexrank_scores)[::-1][:n]
        return top_sentences

    def summarize(self, corpus: list[str], n: int = 3) -> list[str]:
        """
        Calculate the LexRank score for each sentence in the corpus and return the top sentences
        :param n: number of sentences to return
        :return: list of sentences with the highest LexRank score
        """

        lexrank_scores = self.get_lexrank_scores(corpus)
        top_sentences = self._get_top_sentences(lexrank_scores, n)
        return [corpus[i] for i in top_sentences]

    def __call__(self, n: int = 3) -> list:
        """
        Calculate the LexRank score for each sentence in the corpus and return the top sentences
        :param n: number of sentences to return
        :return: list of sentences with the highest LexRank score
        """
        return self.summarize(n)

    def __repr__(self) -> str:
        return f"fastLexRankSummarizer(corpus={self.corpus}, model_path={self.model_path}, threshold={self.threshold}, alpha={self.alpha})"

    def __str__(self) -> str:
        return f"fastLexRankSummarizer(corpus={self.corpus}, model_path={self.model_path}, threshold={self.threshold}, alpha={self.alpha})"


if __name__ == "__main__":
    sentences = [
        "One of David Cameron's closest friends and Conservative allies, "
        "George Osborne rose rapidly after becoming MP for Tatton in 2001.",
        "Michael Howard promoted him from shadow chief secretary to the "
        "Treasury to shadow chancellor in May 2005, at the age of 34.",
        "Mr Osborne took a key role in the election campaign and has been at "
        "the forefront of the debate on how to deal with the recession and "
        "the UK's spending deficit.",
        "Even before Mr Cameron became leader the two were being likened to "
        "Labour's Blair/Brown duo. The two have emulated them by becoming "
        "prime minister and chancellor, but will want to avoid the spats.",
        "Before entering Parliament, he was a special adviser in the "
        "agriculture department when the Tories were in government and later "
        "served as political secretary to William Hague.",
        "The BBC understands that as chancellor, Mr Osborne, along with the "
        "Treasury will retain responsibility for overseeing banks and "
        "financial regulation.",
        "Mr Osborne said the coalition government was planning to change the "
        'tax system "to make it fairer for people on low and middle '
        'incomes", and undertake "long-term structural reform" of the '
        "banking sector, education and the welfare state.",
    ]
    summarizer = FastLexRankSummarizer()
    print(summarizer.summarize(sentences))
