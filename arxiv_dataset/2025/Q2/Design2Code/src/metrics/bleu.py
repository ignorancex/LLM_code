from typing import Any, Set, Tuple, List
import nltk, os

from lxml import etree
from bs4 import BeautifulSoup
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .base_metrics import Metrics

import networkx as nx
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "html_keywords.txt"), 'r') as file:
    important_keywords = [line.strip() for line in file if line.strip()]

class BLEU(Metrics):
    def __call__(self, reference: str, hypothesis: str) -> float:
        reference_tokens = nltk.word_tokenize(reference)
        hypothesis_tokens = nltk.word_tokenize(hypothesis)
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method1)


class WeightedBLEU(Metrics):
    def __init__(
        self, 
        important_keywords: List[str] = important_keywords, 
        weight: int = 2
    ):
        super().__init__()
        self.important_keywords = important_keywords
        self.weight = weight

    def _weight_tokens(self, tokens: List[str]) -> List[str]:
        weighted = []
        for token in tokens:
            if token in self.important_keywords:
                weighted.extend([token] * self.weight)
            else:
                weighted.append(token)

        return weighted

    def __call__(self, reference: str, hypothesis: str) -> float:
        ref_tokens = nltk.word_tokenize(reference)
        hyp_tokens = nltk.word_tokenize(hypothesis)

        weighted_ref = self._weight_tokens(ref_tokens)
        weighted_hyp = self._weight_tokens(hyp_tokens)
        
        return sentence_bleu([weighted_ref], weighted_hyp, smoothing_function=SmoothingFunction().method1)

        # return sentence_bleu([weighted_ref], hyp_tokens, weighted_hyp, smoothing_function=SmoothingFunction().method1)

class HTMLBLEU(Metrics):
    def __init__(
            self, 
            important_keywords: List[str] = important_keywords,
            alpha: float = 0.25, beta: float = 0.25, gamma: float = 0.25, delta: float = 0.25
        ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bleu = BLEU()
        self.weighted_bleu = WeightedBLEU(important_keywords)

    def __call__(self, reference: str, hypothesis: str) -> float:
        bleu_score = self.bleu(reference, hypothesis)
        weighted_bleu_score = self.weighted_bleu(reference, hypothesis)
        
        ref_tree = self._parse_html(reference)
        # self.print_dom_tree(ref_tree)
        # self._visualize_tree(ref_tree)
        hyp_tree = self._parse_html(hypothesis)
        
        dom_match = self._calculate_dom_match(ref_tree, hyp_tree)
        attr_match = self._calculate_attribute_match(ref_tree, hyp_tree)
        
        # print(bleu_score, weighted_bleu_score, dom_match, attr_match)

        return (self.alpha * bleu_score +
                self.beta * weighted_bleu_score +
                self.gamma * dom_match +
                self.delta * attr_match)

    def _parse_html(self, html_string: str) -> etree.Element:
        parser = etree.HTMLParser()
        return etree.fromstring(html_string, parser)

    def _calculate_dom_match(self, reference_tree: etree.Element, hypothesis_tree: etree.Element) -> float:
        ref_elements = reference_tree.xpath('//*')
        hyp_elements = hypothesis_tree.xpath('//*')
        
        common = set(el.tag for el in ref_elements) & set(el.tag for el in hyp_elements)
        return len(common) / max(len(ref_elements), len(hyp_elements))

    def _calculate_attribute_match(self, reference_tree: etree.Element, hypothesis_tree: etree.Element) -> float:
        ref_attrs = set()
        hyp_attrs = set()
        
        for el in reference_tree.xpath('//*'):
            ref_attrs.update((el.tag, attr) for attr in el.attrib)
        
        for el in hypothesis_tree.xpath('//*'):
            hyp_attrs.update((el.tag, attr) for attr in el.attrib)
        
        common = ref_attrs & hyp_attrs
        return len(common) / max(len(ref_attrs), len(hyp_attrs))
    
class TreeBLEU(Metrics):
    def __call__(self, reference: str, hypothesis: str) -> float:
        """
        Computes TreeBLEU score by comparing the DOM tree structure of the reference 
        and hypothesis HTML strings.

        Args:
            reference (str): The reference HTML string.
            hypothesis (str): The hypothesis HTML string.

        Returns:
            float: The computed TreeBLEU score.
        """
        ref_subtrees = self.extract_1_height_subtrees(reference)
        hyp_subtrees = self.extract_1_height_subtrees(hypothesis)

        matching_subtrees = len(ref_subtrees.intersection(hyp_subtrees))
        total_subtrees = len(ref_subtrees)

        return matching_subtrees / total_subtrees if total_subtrees > 0 else 0.0

    def extract_1_height_subtrees(self, html: str) -> Set[Tuple[str, ...]]:
        """
        A helper method to extract 1-height subtrees from the HTML string.

        Args:
            html (str): The HTML string.

        Returns:
            Set[Tuple[str, ...]]: A set of 1-height subtrees.
        """
        subtrees = set()
        soup = BeautifulSoup(html, 'html.parser')

        for element in soup.find_all():
            # Create a tuple representing the parent and its immediate children
            subtree = (element.name,) + tuple(child.name for child in element.children if child.name)
            if len(subtree) > 1:  # Only add if there's at least one child
                subtrees.add(subtree)

        return subtrees


if __name__ == "__main__":
    html_keywords = ["div", "span", "p", "a", "img", "ul", "li"]

    bleu = BLEU()
    html_bleu = HTMLBLEU(html_keywords)
    tree_bleu = TreeBLEU()

    reference_code = "def hello_world():\n    print('Hello, World!')"
    hypothesis_code = "def greet():\n    print('Hello, World!')"


    with open("./testcases/hypothesis.html") as f:
        reference_html = f.read()
    # reference_html = "<div><p>Hello, <a href='#'>World</a>!</p></div>"
    
        
    hypothesis_html = "<div><p>Hello, <span>World</span>!</p></div>"

    print(f"BLEU: {bleu(reference_html, hypothesis_html)}")
    print(f"HTMLBLEU: {html_bleu(reference_html, hypothesis_html)}")
    print(f"TreeBLEU: {tree_bleu(reference_html, hypothesis_html)}")