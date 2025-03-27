#coding=utf8
import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Domain():

    def __init__(self):
        super(Domain, self).__init__()
        self.dataset = None
        self.denotation = False

    @classmethod
    def from_dataset(self, dataset):
        from .domain_overnight import OvernightDomain
        return OvernightDomain(dataset)

    def compare_logical_form(self, predictions, references):
        """
            predictions and references should be list of token list
        """
        predictions = self.normalize(predictions)
        references = self.normalize(references)
        if self.denotation:
            predictions = self.obtain_denotations(predictions)
            references = self.obtain_denotations(references)
            assert len(predictions) == len(references)
        return list(map(lambda x, y: 1.0 if x == y and self.is_valid(x) and self.is_valid(y) else 0.0, predictions, references))

    def execute_logical_forms(self, logical_forms):
        """
        Convert logical forms to denotations
        """
        normalized_logical_forms = self.normalize(logical_forms)
        denotations = self.obtain_denotations(normalized_logical_forms)
        return denotations

    def normalize(self, lf_list):
        """
            Normalize each logical form, at least changes token list into string list
        """
        raise NotImplementedError

    def obtain_denotations(self, lf_list):
        """
            Obtain denotations for each logical form
        """
        raise NotImplementedError

    def is_valid(self, ans):
        """
            Check whether ans is syntax or semantic invalid
            ans_list(str list): denotation list or logical form list
        """
        raise NotImplementedError
