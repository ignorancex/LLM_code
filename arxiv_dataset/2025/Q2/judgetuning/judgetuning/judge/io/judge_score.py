import json
import re
from typing import Dict

import numpy as np


def get_regexp_match(s: str, regex: str, group_index: int = 1):
    m = re.search(re.compile(regex), s)
    if m is None:
        return None
    else:
        return float(m.group(group_index).strip(" "))


class JudgeScore:
    """
    A Judge score encapsulate a strategy to prompt a judge LLM for a preference and parse its output.
    The strategies supported are "best-model-identifier", "likert", "preference", "pair", "multi".
    The first two corresponds to Alpaca-Eval and Arena-hard choices.
    """

    def parse(self, judge_completion: str, json_output: bool) -> float | None:
        if json_output:
            json_dict = json.loads(judge_completion)
            return self.parse_model_json(json_dict)
        else:
            return self.parse_model_raw(judge_completion)

    def parse_model_json(self, json_dict: dict) -> float | None:
        raise NotImplementedError()

    def parse_model_raw(self, judge_completion: str) -> float | None:
        raise NotImplementedError()

    def preference_to_dict(self, preference: float, json_output: bool) -> dict:
        assert 0 <= preference <= 1
        return self._preference_to_dict(preference, json_output)

    def _preference_to_dict(self, preference: float, json_output: bool) -> dict:
        raise NotImplementedError()

    def json_example(self) -> Dict[str, str]:
        raise NotImplementedError()

    def score_description(self) -> str | None:
        return None

    def raw_description(self) -> str:
        raise NotImplementedError()

    @classmethod
    def from_str(cls, score_type: str, **kwargs):
        match score_type:
            case "likert":
                return LikertScore(**kwargs)
            case "preference":
                return PreferenceScore(**kwargs)
            case "best-model-identifier":
                return BestModelIdentifierScore(**kwargs)
            case "pair":
                return PairScore(**kwargs)
            case "multi":
                return MultiCriteriaScore(**kwargs)
            case _:
                raise NotImplementedError(score_type)


class PairScore(JudgeScore):
    def __init__(self):
        super(PairScore).__init__()
        self.temperature = 0.3
        self.score_a_keys = ["score_A", "score A", "scoreA"]
        self.score_b_keys = ["score_B", "score B", "scoreB"]

    def parse_model_json(self, json_dict: dict) -> float | None:
        score_a = None
        for key in self.score_a_keys:
            if key in json_dict:
                score_a = json_dict[key]
        score_b = None
        for key in self.score_b_keys:
            if key in json_dict:
                score_b = json_dict[key]
        try:
            score_a = float(score_a)
            score_b = float(score_b)
        except TypeError:
            return None
        if score_a is None or score_b is None:
            return None
        return self.preference_from_scores(score_a, score_b)

    def preference_from_scores(self, score_a: float, score_b: float) -> float:
        return 1 - np.exp(self.temperature * score_a) / (
            np.exp(self.temperature * np.array([score_a, score_b])).sum()
        )

    def parse_model_raw(self, judge_completion: str) -> float | None:
        # lower case to avoid confusion, e.g. when "a" is used instead of "A"
        score_a = get_regexp_match(
            judge_completion.lower(), r'score[ _]*a[": *\n]*(-?\d+)'
        )
        score_b = get_regexp_match(
            judge_completion.lower(), r'score[ _]*b[": *\n]*(-?\d+)'
        )
        if score_a is None or score_b is None:
            return None
        else:
            return self.preference_from_scores(score_a, score_b)

    # def score_description(self) -> str | None:
    #     return None #f'The keys "{self.score_a_keys[0]}" and "{self.score_b_keys[0]}" should be the score for model A and B between 0 and 10.'

    def json_example(self) -> Dict[str, str]:
        return {
            self.score_a_keys[
                0
            ]: "<a number between 0 and 10 to indicate the quality of Assistant A's answer>",
            self.score_b_keys[
                0
            ]: "<a number between 0 and 10 to indicate the quality of Assistant B's answer>",
        }

    def _preference_to_dict(self, preference: float, json_output: bool) -> dict:
        # TODO would need to inverse softmax...
        return {
            self.score_a_keys[0]: 2,
            self.score_b_keys[0]: 8,
        }


class BestModelIdentifierScore(JudgeScore):
    def __init__(self, first_model: str = "A", second_model: str = "B"):
        super(BestModelIdentifierScore).__init__()
        self.best_model_str = "## Best Assistant Identifier"
        self.best_model_key = "best_assistant"
        self.first_model = first_model
        self.second_model = second_model

    def _preference_to_dict(self, preference: float, json_output: bool) -> dict:
        return {
            self.best_model_key: (
                self.first_model if preference < 0.5 else self.second_model
            )
        }

    def parse_model_json(self, json_dict: dict) -> float | None:
        best_model = json_dict.get(self.best_model_key)
        if best_model is None:
            for key, value in json_dict.items():
                if "best_" in key:
                    best_model = value
        if best_model is None:
            return None
        else:
            if self.first_model in best_model:
                return 0
            elif self.second_model in best_model:
                return 1
            else:
                return None

    def parse_model_raw(self, judge_completion: str) -> float | None:
        judge_completion = judge_completion.split(self.best_model_str)[-1]
        if self.first_model in judge_completion:
            return 0
        elif self.second_model in judge_completion:
            return 1
        else:
            return 0.5
            return None

    def json_example(self) -> Dict[str, str]:
        return {
            self.best_model_key: '<either "A" or "B" to indicate which Assistant gave the best answer>',
        }

    def score_description(self) -> str:
        return f'For the best assistant, use only "{self.first_model}" or "{self.second_model}".'

    def raw_description(self) -> str:
        return f"Finally to provide your assessment to indicate which assistant was better, you should answer either {self.best_model_key}={self.first_model} or {self.best_model_key}={self.second_model}."


class PreferenceScore(JudgeScore):
    def __init__(self):
        super(PreferenceScore).__init__()
        self.score_key = "preference"

    def parse_model_json(self, json_dict: dict) -> float | None:
        if self.score_key in json_dict:
            return float(json_dict[self.score_key])
        else:
            return None

    def parse_model_raw(self, judge_completion: str) -> float | None:
        # regex = re.compile(r"[p|P]reference=([+-]?([0-9]*[.])?[0-9]+)")
        # regex = re.compile(r"([+-]?([0-9]*[.])?[0-9]+)")
        return get_regexp_match(
            judge_completion.lower(),
            rf'\*?{self.score_key.lower()}\*?[":= \n]*(([0-9]*[.])?[0-9]+)',
            group_index=1,
        )

    def json_example(self) -> Dict[str, str]:
        return {
            self.score_key: "<a number between [0, 1] to indicate your preference between assistant A and B in [0, 1], "
            "0 indicates a preference for A while 1 indicates a preference for assistant B>",
        }

    # def score_description(self) -> str:
    #     return f'The key "{self.score_key}" should indicates your preference between model A and B in [0, 1], 0 indicates a preference for A while 1 indicates a preference for model B.'

    def _preference_to_dict(self, preference: float, json_output: bool) -> dict:
        return {self.score_key: preference}


def parse_likert(likert_str: str, alpha: float = 0.25):
    assert 0 <= alpha < 0.5
    mapping = {
        "A>>B": 0,
        "B<<A": 0,
        "A>B": alpha,
        "B<A": alpha,
        "B>>A": 1,
        "A<<B": 1,
        "B>A": 1 - alpha,
        "A<B": 1 - alpha,
        "A=B": 0.5,
        "B=A": 0.5,
    }
    # # we assign a tie in case the score was not properly formatted, this could silent errors though
    if likert_str not in mapping:
        raise ValueError(likert_str)
    return mapping.get(likert_str, 0.5)


class LikertScore(JudgeScore):
    def __init__(self):
        super(LikertScore).__init__()
        self.score_key = "score"

    def parse_model_json(self, json_dict: dict) -> float | None:
        return parse_likert(json_dict[self.score_key])

    def parse_model_raw(self, judge_completion: str) -> float | None:
        regex = re.compile(r"([A|B][><=]+[B|A])")
        # Upper case to match cases where LLM output is B>>a
        m = re.search(regex, judge_completion.upper())
        if m is None:
            # return 0.5
            # raise ValueError(f"score not found in judge completion {judge_completion}")
            return None
        else:
            return parse_likert(m.group(0))

    def json_example(self) -> Dict[str, str]:
        return {
            self.score_key: "<one of A>>B, A>B, A=B, A<B, A<<B, see instruction bellow>",
        }

    def score_description(self) -> str:
        labels = [
            ("Assistant A is significantly better", "A>>B"),
            ("Assistant A is slightly better", "A>B"),
            ("Tie, relatively the same", "A=B"),
            ("Assistant B is significantly better", "B>A"),
            ("Assistant B is significantly better", "B>>A"),
        ]
        label_explanation = "\n".join(
            f"{label}: {explanation}" for explanation, label in labels
        )
        return (
            f'The "{self.score_key}" value should indicate your preference for the assistant. '
            f"You must output only one of the following choices as your final verdict with a label:\n\n{label_explanation}"
        )

    def _preference_to_dict(self, preference: float, json_output: bool) -> dict:
        return {self.score_key: "B>>A"}


class MultiCriteriaScore(JudgeScore):
    def __init__(self):
        self.score_keys_to_description = {
            "conciseness": "conciseness",
            "clarity": "clarity",
            "adherence": "adherence to instructions",
            "comprehensiveness": "comprehensiveness",
            "style": "style",
        }

    def json_example(self) -> Dict[str, str]:
        return {
            key: f"[one of A>>B, A>B, A=B, A<B, A<<B]"
            for key, value in self.score_keys_to_description.items()
        }

    def score_description(self) -> str:
        labels = [
            ("Assistant A is significantly better", "A>>B"),
            ("Assistant A is slightly better", "A>B"),
            ("Tie, relatively the same", "A=B"),
            ("Assistant B is significantly better", "B>A"),
            ("Assistant B is significantly better", "B>>A"),
        ]
        label_explanation = "\n".join(
            f"{label}: {explanation}" for explanation, label in labels
        )
        return (
            f"The score key should indicate your preference between the two assistants A and B for each of the 5 categories. "
            f"You must output only one of the following choices as your final verdict with a label:\n\n{label_explanation}"
        )

    def _preference_to_dict(self, preference: float, json_output: bool) -> dict:
        """
        instruction="What is the square root of 81? Just provide the answer.",
        output1="The answer is 81, this can be seen as 9*9 = 81.",
        output2="81",
        """
        if preference == 1.0:
            return {
                "conciseness": "B>>A",
                "clarity": "A=B",
                "adherence": "B>>A",
                "comprehensiveness": "A=B",
                "style": "A=B",
            }

    def parse_liker_str(self, s: str) -> float | None:
        regex = re.compile(r"([A|B][><=]+[B|A])")
        # Upper case to match cases where LLM output is B>>a
        m = re.search(regex, s.upper())
        if m is None:
            # return 0.5
            # raise ValueError(f"score not found in judge completion {judge_completion}")
            return None
        else:
            return parse_likert(m.group(0))

    def parse_model_json_all(self, json_dict: dict) -> list[float | None]:
        return [self.parse_liker_str(value) for key, value in json_dict.items()]

    def parse_model_json(self, json_dict: dict) -> float | None:
        return self.average_criterion(self.parse_model_json_all(json_dict))

    def average_criterion(self, criteria):
        criteria = [x for x in criteria if x is not None]
        if len(criteria) == 0:
            return 0.5
        else:
            return np.mean(criteria)

    def parse_model_raw_all(self, judge_completion: str) -> list[float | None]:
        regex = re.compile(r"([A|B][><=]+[B|A])")
        m = re.findall(regex, judge_completion.upper())
        return [parse_likert(x) for x in m]

    def parse_model_raw(self, judge_completion: str) -> float | None:
        return self.average_criterion(self.parse_model_raw_all(judge_completion))


if __name__ == "__main__":

    for cls in [LikertScore, MultiCriteriaScore]:
        print("\n{}".format(cls))
        x = cls()
        print(x.json_example())
        print(x.score_description())
