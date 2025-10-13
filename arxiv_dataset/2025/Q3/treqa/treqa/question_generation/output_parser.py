import re
from abc import ABC


class OutputParser(ABC):
    def __init__(
        self,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    ):

        self.model_name = model_name
        if self.model_name == "aya":
            self.parse = self.parse_output_aya
        else:
            self.parse = self.parse_output_default

    def check_answer_overlap(self, answer, target):
        # Generate character n-grams for a given string
        def get_char_ngrams(s, n):
            s = s.replace(" ", "")  # Remove spaces for pure character-based comparison
            return [s[i : i + n] for i in range(len(s) - n + 1)]

        # Calculate recall
        def recall_chrf(str1, str2, n):
            # Get n-grams for both strings
            ngrams1 = get_char_ngrams(str1, n)
            ngrams2 = get_char_ngrams(str2, n)

            # Calculate common n-grams
            common_ngrams = set(ngrams1) & set(ngrams2)

            # Calculate recall
            recall = len(common_ngrams) / len(ngrams2) if ngrams2 else 0

            return recall * 100  # Return as a percentage

        return recall_chrf(answer, target, n=6)

    def parse_output_default(
        self,
        output: str,
        target: str,
        answer_overlap_threshold: float,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        pairs = []
        skipped_outs = []
        for qa in output.split("\n\n"):
            # skip if there is no content
            if not qa.strip():
                reason = "no content"
                skipped_outs.append((reason, qa))
                continue
            # skip if there are not two lines
            if len(qa.split("\n")) != 2:
                reason = "not two lines"
                skipped_outs.append((reason, qa))
                continue

            q, a = qa.split("\n")
            # skip if the qa pair don't start with Q: and A:
            if not q.startswith("Q:") or not a.startswith("A:"):
                reason = "no Q: or A:"
                skipped_outs.append((reason, qa))
                continue

            q = q.replace("Q:", "").strip()
            a = a.replace("A:", "").strip()
            if answer_overlap_threshold is not None:
                # skip if the answer is not in the target and answer_overlap_threshold is
                # defined (For QE version the default value is None).
                if self.check_answer_overlap(target, a) < answer_overlap_threshold:
                    reason = "answer not in target"
                    skipped_outs.append((reason, qa))
                    continue

            pairs.append((q, a))

        return list(set(pairs)), skipped_outs

    def parse_output_aya(
        self,
        output: str,
        target: str,
        answer_overlap_threshold: float,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        pairs = []
        skipped_outs = []
        possible_answers = output.split("\n\n")[1:]

        for qa in possible_answers:
            # skip if there is no content
            if not qa.strip():
                reason = "no content"
                skipped_outs.append((reason, qa))
                continue
            # skip if there are not two lines
            if len(qa.split("\n")) != 2:
                reason = "not two lines"
                skipped_outs.append((reason, qa))
                continue

            q, a = qa.split("\n")
            matchq = re.search(r"\*\*(.*?)\*\*", q)
            matcha = re.search(r"\*\*(.*?)\*\*", a)
            if matchq:
                q = matchq.group(1)
            if matcha:
                a = matcha.group(1)

            q = q.strip()
            a = a.strip()

            # skip if the qa pair don't start with Q: and A:
            if not q.startswith("Q:") or not a.startswith("A:"):
                reason = "no Q: or A:"
                skipped_outs.append((reason, qa))
                continue

            q = q.replace("Q:", "").strip()
            a = a.replace("A:", "").strip()
            if answer_overlap_threshold is not None:
                # skip if the answer is not in the target and answer_overlap_threshold is
                # defined (For QE version the default value is None).
                if self.check_answer_overlap(target, a) < answer_overlap_threshold:
                    reason = "answer not in target"
                    skipped_outs.append((reason, qa))
                    continue

            pairs.append((q, a))

        return list(set(pairs)), skipped_outs
