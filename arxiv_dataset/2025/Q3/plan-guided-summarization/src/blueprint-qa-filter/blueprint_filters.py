from argparse import ArgumentParser
import spacy
import jsonlines
from tqdm import tqdm
from qa_model import QuestionAnsweringModel

split_words = [
    ',', '.', ';',  # Punctuation
    'for', 'and', 'nor', 'but', 'or', 'yet', 'so',  # Coordination
    'who', 'whom', 'which', 'what', 'that', 'whose', # Relative pronouns 
    'whoever', 'whomever', 'whichever', 'whatever', 
    'from', 'at', 'with', 'since', 'on', 'by', # Prepositions
    'beside', 'under', 'below', 'over', 'above', 'across', 
    'through', 'into', 'towards', 'onto', 'before', 'until'
]

PROPOSITION_MIN_LEN = 3

class BlueprintQAFilter:
    def __init__(self, model_dir = "./qa_model", cuda_device=-1, qa_batch_size = 8):
        self.qa_model = QuestionAnsweringModel(
            model_dir = model_dir,
            cuda_device = cuda_device,
            batch_size = qa_batch_size,
        )

        self.spacy_nlp = spacy.load("en_core_web_sm")

    def round_trip_filter(self, summary, questions, selected_answers):
        """Round trip filter"""

        qa_input_data = [(q, summary) for q in questions]
        qa_answers = self.qa_model.answer_all(qa_input_data)
        generated_answers = [ans[0] for ans in qa_answers]

        returned_questions = []
        returned_answers = []
        for q, s_ans, g_ans in zip(questions, selected_answers, generated_answers):
            if s_ans == g_ans or s_ans in g_ans or g_ans in s_ans:
                returned_questions.append(q)
                returned_answers.append(s_ans)
        return returned_questions, returned_answers

    def _sentence_to_propositions(self, sentence_tokens):
        if sentence_tokens[-1].text == '.':
            sentence_tokens = sentence_tokens[:-1]  # We already split periods off earlier.
        curr_idx = 0
        last_split_idx = 0
        propositions = []
        curr_proposition = []
        while curr_idx < len(sentence_tokens):
            curr_token = sentence_tokens[curr_idx].text
            if (curr_token in split_words and # Check if split word
                curr_idx - last_split_idx - 1 >= PROPOSITION_MIN_LEN and # Check left bound
                len(sentence_tokens) - curr_idx - 1 >= PROPOSITION_MIN_LEN): # Check right bound            
                propositions.append(curr_proposition)
                curr_proposition = []
                last_split_idx = curr_idx
            else:
                curr_proposition.append(curr_token)
            curr_idx += 1
        
        if curr_proposition:
            propositions.append(curr_proposition)
        return propositions

    def _summary_to_propositions(self, summary):
        doc = self.spacy_nlp(summary)
        propositions = []
        for sent in doc.sents:
            propositions += self._sentence_to_propositions(sent)
        # print(f"Propositions are: {propositions}")
        return propositions

    def _is_right_identical(self, proposition, answer):
        if len(answer) > len(proposition):
            return False

        rev_prop = proposition[::-1]
        rev_answer = answer[::-1]
        for i in range(len(rev_answer)):
            if rev_answer[i] != rev_prop[i]:
                return False
        return True
    
    def _find_rheme_qa(self, proposition_tokens, questions, answers):
        """Find rheme question and answer pair for a proposition."""
        return_q, return_a = None, None
        best_answer_len = -1
        answers_tokens = [self._text_to_tokens(answer) for answer in answers]
        for question, answer, answer_tokens in zip(questions, answers, answers_tokens):
            if self._is_right_identical(proposition_tokens, answer_tokens) and len(answer_tokens) > best_answer_len:
                return_q = question
                return_a = answer
                best_answer_len = len(answer)
        return return_q, return_a

    def rheme_filter(self, summary, questions, answers):
        """Rheme filter"""
        propositions = self._summary_to_propositions(summary)
        returned_questions = []
        returned_answers = []
        for prop in propositions:
            question, answer = self._find_rheme_qa(prop, questions, answers)
            if question:
                returned_questions.append(question)
                returned_answers.append(answer)
        return returned_questions, returned_answers, propositions

    def _text_to_tokens(self, text):
        doc = self.spacy_nlp(text)
        return [token.text for token in doc]

    def coverage_filter(self, summary, questions, answers):
        """Coverage filter"""

        # Convert summary and QA pairs into bags of tokens
        summary_token_set = set(self._text_to_tokens(summary))
        qa_token_sets = {}
        for i, (question, answer) in enumerate(zip(questions, answers)):
            token_set = set(self._text_to_tokens(question) + self._text_to_tokens(answer))
            qa_token_sets[i] = token_set

        remaining_summary_token_set = summary_token_set
        returned_questions = []
        returned_answers = []
        while len(remaining_summary_token_set) > 0 and len(qa_token_sets) > 0:
            # Get max overlap qa pair
            max_overlap_idx, top_qa_set = max(qa_token_sets.items(), key=lambda x: len(remaining_summary_token_set.intersection(x[1])))
            overlap_score = len(remaining_summary_token_set.intersection(top_qa_set))

            if overlap_score > 0:
                returned_questions.append((max_overlap_idx, questions[max_overlap_idx]))
                returned_answers.append((max_overlap_idx, answers[max_overlap_idx]))
                remaining_summary_token_set.difference_update(top_qa_set)
                qa_token_sets.pop(max_overlap_idx)
            else:
                break
        
        returned_questions.sort(key=lambda x: x[0])
        returned_answers.sort(key=lambda x: x[0])
        returned_questions = [q for _, q in returned_questions]
        returned_answers = [a for _, a in returned_answers]
        return returned_questions, returned_answers

    def _format_plan(self, questions, answers):
        qa_pairs = []
        for question, answer in zip(questions, answers):
            qa_pairs.append(f"Q: {question} A: {answer}")
        return " ".join(qa_pairs)

    def run_blueprint_qa_filter(self, summary, questions, answers):
        """Given summary, quesitons and answers, run the three filters from Blueprint QA: Round trip, Rheme and Coverage."""
        rt_questions, rt_answers = self.round_trip_filter(summary, questions, answers)
        rh_questions, rh_answers, _ = self.rheme_filter(summary, rt_questions, rt_answers)
        cv_questions, cv_answers = self.coverage_filter(summary, rh_questions, rh_answers)

        plan = self._format_plan(cv_questions, cv_answers)
        return plan


def run_all_examples(input_qa_jsonl_file, output_jsonl_file, qa_model_dir, cuda_device, qa_batch_size,
                     idx_key, document_key, summary_key, questions_key, answers_key):
    print("Loading data.")
    examples = []
    with jsonlines.open(input_qa_jsonl_file, 'r') as f:
        for line in f:
            examples.append(line)

    print("Setting up QA Model.")
    blueprint_qa_filter = BlueprintQAFilter(qa_model_dir, cuda_device, qa_batch_size)

    print("Start filtering.")
    docs_with_blueprint_plan = []
    for example in tqdm(examples):
        idx = example[idx_key]
        document = example[document_key]
        summary = example[summary_key]
        questions = example[questions_key]
        answers = example[answers_key]

        try:
            blueprint_qa_plan = blueprint_qa_filter.run_blueprint_qa_filter(summary, questions, answers)
        except Exception as e:
            raise Exception(f"Crashed while processing document {idx}") from e
        
        docs_with_blueprint_plan.append({
            'idx': idx,
            'document': document,
            'summary': summary,
            'plan': blueprint_qa_plan,
        })

    print("Writing output to file.")
    with jsonlines.open(output_jsonl_file, 'w') as f:
        for line in docs_with_blueprint_plan:
            f.write(line)
    return

 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_qa_jsonl_file", type=str, required=True, help="Path to jsonl file with the summary and QA data.")
    parser.add_argument("--output_jsonl_file", type=str, required=True, help="Output path ending in jsonl.")
    parser.add_argument("--qa_model_dir", type=str, default="./qa_model", help="Path to model directory for the QA model (for Roundtrip Consistency)")
    parser.add_argument("--cuda_device", type=int, default=-1)
    parser.add_argument("--qa_batch_size", type=int, default=8)

    parser.add_argument("--idx_key", type=str, default='idx', help="Document ID key")
    parser.add_argument("--document_key", type=str, default='document', help="Dataset document key")
    parser.add_argument("--summary_key", type=str, default='summary', help="Dataset summary key")
    parser.add_argument("--questions_key", type=str, default='questions', help="Dataset questions key")
    parser.add_argument("--answers_key", type=str, default="answers", help="Dataset answers key")
    args = parser.parse_args()

    run_all_examples(**vars(args))
