import collections
from typing import Dict, List, Tuple, Union
from transformers.data.metrics.squad_metrics import (
    get_final_text,
    _get_best_indexes,
    _compute_softmax,
)

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadResult, SquadExample

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

Prediction = Union[
    Tuple[str, float, float],
    Tuple[str, float, float, Tuple[int, int]],
    Dict[str, Union[str, float, Tuple[int, int]]],
]

def compute_predictions_logits_with_null(
        tokenizer,
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        verbose_logging,
        version_2_with_negative,
        return_offsets = False
):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    all_probs = collections.OrderedDict()
    null_scores = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "doc_start", "doc_end"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                orig_doc_start = None
                orig_doc_end = None
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,
                                          doc_start=orig_doc_start, doc_end=orig_doc_end))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit,
                                              doc_start=None, doc_end=None))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, doc_start=None,
                                                 doc_end=None))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, doc_start=None,
                                          doc_end=None))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        best_non_null_entry_index = None
        for i, entry in enumerate(nbest):
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
                    best_non_null_entry_index = i

        probs = _compute_softmax(total_scores)

        nbest_json = []
        null_prob = None
        best_prob = None
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            if entry.text == '':
                null_prob = probs[i]
            if i == best_non_null_entry_index:
                best_prob = probs[i]
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # Always predict the best non-null text
            all_predictions[example.qas_id] = best_non_null_entry.text
            all_probs[example.qas_id] = best_prob
            null_scores[example.qas_id] = null_prob

            # # predict "" iff the null score - the score of best non-null > threshold
            # score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            # scores_diff_json[example.qas_id] = score_diff
            # if score_diff > null_score_diff_threshold:
            #     all_predictions[example.qas_id] = ""
            # else:
            #     all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    output = (all_predictions, all_probs, null_scores)
    return output



class QuestionAnsweringModel(object):
    def __init__(self,
                 model_dir: str,
                 cuda_device: int = 0,
                 batch_size: int = 8,
                 silent: bool = True) -> None:
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=True, use_fast=False)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir, config=self.config)
        if cuda_device >= 0:
            self.model.to(cuda_device)

        self.model_type = 'electra'
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.max_seq_length = 384
        self.doc_stride = 128
        self.silent = silent

    def _to_list(self, qa_output):
        start_logits = qa_output.start_logits.detach().cpu().tolist()
        end_logits = qa_output.end_logits.detach().cpu().tolist()
        return start_logits, end_logits

    def answer(
        self,
        question: str,
        context: str,
        return_offsets: bool = False,
        try_fixing_offsets: bool = False,
        return_dict: bool = False,
    ) -> Prediction:
        """
        Returns a tuple of (prediction, probability, null_probability). If `return_offsets = True`, the tuple
        will include rough character offsets of where the prediction is in the context. Because the tokenizer that
        the QA model uses does not support returning the character offsets from the BERT tokenization, we cannot
        directly provide exactly where the answer came from. However, the offsets should be pretty close to the
        prediction, and the prediction should be a substring of the offsets (modulo whitespace). If
        `return_offsets` and `try_fixing_offsets` are `True`, we will try to fix the character offsets via
        an alignment. See below.

        The `SquadExample` class maintains a list of whitespace separated tokens `doc_tokens` and a mapping
        from the context string characters to the token indices `char_to_word_offset`. Whitespace
        is included in the previous token. The `squad_convert_example_to_features` method takes each of these
        tokens and breaks it into the subtokens with the transformers tokenizer, which are passed into the model.
        It also keeps a mapping from the subtokens to the `doc_tokens` called `tok_to_orig_index`. The QA model
        predicts a span in the subtokens. In the `_get_char_offsets` method, we use these data structures to map
        from the subtoken span to character offsets. However, we cannot separate subtokens, so they are merged together.
        See the below example

            context: " My name is  Dan!"
            doc_tokens: [My, name, is, Dan!]
            char_to_word_offset: [-1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            subtokens: [My, name, is, Dan, ##!]
            tok_to_orig_index: [0, 1, 2, 3, 3]

            prediction: "name is Dan"
            prediction subtokens: [name, is, Dan]
            prediction in doc_tokens: [name, is, Dan!]
            prediction in context: "name is  Dan!"

        The prediction includes the extra whitespace between "is" and "Dan" as well as the "!"

        If `try_fixing_offsets=True`, we will try to fix the character offsets to be correct based on an alignment
        algorithm. We use the `edlib` python package to create a character alignment between the actual prediction
        string and the span given by the original offsets. We then update the offsets based on the alignment. If
        this procedure fails, the original offsets will be returned.
        """
        return self.answer_all(
            [(question, context)], return_offsets=return_offsets,
            try_fixing_offsets=try_fixing_offsets, return_dicts=return_dict
        )[0]

    def answer_all(
        self,
        input_data: List[Tuple[str, str]],
        return_offsets: bool = False,
        try_fixing_offsets: bool = True,
        return_dicts: bool = False,
    ) -> List[Prediction]:
        # Convert all of the instances to squad examples
        examples = []
        for i, (question, context) in enumerate(input_data):
            examples.append(SquadExample(
                qas_id=str(i),
                question_text=question,
                context_text=context,
                answer_text=None,
                start_position_character=None,
                title=None,
                is_impossible=True,
                answers=[]
            ))

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=64,
            tqdm_enabled=not self.silent
        )

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.batch_size)

        self.model.eval()
        all_results = []
        generator = eval_dataloader
        if not self.silent:
            generator = tqdm(generator, desc='Evaluating')

        for batch in generator:
            if self.cuda_device >= 0:
                batch = tuple(t.to(self.cuda_device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                feature_indices = batch[3]
                outputs = self.model(**inputs)
                outputs = self._to_list(outputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                # output = [self._to_list(output[i]) for output in outputs]
                start_logits, end_logits = outputs[0][i], outputs[1][i]
                # print(start_logits)
                # print(end_logits)
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        model_predictions = compute_predictions_logits_with_null(
            self.tokenizer,
            examples,
            features,
            all_results,
            20,
            30,
            True,
            False,
            True,
            return_offsets=return_offsets
        )

        predictions, prediction_probs, no_answer_probs = model_predictions

        results = []
        for i in range(len(input_data)):
            i = str(i)
            r = (predictions[i], prediction_probs[i], no_answer_probs[i])
            if return_dicts:
                r = {
                    'prediction': r[0],
                    'probability': r[1],
                    'null_probability': r[2],
                }
            results.append(r)
        return results

if __name__ == "__main__":
    qa_model = QuestionAnsweringModel(
        model_dir = "./qa_model",
        cuda_device = -1,
        batch_size = 8,
    )

    qa_model.answer_all(
        input_data=[
            ("What is my name?", "My name is Dan."),
            ("What is his name?", "His name is Pat.")
        ]
    )
