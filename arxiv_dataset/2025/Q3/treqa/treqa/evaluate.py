"""Evaluate MT systems using QA-based evaluation."""

import argparse
import json

from treqa.scorers import METRICS_REGISTRY
from treqa.question_answering import QA_MODELS_REGISTRY
from treqa.answer_extraction import AE_MODELS_REGISTRY
from treqa.answer_matching import AM_MODELS_REGISTRY
from treqa.prompt_model import PromptModel


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp",
        type=str,
        required=True,
        help="Path to the file containing MT system outputs.",
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the file containing source translations.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Path to the file containing reference translations.",
    )

    # scorer arguments
    parser.add_argument(
        "--scorer",
        type=str,
        choices=METRICS_REGISTRY.keys(),
        default="treqa",
        help="Name of the scorer to use.",
    )
    parser.add_argument(
        "--scorer-init-args",
        type=json.loads,
        default=r"{}",
        help="Scorer arguments.",
    )

    # QAG-related arguments
    parser.add_argument(
        "--qa-file",
        type=str,
        default=None,
        help="Path to the file containing QA pairs. If not set, the model will generate QA pairs.",
    )

    # QA-related arguments
    parser.add_argument(
        "--qa-model",
        type=str,
        choices=QA_MODELS_REGISTRY.keys(),
        default="prompt_qa",
        help="Name of the model to use for question answering.",
    )
    parser.add_argument(
        "--qa-model-args",
        type=json.loads,
        default=r"{}",
        help="Arguments to pass to the question answering model, in the format of a JSON dictionary string.",
    )

    # AE-related arguments
    parser.add_argument(
        "--ae-model",
        type=str,
        choices=AE_MODELS_REGISTRY.keys(),
        default="spacy",
        help="Name of the model to use for answer/keyphrase extraction.",
    )
    parser.add_argument(
        "--ae-model-args",
        type=json.loads,
        default=r"{}",
        help="Arguments to pass to the answer extraction model, in the format of a JSON dictionary string.",
    )
    parser.add_argument(
        "--keyphrase-comparator",
        default="jaccard",
        choices=["jaccard"],
        help="Comparator to use for matching answers.",
    )

    # AM-related arguments
    parser.add_argument(
        "--am-model",
        type=str,
        choices=AM_MODELS_REGISTRY.keys(),
        default="quip",
        help="Name of the model to use for answer matching.",
    )
    parser.add_argument(
        "--am-model-args",
        type=json.loads,
        default=r"{}",
        help="Arguments to pass to the answer matching model, in the format of a JSON dictionary string.",
    )

    parser.add_argument(
        "--save-scores",
        type=str,
        default=None,
        help="Path to save the scores.",
    )
    parser.add_argument(
        "--save-detailed-evaluation",
        type=str,
        default=None,
        help="Path to save the full evaluation results.",
    )

    return parser.parse_args()


def load_qa_pairs(qa_file: str) -> list[list[dict[str, str]]]:
    with open(qa_file, "r") as f:
        return [json.loads(line.strip()) for line in f]


def main():
    args = read_args()
    with open(args.src, "r") as f:
        sources = f.readlines()
    with open(args.hyp, "r") as f:
        hypotheses = f.readlines()

    # TODO: is this the best way to handle optional arguments?
    init_kwargs = args.scorer_init_args
    score_kwargs = {}

    if args.ref is not None:
        with open(args.ref, "r") as f:
            references = f.readlines()
        score_kwargs["references"] = references

    # for treqa, we need to load QA pairs and QA model
    if args.scorer == "treqa" or args.scorer == "treqa_qe":
        if args.qa_file is None:
            raise ValueError(
                "On-the-fly QA generation is not supported yet. Please generate QA pairs first using treqa-generate."
            )

        # In case we have already computed QA pairs, load them
        qa_pairs = load_qa_pairs(args.qa_file)

        # Check if the number of QA pairs matches the number of MT outputs
        if len(qa_pairs) != len(hypotheses):
            raise ValueError(
                "The number of QA pairs does not match the number of MT outputs."
            )

        # Initialize QA model
        qa_model = QA_MODELS_REGISTRY[args.qa_model](**args.qa_model_args)
        init_kwargs["qa_model"] = qa_model
        if (
            args.am_model == "prompt_am"
            and args.am_model_args["provider"] == "parent_prompt_model"
            and isinstance(qa_model, PromptModel)
        ):
            args.am_model_args["parent_prompt_model"] = qa_model
        init_kwargs["answer_comparator"] = AM_MODELS_REGISTRY[args.am_model](
            **args.am_model_args
        )
        score_kwargs["qa_pairs"] = qa_pairs
        score_kwargs["return_detailed_evaluation"] = (
            args.save_detailed_evaluation is not None
        )

    elif args.scorer == "keyphrase":
        # Initialize AE model
        ae_model = AE_MODELS_REGISTRY[args.ae_model](**args.ae_model_args)
        init_kwargs["ae_model"] = ae_model
        init_kwargs["comparator"] = args.keyphrase_comparator

    scorer = METRICS_REGISTRY[args.scorer](**init_kwargs)

    # Compute scores
    outputs = scorer.get_scores(hypotheses, sources=sources, **score_kwargs)
    if not args.save_detailed_evaluation:
        segment_scores = outputs
    else:
        if args.scorer not in ["treqa", "treqa_qe"]:
            raise ValueError(
                f"Only TREQA metrics support saving detailed evaluation. {args.scorer} is not supported."
            )
        segment_scores, predicted_answers, reference_answers, per_q_scores = outputs

    # TODO: for now assume overall score is the mean of segment scores
    # this breaks for chrf, but lets assume it for now
    overall_score = sum(segment_scores) / len(segment_scores)

    print(f"{args.scorer}: {overall_score:.4f}")

    if args.save_scores:
        with open(args.save_scores, "w") as f:
            for score in segment_scores:
                print(score, file=f)
    if args.save_detailed_evaluation:
        # create jsonl with predicted_answers, reference_answers, per_q_scores
        with open(args.save_detailed_evaluation, "w") as f:
            for i, source in enumerate(sources):
                print(
                    json.dumps(
                        {
                            "predicted_answers": predicted_answers[i],
                            "reference_answers": reference_answers[i],
                            "per_q_scores": per_q_scores[i],
                        },
                    ),
                    file=f,
                )


if __name__ == "__main__":
    main()

# example command
# python treqa/evaluate.py --hyp-file data/test.en --qa-file data/test.en.jsonl --output-file data/test.en.scores.jsonl
