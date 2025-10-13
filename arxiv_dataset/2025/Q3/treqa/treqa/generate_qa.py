"""Given a source/target parallel corpus, generate questions for the target language, to be use for QA-based evaluation of MT systems."""

import argparse
import json

from treqa.prompt_model import PromptModel
from treqa.question_generation import (
    QG_MODELS_REGISTRY,
    BaseQAGModel,
    BaseQGModel,
)
from treqa.answer_extraction import AE_MODELS_REGISTRY


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file. It will save a list of QA pairs for each source/target pair, in .jsonl format.",
    )
    parser.add_argument(
        "--src", type=str, default=None, help="Path to the source texts file."
    )
    parser.add_argument(
        "--tgt", type=str, default=None, help="Path to the target texts file."
    )
    parser.add_argument(
        "--hyp", type=str, nargs="*", help="Path to (multiple) hypothesis texts file."
    )

    # QG-related arguments
    parser.add_argument(
        "--qg-model",
        type=str,
        choices=QG_MODELS_REGISTRY.keys(),
        default="prompt_qag",
        help="Name of the model to use for question generation.",
    )
    parser.add_argument(
        "--qg-model-args",
        type=json.loads,
        default=r"{}",
        help="Arguments to pass to the question generation model, in the format of a JSON dictionary string.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to generate.",
    )

    # AE-related arguments
    parser.add_argument(
        "--ae-model",
        type=str,
        choices=AE_MODELS_REGISTRY.keys(),
        default="spacy",
        help="Name of the model to use for answer extraction.",
    )
    parser.add_argument(
        "--ae-model-args",
        type=json.loads,
        default=r"{}",
        help="Arguments to pass to the answer extraction model, in the format of a JSON dictionary string.",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default=None,
        help="Path to the file containing the answers to the questions. If provided, the answers will be used to generate the questions",
    )
    parser.add_argument(
        "--num-answers",
        type=int,
        default=5,
        help="Number of answers to extract.",
    )

    args = parser.parse_args()

    if args.src is None and args.tgt is None:
        raise ValueError("Either --src or --tgt must be provided")

    return args


def main():
    args = read_args()

    # open files
    texts = None
    alt_texts = None
    hypotheses = None
    num_texts = None
    default_template = None
    if args.src is not None:
        with open(args.src, "r") as src_file:
            sources = [line.strip() for line in src_file.readlines()]
        texts = sources
        num_texts = len(sources)
        default_template = "eng-nocands"

    if args.tgt is not None:
        with open(args.tgt, "r") as tgt_file:
            targets = [line.strip() for line in tgt_file.readlines()]

        if texts is None:
            texts = targets
            num_texts = len(targets)
        else:
            assert (
                len(targets) == num_texts
            ), "Number of sources and targets must be the same"
            alt_texts = targets
        default_template = "eng-both-nocands-0shot"

    if args.hyp is not None and len(args.hyp) > 0:
        hypotheses_per_file = []
        for hyp_file in args.hyp:
            with open(hyp_file, "r") as hyp_file:
                hypotheses_per_file.append(
                    [line.strip() for line in hyp_file.readlines()]
                )
            assert (
                len(hypotheses_per_file[-1]) == num_texts
            ), "Number of hypotheses must be the same as the number of sources/targets"

        # make a list of list of hypotheses
        hypotheses = list(zip(*hypotheses_per_file))
        default_template = "eng-cands-0shot" if alt_texts is not None else "eng-cands"

    # pretty sure this is not the best way to do this:
    # if "BaseQAGModel" in qg_model.__class__.__base__.__name__:
    qg_model_args = args.qg_model_args.copy()
    if "template" not in qg_model_args:
        qg_model_args["template"] = default_template

    qg_model = QG_MODELS_REGISTRY[args.qg_model](**qg_model_args)

    if isinstance(qg_model, BaseQAGModel):
        # generate questions
        corpus_qa_pairs = qg_model.generate_qa_pairs(
            passages=texts,
            alt_passages=alt_texts,
            num_questions=args.num_questions,
            candidates=hypotheses,
        )
    else:  # in that case qg model is BaseQGModel or BaseQGQEModel
        if args.answers_file:
            print(f"Loading answers from {args.answers_file}")
            with open(args.answers_file, "r") as answers_file:
                answers = [
                    json.loads(line.strip()) for line in answers_file.readlines()
                ]
        else:
            if (
                args.ae_model == "prompt_ae"
                and args.ae_model_args["provider"] == "parent_prompt_model"
                and isinstance(qg_model, PromptModel)
            ):
                args.ae_model_args["parent_prompt_model"] = qg_model

            answer_extractor = AE_MODELS_REGISTRY[args.ae_model](**args.ae_model_args)
            answers = answer_extractor.extract_answers(
                passages=texts, num_answers=args.num_answers
            )
            answer_extractor = None

        if isinstance(qg_model, BaseQGModel):
            corpus_qa_pairs = qg_model.generate_questions(
                passages=texts, answers=answers
            )
        else:
            # i dont think the code can reach that point but i added a catch just in case.
            raise ValueError(
                f"Handling this case of qg-model is not implemented.\
                                       qg-model:{args.qg_model}. Change --qg-model arg or implement..."
            )

    # write the questions to the output file
    with open(args.output_file, "w") as output_file:
        for qa_pairs in corpus_qa_pairs:
            qa_pairs = [
                {"question": qa_pair[0], "answer": qa_pair[1]} for qa_pair in qa_pairs
            ]
            print(json.dumps(qa_pairs), file=output_file)


if __name__ == "__main__":
    main()

# example command
# python treqa/evaluate.py --hyp-file data/test.en --qa-file data/test.en.jsonl --output-file data/test.en.scores.jsonl
