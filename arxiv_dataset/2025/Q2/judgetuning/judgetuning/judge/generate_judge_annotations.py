import json
from dataclasses import dataclass
from typing import List, Tuple

from judgetuning.annotation_dataset import AnnotationDataset, ArenaHardDataset
from judgetuning.annotation_dataset.tables.table_annotation_dataset import (
    TableAnnotationDataset,
)
from judgetuning.experiment_tracker import ExperimentTracker
from judgetuning.judge import Judge, JudgeAnnotation, AnnotationRequest
from judgetuning.judge.judge_length import JudgeLength
from judgetuning.judge_paths import path_annotation_instruction
from judgetuning.llm_client import OllamaCompletionClient


@dataclass
class RequestInfo:
    instruction_index: str
    instruction: str
    model1: str
    output1: str
    model2: str
    output2: str


def load_annotation_requests(
    annotation_dataset: TableAnnotationDataset,
    baseline_model: str,
    instructions_index: list[str] | None = None,
    models: list[str] | None = None,
) -> list[RequestInfo]:
    if models is None:
        models = annotation_dataset.models
    requests_and_models = []
    for model in models:
        if model == baseline_model:
            # do not generate judge annotation for the baseline as we would have twice the same output
            continue
        baseline_outputs = annotation_dataset.model_output(baseline_model).reindex(
            instructions_index
        )
        model_outputs = (
            annotation_dataset.model_output(model)
            .reindex(instructions_index)
            .fillna("")
        )
        requests_and_models.append(
            [
                RequestInfo(
                    instruction=annotation_dataset.get_instruction(instruction_index),
                    instruction_index=instruction_index,
                    model1=baseline_model,
                    model2=model,
                    output1=baseline_output,
                    output2=model_output,
                )
                for (instruction_index, baseline_output, model_output) in zip(
                    instructions_index, baseline_outputs, model_outputs
                )
            ]
        )

    # transpose the list
    requests_and_models = list(map(list, zip(*requests_and_models)))

    # iterate to get instructions first and then model in the loop, e.g. to have something like
    # for instruction in instructions:
    #     for model in models:
    res = []
    for instruction_list in requests_and_models:
        for request_info in instruction_list:
            res.append(request_info)
    return res


@dataclass
class ResultRow:
    model1: str
    model2: str
    judge_annotation: JudgeAnnotation


def generate_judge_annotations(
    judge: Judge,
    judge_name: str,
    annotation_dataset: AnnotationDataset,
    baseline_model: str,
    instructions_index: list[str] | None = None,
    models: list[str] | None = None,
    override: bool = True,
    expid: str = "dummy",
    support_missing: bool = True,
    resuse_cache_annotations: bool = True,
    experiment_tracker: ExperimentTracker | None = None,
):
    # for each model and prompt, calls the judge and generate judge annotation and save it
    # into alpaca_eval/results/{model}/{judge_name}/annotations.json
    models = annotation_dataset.models if models is None else models
    # assert baseline_model in models
    requests_info = load_annotation_requests(
        annotation_dataset=annotation_dataset,
        instructions_index=instructions_index,
        models=models,
        baseline_model=baseline_model,
    )

    def request_already_present(instruction_index: str, model: str) -> bool:
        path = path_annotation_instruction(
            instruction_dataset=annotation_dataset.name,
            model=model,
            judge_name=judge_name,
            instruction_index=instruction_index,
            expid=expid,
        )
        file_exists = path.exists()
        if not file_exists:
            return False
        else:
            try:
                with open(path, "r") as f:
                    return len(json.load(f)) > 0
            except json.decoder.JSONDecodeError:
                return False

    if resuse_cache_annotations:
        requests_info_to_be_generated = requests_info
    else:
        requests_info_to_be_generated = [
            request
            for request in requests_info
            if not request_already_present(
                instruction_index=request.instruction_index, model=request.model2
            )
        ]

    if requests_info_to_be_generated:
        n_requests_already_present = len(requests_info) - len(
            requests_info_to_be_generated
        )
        print(
            f"Going to generate {len(requests_info_to_be_generated)} annotations ({n_requests_already_present}"
            f" requests were already present.)"
        )
        requests = [
            AnnotationRequest(
                instruction_index=x.instruction_index,
                instruction=x.instruction,
                output1=x.output1,
                output2=x.output2,
                model1=x.model1,
                model2=x.model2,
            )
            for x in requests_info_to_be_generated
        ]
        # Note: we only log the annotations that were freshly generated. This saves space but makes annotations
        # incremental and harder to access,we can revisit this design decision if we need to.
        judge_annotations = judge.annotate(requests)
        if experiment_tracker:
            experiment_tracker.track_judge_annotations(
                [t for x in judge_annotations for t in x]
            )

        for request_info, request_annotation in zip(
            requests_info_to_be_generated, judge_annotations
        ):
            for judge_annotation in request_annotation:
                if (
                    judge_annotation.judge_completion is None
                    or len(judge_annotation.judge_completion) == 0
                ):
                    msg = (
                        f"Completion empty for judge {judge_name} and instruction #"
                        f"{request_info.instruction_index} {request_info.model2}"
                    )
                    if not support_missing:
                        raise ValueError(msg)
                    else:
                        print(msg)

        flat_annotations = [x for annotation in judge_annotations for x in annotation]
        total_time = sum([x.time for x in flat_annotations])
        total_cost = sum([x.cost for x in flat_annotations])
        print(
            f"Generated {len(judge_annotations)} annotations in {total_time}s for {total_cost}$"
        )

    # # TODO make this happens in experiment tracker, eg implement a local option
    try:
        if requests_info_to_be_generated:
            # save annotations
            for request_info, annotations in zip(
                requests_info_to_be_generated, judge_annotations
            ):
                json_file = path_annotation_instruction(
                    instruction_dataset=annotation_dataset.name,
                    model=request_info.model2,
                    judge_name=judge_name,
                    instruction_index=request_info.instruction_index,
                    expid=expid,
                )
                if override or not json_file.exists():
                    json_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(json_file, "w") as f:
                        annotations_to_save = [
                            dict(
                                # TODO if annotation does not contain model, then we should uncomment those
                                #  I think all judges now add model so it should be fine
                                # model1=request_info.model1,
                                # model2=request_info.model2,
                                **x.__dict__,
                            )
                            for x in annotations
                        ]
                        json.dump(annotations_to_save, f, indent=2)
    except Exception as e:
        return


if __name__ == "__main__":
    # completion_client = OllamaCompletionClient()
    # completion_client = VLLMCompletionClient(
    #     model="meta-llama/Meta-Llama-3.1-8B-Instruct"
    # )
    models = ["dbrx-instruct", "claude-2.1"]
    annotation_dataset = ArenaHardDataset(models=models)
    judge = JudgeLength()
    instructions = annotation_dataset.instructions

    generate_judge_annotations(
        judge=judge,
        annotation_dataset=annotation_dataset,
        models=models,
        judge_name="score-1",
        expid="dummy",
        instructions=instructions,
    )
