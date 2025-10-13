import asyncio
import copy
import json
import math
import os
import time
from typing import Union

import aiohttp
import google.generativeai as genai
import openai
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score

import properties
import scorer_utils
import utils
from kings_aihub_api import AIHub as LlamaModel

_SEED = 10
_OPENAI_KEY = open('/Users/user/Desktop/openai_key_fc_eval.txt', 'r').read()

_GEMINI_KEY = open('/Users/user/Desktop/gemini_key_fc_eval.txt', 'r').read()
_GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-pro", generation_config={
    "response_mime_type": "application/json"})
genai.configure(api_key=_GEMINI_KEY)

_LLAMA_MODEL = LlamaModel()

_MAX_TOKENS = 3000
_MAX_RETRIES = 10
_TIMEOUT = 180
_TEMPERATURE = 0
_BASE_URL = "https://api.openai.com/v1/chat/completions"
_IGNORE_LABELS_DEFAULT = ["conflicting evidence/cherrypicking"]


def _query_gemini(prompt: str, max_tokens=_MAX_TOKENS):
    try:
        return _GEMINI_MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=max_tokens,
                temperature=_TEMPERATURE,
            ),
        )
    except Exception as e:
        print(e)
        return ""


def _query_llama(prompt: str, max_tokens=_MAX_TOKENS):
    try:
        return _LLAMA_MODEL.ask(prompt)
    except Exception as e:
        print(e)
        return ""


def _query_openai(prompt: str, client, model: properties.ModelApi = None, seed=_SEED, max_tokens=_MAX_TOKENS,
                  response_format="json_object", logprob=False):
    prompting_model = model.value
    if logprob:
        return client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=prompting_model,
            max_tokens=max_tokens,
            response_format={"type": response_format},
            seed=seed,
            temperature=0,
            timeout=_TIMEOUT,
            logprobs=True,
            top_logprobs=5
        )
    else:
        return client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=prompting_model,
            max_tokens=max_tokens,
            response_format={"type": response_format},
            seed=seed,
            temperature=0,
            timeout=_TIMEOUT,
        )


def _get_response_text(response: Union[
    openai.types.chat.chat_completion.ChatCompletion, genai.types.generation_types.GenerateContentResponse]):
    if type(response) == openai.types.chat.chat_completion.ChatCompletion:
        return response.choices[0].message.content
    if type(response) == genai.types.generation_types.GenerateContentResponse:
        try:
            return response.text
        except Exception as e:
            print("Error in extracting Gemini response: {}".format(e))
            return ""
    else:
        return response


def _process_output(dataset_sample: properties.AveritecEntry,
                    response: Union[
                        openai.types.chat.chat_completion.ChatCompletion, genai.types.generation_types.GenerateContentResponse],
                    logprob: bool = False):
    if logprob:
        logprob_inp = response.choices[0].logprobs.content
    else:
        logprob_inp = None
    return properties.OpenAIResponse(claim=dataset_sample.claim, evidence=dataset_sample.evidence,
                              response=_get_response_text(response),
                              gold=dataset_sample.label.lower(), id=dataset_sample.id,
                              logprobs=logprob_inp)


def _averitec_qa_to_str(evidence: properties.AveritecQA):
    evidence_as_str = (evidence.question + "? ").replace("??", "?")  # sometimes question mark or fullstop missing
    for a in evidence.answers:
        evidence_as_str += (a.answer + ". ").replace("..", ".")
        if a.answer_type.lower() == "boolean":
            evidence_as_str += (a.boolean_explanation + ". ").replace("..", ".")
    return evidence_as_str


def _prepare_prompt(dataset_sample: properties.AveritecEntry, prediction: properties.AveritecEntry,
                    prompt_type: properties.PromptTypes):
    """Formats prompt using dataset sample as input."""
    if type(dataset_sample.evidence) == properties.AveritecQA:
        return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim,
                                                             " ".join([_averitec_qa_to_str(e) for e in
                                                                       dataset_sample.evidence]))
    else:
        if prompt_type in [properties.PromptTypes.ATOMIC_REFERENCE_FACTS,
                           properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL]:
            return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim,
                                                                 dataset_sample.evidence,
                                                                 prediction.evidence)
        else:
            # e.g. proxy based evaluation using the claim and predicted evidence!
            return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim, prediction.evidence)


def _get_system_prediction(sample: properties.AveritecEntry, predictions: list[properties.AveritecEntry]):
    for entry in predictions:
        if entry.claim.lower().strip() == sample.claim.lower().strip():
            return entry
    return None


def calculate_prediction_scores(input_data: pd.DataFrame, preds: list[properties.OpenAIResponse],
                                prompt_type: properties.PromptTypes) -> list[
    properties.OpenAIResponse]:
    predictions_w_scores = []
    for i, pred in enumerate(preds):
        if prompt_type == properties.PromptTypes.ATOMIC_FACTS:
            predictions_w_scores.append(calculate_atomic_score_openai(pred))
        elif prompt_type == properties.PromptTypes.ATOMIC_REFERENCE_FACTS:
            predictions_w_scores.append(calculate_atomic_score_openai_response(pred))
        elif prompt_type == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
            pred_w_scores = calculate_atomic_score_prec_recall_openai_response(pred)
            if pred_w_scores:
                predictions_w_scores.append(pred_w_scores)
        elif prompt_type == properties.PromptTypes.COT:
            predictions_w_scores.append(calculate_pseudo_score_openai_response(input_data.iloc[i], pred))

    return predictions_w_scores


def prompt_api_model(dataset: list[properties.AveritecEntry], predictions: list[properties.AveritecEntry],
                     prompt_type: properties.PromptTypes, client, match_system_preds=True,
                     responses_output_path: str = None, logprob: bool = False,
                     api: properties.ModelApi = properties.ModelApi.GPT4o) -> list[properties.OpenAIResponse]:
    """Prompts OpenAI models."""
    # load previously generated responses
    if os.path.exists(responses_output_path):
        responses = utils.load_jsonl_file(responses_output_path, dataclass=properties.OpenAIResponse)
        prev_claims = [x.claim for x in responses]
    else:
        responses = []
        prev_claims = []
    for i, sample in enumerate(dataset):
        if sample.claim in prev_claims:
            continue
        if match_system_preds:
            # search in predictions for matching prediction
            pred = _get_system_prediction(sample, predictions)
            if not pred:
                responses.append("")
                continue
        else:
            pred = predictions[i]
        prompt = _prepare_prompt(sample, pred, prompt_type)
        attempt = 0
        while attempt < _MAX_RETRIES:
            try:
                if api in [properties.ModelApi.GPT4o, properties.ModelApi.GPT4]:
                    response = _query_openai(prompt=prompt, client=client, response_format="json_object", model=api,
                                             logprob=logprob)
                elif api in [properties.ModelApi.GEMINI_FLASH, properties.ModelApi.GEMINI_PRO]:
                    response = _query_gemini(prompt)
                elif api == properties.ModelApi.LLAMA:
                    response = _query_llama(prompt)
                else:
                    raise Exception("Specified model not available: {}".format(api.value))
                responses.append(
                    _process_output(sample, response, logprob=logprob))
                # save results in between
                utils.save_jsonl_file(responses, responses_output_path)
                print("One request successfully processed..")
                break
            except openai.APITimeoutError as e:
                attempt += 1
                wait_time = 10 ** attempt  # Exponential backoff
                print(f"Request timed out. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        if attempt >= _MAX_RETRIES:
            raise Exception("Maximum retries reached. Request failed.")
    return responses


async def prompt_openai_model_asynchron(dataset: list[properties.AveritecEntry],
                                        predictions: list[properties.AveritecEntry],
                                        prompt_type: properties.PromptTypes, client, match_system_preds=True,
                                        model: str = None,
                                        responses_output_path: str = None) -> \
        list[
            properties.OpenAIResponse]:
    """Prompts OpenAI models."""
    # load previously generated responses
    if os.path.exists(responses_output_path):
        responses = utils.load_jsonl_file(responses_output_path, dataclass=properties.OpenAIResponse)
        prev_claims = [x.claim for x in responses]
    else:
        responses = []
        prev_claims = []
    prompts = []
    for i, sample in enumerate(dataset):
        if sample.claim in prev_claims:
            continue
        if match_system_preds:
            # search in predictions for matching prediction
            pred = _get_system_prediction(sample, predictions)
            if not pred:
                responses.append("")
                continue
        else:
            pred = predictions[i]
        prompts.append(_prepare_prompt(sample, pred, prompt_type))
    #
    print("Starting processing of ten prompts..")
    responses_api = await batch_requests(prompts[:10], model, _MAX_TOKENS, "json_object", _SEED)
    for i, response in enumerate(responses_api):
        if "error" in response:
            print(f"Error for prompt {i + 1}: {response['message']}")
        else:
            responses.append(_process_output(dataset[i], response))
            print(f"Response to prompt {i + 1}: {response['choices'][0]['message']['content']}")
    return responses


def calculate_atomic_score_openai_response(response_openai):
    try:
        response_openai_copy = copy.deepcopy(response_openai)
        response = json.loads(response_openai.response)
        response_openai_copy.response = response
        if ("refute" in response and response["refute"] > 0) or (
                "contradict" in response and response["contradict"] > 0):
            # evidence clearly contradicts a sub-fact of the claim
            response_openai_copy.response['score'] = 1
        elif "supports" in response:
            response_openai_copy.response['score'] = response["supports"] / (
                    response["supports"] + response["not enough information"])
        else:
            response_openai_copy.response['score'] = response["support"] / (
                    response["support"] + response["not enough information"])
        return response_openai_copy
    except Exception as e:
        return 0


def calculate_atomic_score_openai(response_openai):
    response_openai_copy = copy.deepcopy(response_openai)
    try:
        if type(response_openai.response) == str:
            response_openai.response = response_openai.response.replace(": '", ": \"").replace("',", "\",")
            response = json.loads(response_openai.response)
            response_openai_copy.response = response
        else:
            response = response_openai_copy.response
        response_openai_copy.response['score'] = (response["support"]+response["contradict"]) / response["facts count"]
    except Exception as e:
        print(e)
        response_openai_copy.response = None
    return response_openai_copy


def calculate_atomic_score_prec_recall_openai_response(response_openai):
    response_openai_copy = copy.deepcopy(response_openai)
    try:
        if type(response_openai.response) == str:
            response = json.loads(response_openai.response.replace(": '", ": \"").replace("',", "\",").replace("':", "\":"))
        else:
            response = response_openai.response
        response_openai_copy.response = response
        response_openai_copy.response['precision'] = response["support predicted evidence"] / response[
            "facts count predicted evidence"]
        response_openai_copy.response['recall'] = response["support reference evidence"] / response[
            "facts count reference evidence"]
    except Exception as e:
        print("Following exception occurred: {}".format(e))
        return None
    return response_openai_copy


def get_token_prob(probs: list, proxy: str):
    """
    Given a list of logprobs find the entry for search_token
    :return: entry for search_token
    """
    for p in probs[-10:]:
        if type(p) == dict:
            if p['token'].lower() in [label.value for label in properties.Logprobs]:
                if p['token'].lower() == proxy:
                    return p['logprob']
                # else iterate through whole list and find a label
                else:
                    for entry in p['top_logprobs']:
                        if entry['token'].lower() == proxy:
                            return entry['logprob']
        else:
            if p.token.lower() in [label.value for label in properties.Logprobs]:
                if p.token.lower() == proxy:
                    return p.logprob
                # else iterate through whole list and find a label
                else:
                    for entry in p.top_logprobs:
                        if entry.token.lower() == proxy:
                            return entry.logprob
    return None


def calculate_pseudo_score_openai_response(input_entry: pd.Series, response_openai: properties.OpenAIResponse,
                                           use_logprob=True):
    response_openai_copy = copy.deepcopy(response_openai)
    try:
        if type(response_openai.response) == str:
            response = json.loads(response_openai.response)
        else:
            response = response_openai.response
        response_openai_copy.response = response
        if use_logprob:
            # get log prob for gold label which is used as proxy for evaluation
            if input_entry['gold label'].strip().lower() == "refuted":
                proxy = properties.Logprobs.REFUTED.value
            elif input_entry['gold label'].strip().lower() == "supported":
                proxy = properties.Logprobs.SUPPORTED.value
            else:
                # if input_entry['label_majority'] == "not enough information":
                proxy = properties.Logprobs.NEI.value

            proxy_prob = get_token_prob(response_openai_copy.logprobs, proxy)
            # response_openai_copy.response['score'] = math.exp(proxy_prob) if proxy_prob else 0
            if proxy_prob or proxy_prob == 0:
                response_openai_copy.response['score'] = math.exp(proxy_prob)
            else:
                response_openai_copy.response['score'] = 0

        else:
            response_openai_copy.response['score'] = 1 if properties.LABEL_DICT[properties.Label(response['label'])] == \
                                                          properties.LABEL_DICT[
                                                              properties.Label(
                                                                  input_entry['label_majority'])] else 0
    except Exception as e:
        print(e)
        response_openai_copy.response['score'] = None
    return response_openai_copy


def calculate_atomic_score(response: dict):
    try:
        return response["support"] / response["facts count"]
    except Exception:
        return 0


def calculate_atomic_score_prec_recall(response: dict):
    try:
        prec = response["support predicted evidence"] / response["facts count predicted evidence"]
        recall = response["support reference evidence"] / response["facts count reference evidence"]
        return prec, recall
    except Exception:
        return None, None


def evaluate_openai_output(output_all, prompt_type: properties.PromptTypes, ignore_labels=_IGNORE_LABELS_DEFAULT,
                           is_two_classes=False):
    output = [response for response in output_all if response.gold.lower() not in ignore_labels]
    if prompt_type in [properties.PromptTypes.ATOMIC_FACTS, properties.PromptTypes.ATOMIC_REFERENCE_FACTS]:
        # calculate output score
        scores = [calculate_atomic_score(json.loads(pred.response)) for pred in output]
        return sum(scores) / len(scores)
    elif prompt_type == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
        if type(output[0].response) == str:
            scores = [calculate_atomic_score_prec_recall(json.loads(pred.response)) for pred in output]
        else:
            scores = [calculate_atomic_score_prec_recall(pred.response) for pred in output]
        prec_scores = [x[0] for x in scores if x[0]]
        recall_scores = [x[1] for x in scores if x[1]]
        return sum(prec_scores) / len(prec_scores), sum(recall_scores) / len(recall_scores)

    # map output to labels
    pred_labels = []
    gold_labels = []
    for response in output:
        try:
            pred_label = scorer_utils.map_label(response.response, is_two_classes)
            if response.gold.lower() not in ignore_labels and pred_label in range(3):
                pred_labels.append(pred_label)
                gold_labels.append(properties.LABEL_DICT[properties.Label(response.gold.lower())])
        except Exception as e:
            print("Exception {} for the following claim: {}".format(e, response.claim))

    # calculate metrics (F1 micro/macro) todo replace this by own confusion matrix based on probabilities
    f1_micro = f1_score(y_true=gold_labels, y_pred=pred_labels, average='micro')
    f1_macro = f1_score(y_true=gold_labels, y_pred=pred_labels, average='macro')

    confusion_matrix = metrics.confusion_matrix(gold_labels, pred_labels)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "confusion_metrics": confusion_matrix.tolist()}


async def fetch_response(session, prompt, prompting_model, max_tokens, response_format, seed):
    headers = {
        "Authorization": f"Bearer {_OPENAI_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": prompting_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": response_format},
        "seed": seed,
        "temperature": 0,
        "timeout": _TIMEOUT
    }

    async with session.post(_BASE_URL, headers=headers, json=json_data) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {"error": response.status, "message": await response.text()}


async def batch_requests(prompts, prompting_model, max_tokens, response_format, seed):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_response(session, prompt, prompting_model, max_tokens, response_format, seed)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
