import pickle
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI, APIStatusError
from pyparfor import parfor
from timeblock import Timeblock
from together import Together

from judgetuning.jsonfix.json_fixer import JSONFixer
from judgetuning.judge import AnnotationRequest
from judgetuning.llm_client.llm_specs import name_to_llm_spec, LLMModelSpec, model_specs
from judgetuning.llm_client.mockup_client import MockUpClient
from transformers import AutoTokenizer, GenerationConfig


@dataclass
class CompletionOutput:
    completions: list[str]
    price: float | None = None
    time: float | None = None
    n_prompt_token: int | None = None
    n_decoder_token: int | None = None


class CompletionClient:
    @property
    def model_endpoint_name(self):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()

    # TODO safe check that requests is not empty, n>0, ...
    def complete_text(
        self,
        requests: list[str],
        temperature: float = 1.0,
        system_prompt: str | None = None,
        **kwargs,
    ) -> list[CompletionOutput]:
        """
        Returns a completion output for each of the requests, CompletionOutput.completions may itself contain multiple
        completion to support returning multiple seed per request for instance.
        :param requests:
        :param temperature:
        :param system_prompt:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def complete_json(
        self,
        requests: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> list[CompletionOutput]:
        """
        Generate a completion for each of the requests `n` times. The loop is performed over seed then requests.
        The completion is guaranted to be JSON for some classes but not for other (e.g. together does not support
        this for all models).
        """
        # By default returns text without ensuring the format. Classes that support JSON enforcing can overload.
        outputs = self.complete_text(
            requests=requests,
            system_prompt=system_prompt,
            temperature=temperature,
            **kwargs,
        )
        json_fixer = JSONFixer()
        return [
            CompletionOutput(
                completions=[json_fixer(x) for x in output.completions],
                price=output.price,
                time=output.time,
                n_prompt_token=output.n_prompt_token,
                n_decoder_token=output.n_decoder_token,
            )
            for output in outputs
        ]


def handle_request(
    index: int,
    client: OpenAI | Together,
    request: str,
    model: str,
    temperature: float,
    n: int,
    system_prompt: str,
    max_tokens: int,
    json_output: bool,
    backoff: bool = False,
):
    cache_dir = Path("~/Downloads/cache-openai/").expanduser()
    cache_dir.mkdir(exist_ok=True)

    import hashlib

    hash_request = hashlib.sha1(request.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{index}-{hash_request}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
            return cache
    else:
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential_jitter,
        )  # for exponential backoff

        @retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(10))
        def completion_with_backoff(**kwargs):
            return client.chat.completions.create(**kwargs)

        with Timeblock(verbose=False) as time_model_response:
            messages = (
                [{"role": "system", "content": system_prompt}]
                if system_prompt is not None
                else []
            )
            messages.append({"role": "user", "content": request})
            response_format = {"type": "json_object"} if json_output else None
            try:
                completion_fun = (
                    completion_with_backoff
                    if backoff
                    else client.chat.completions.create
                )
                response = completion_fun(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    n=n,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            except APIStatusError as e:
                return CompletionOutput(
                    completions=["Server error: " + str(e)],
                    price=0,
                    time=0,
                    n_prompt_token=0,
                    n_decoder_token=0,
                )
        usage = response.usage
        price = name_to_llm_spec[model].cost(usage)
        res = CompletionOutput(
            completions=[x.message.content for x in response.choices],
            price=price,
            time=time_model_response.duration,
            n_prompt_token=response.usage.prompt_tokens,
            n_decoder_token=response.usage.completion_tokens,
        )
        with open(cache_file, "wb") as f:
            pickle.dump(res, f)
    return res


class OpenAICompletionClient(CompletionClient):
    def __init__(self, model: str, max_tokens: int = 4096, engine: str | None = None):
        if engine is None:
            # engine = "sequential"
            engine = "futures"
        assert engine in ["sequential", "ray", "joblib", "futures"]
        # client for openai server (supports OpenAI and togetherAI)
        available_models = list(name_to_llm_spec.keys())
        assert (
            model in available_models
        ), f"Judge llm model {model} not available, must be one of {available_models}."
        self.model = model
        self._model_endpoint_name = self.model
        self.spec = name_to_llm_spec[model]
        self.max_tokens = max_tokens
        self.engine = engine
        if self.spec.provider == "openai":
            self.client = OpenAI()
        else:
            self.client = Together()

    def __str__(self):
        return f'OpenAICompletionClient(llm="{self.model}")'

    def to_json(self):
        return {
            "completion_client_cls": self.__class__.__name__,
            "completion_client_kwargs": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "engine": self.engine,
            },
        }

    @property
    def model_endpoint_name(self):
        return self._model_endpoint_name

    # TODO update
    # def complete_json(
    #     self,
    #     requests: list[str],
    #     system_prompt: str | None = None,
    #     n: int = 1,
    #     temperature: float = 1.0,
    #     **kwargs,
    # ) -> CompletionOutput:
    #     response_format = (
    #         {"type": "json_object"} if self.spec.support_json_output() else None
    #     )
    #     for request in requests:
    #         with Timeblock(verbose=False) as time_model_response:
    #             # TODO Important, catch together.error.APIError or openai server error
    #             messages = (
    #                 [{"role": "system", "content": system_prompt}]
    #                 if system_prompt is not None
    #                 else []
    #             )
    #             messages.append({"role": "user", "content": request})
    #
    #             response = self.client.chat.completions.create(
    #                 messages=messages,
    #                 model=self.model_endpoint_name,
    #                 stop=["}"],
    #                 temperature=temperature,
    #                 n=n,
    #                 max_tokens=2048,  # TODO how to set a good limit?
    #                 response_format=response_format,
    #                 **kwargs,
    #             )
    #         usage = response.usage
    #         price = self.spec.cost(usage)
    #
    #         json_fixer = JSONFixer()
    #         completions = []
    #         for choices in response.choices:
    #             completion = choices.message.content
    #             if not JSONFixer.is_valid_json(completion):
    #                 completion = json_fixer(completion)
    #                 if not JSONFixer.is_valid_json(completion):
    #                     continue
    #             completions.append(completion)
    #
    #         return CompletionOutput(
    #             completions=completions,
    #             price=price,
    #             time=time_model_response.duration,
    #         )
    #
    def complete_text(
        self,
        requests: list[str],
        system_prompt: str | None = None,
        n: int = 1,
        temperature: float = 0.0,
        json_schema: dict | None = None,
        **kwargs,
    ) -> list[CompletionOutput]:
        if n == 0:
            return [CompletionOutput(completions=[], price=0, time=0) for _ in requests]
        return parfor(
            handle_request,
            inputs=[
                {"request": request, "index": i} for i, request in enumerate(requests)
            ],
            context={
                "model": self.model_endpoint_name,
                "temperature": temperature,
                "n": n,
                "system_prompt": system_prompt,
                "max_tokens": self.max_tokens,
                "json_output": json_schema is not None,
                "client": self.client,
            },
            engine=self.engine,
        )


class TogetherAICompletionClient(OpenAICompletionClient):
    def __init__(self, model: str, max_tokens: int = 4096, engine: str | None = None):
        super().__init__(model=model, max_tokens=max_tokens, engine=engine)
        self.client = Together()

    def __str__(self):
        return f'TogetherAICompletionClient(llm="{self.model}")'

    @property
    def model_endpoint_name(self):
        return self._model_endpoint_name

    # TODO update
    # def complete_json(
    #     self,
    #     requests: list[str],
    #     system_prompt: str | None = None,
    #     n: int = 1,
    #     temperature: float = 1.0,
    #     **kwargs,
    # ) -> CompletionOutput:
    #     response_format = (
    #         {"type": "json_object"} if self.spec.support_json_output() else None
    #     )
    #     for request in requests:
    #         with Timeblock(verbose=False) as time_model_response:
    #             # TODO Important, catch together.error.APIError or openai server error
    #             messages = (
    #                 [{"role": "system", "content": system_prompt}]
    #                 if system_prompt is not None
    #                 else []
    #             )
    #             messages.append({"role": "user", "content": request})
    #
    #             response = self.client.chat.completions.create(
    #                 messages=messages,
    #                 model=self.model_endpoint_name,
    #                 stop=["}"],
    #                 temperature=temperature,
    #                 n=n,
    #                 max_tokens=2048,  # TODO how to set a good limit?
    #                 response_format=response_format,
    #                 **kwargs,
    #             )
    #         usage = response.usage
    #         price = self.spec.cost(usage)
    #
    #         json_fixer = JSONFixer()
    #         completions = []
    #         for choices in response.choices:
    #             completion = choices.message.content
    #             if not JSONFixer.is_valid_json(completion):
    #                 completion = json_fixer(completion)
    #                 if not JSONFixer.is_valid_json(completion):
    #                     continue
    #             completions.append(completion)
    #
    #         return CompletionOutput(
    #             completions=completions,
    #             price=price,
    #             time=time_model_response.duration,
    #         )
    #


def format_request(model: str, system_prompt: str, message: str):
    if "Llama-3.1" in model:
        # taken from https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        """
        If the model is llama3, format the request as follows:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>
        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        tokenize = lambda s: f"<|{s}|>"
        token_begin_of_text = tokenize("begin_of_text")
        token_start_header = tokenize("start_header_id")
        token_end_header = tokenize("end_header_id")
        token_end_text = tokenize("eot_id")
        return f"{token_begin_of_text}{token_start_header}system{token_end_header}{system_prompt}{token_end_text}{token_start_header}user{token_end_header}{message}{token_end_text}{token_start_header}assistant{token_end_header}"
    elif "gemma" in model:
        """
        if model is gemma, format the request as follows:
        <start_of_turn>user
        You are a helpful assistant.
        Hello, how are you?<end_of_turn>
        <start_of_turn>model
        """
        tokenize = lambda s: f"<{s}>"
        token_start_of_turn = tokenize("start_of_turn")
        token_end_of_turn = tokenize("end_of_turn")
        token_begin = tokenize("bos")
        if "gemma-2-" in model:
            return f"{token_begin}{token_start_of_turn}user\n{system_prompt}\n{message}{token_end_of_turn}{token_start_of_turn}model\n"
        return f"{token_start_of_turn}user\n{system_prompt}\n{message}{token_end_of_turn}{token_start_of_turn}model\n"
    elif "Qwen2.5" in model:
        """
        if model is Qwen2.5, format the request as follows:

        <|im_start|>system
        you are an AI expert<|im_end|>
        <|im_start|>user
        what is bias?<|im_end|>
        <|im_start|>assistant
        """
        tokenize = lambda s: f"<|{s}|>"
        token_start = tokenize("im_start")
        token_end = tokenize("im_end")
        return f"{token_start}system\n{system_prompt}{token_end}\n{token_start}user\n{message}{token_end}\n{token_start}assistant\n"
    elif "smol" in model.lower():
        # chatml format: https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
        # https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/discussions/5
        tokenize = lambda s: f"<|{s}|>"
        token_start = tokenize("im_start")
        token_end = tokenize("im_end")
        return (
            f"{token_start}system\n"
            f"{system_prompt}{token_end}\n"
            f"{token_start}user\n"
            f"{message}{token_end}\n"
            f"{token_start}assistant\n"
        )
    else:
        return system_prompt + "\n" + message


class VLLMCompletionClient(CompletionClient):
    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_pred_len: int = 4096,
        tensor_parallel_size: int = 1,
        dtype: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        print(
            f"VLLM settings: model={model}, max_pred_len={max_pred_len}, tensor_parallel_size={tensor_parallel_size}"
        )
        from vllm import LLM

        self.max_pred_len = max_pred_len
        self.model = model
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=0.95,
            max_model_len=max_pred_len,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
        self.verbose = verbose
        self.spec = name_to_llm_spec[model]

    def to_json(self):
        return {
            "completion_client_cls": self.__class__.__name__,
            "completion_client_kwargs": {
                "model": self.model,
                "max_pred_len": self.max_pred_len,
            },
        }

    @property
    def model_endpoint_name(self):
        return self.model

    def complete_text(
        self,
        requests: list[str],
        n: int = 1,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        json_schema: dict | None = None,
        **kwargs,
    ) -> list[CompletionOutput]:
        if n == 0:
            return [CompletionOutput(completions=[], price=0, time=0) for _ in requests]
        from vllm import SamplingParams
        from vllm.model_executor.guided_decoding.guided_fields import (
            GuidedDecodingRequest,
        )

        # adjust sampling parameters as necessary for the task
        sampling_kwargs = {
            "temperature": temperature,
            "max_tokens": self.max_pred_len,
            "n": n,
        }
        sampling_params = SamplingParams(**sampling_kwargs)

        assert n >= 1
        # Generate texts from the prompts
        with Timeblock("Generate completion with vllm") as time_model_response:
            generate_kwargs = dict(
                sampling_params=sampling_params,
            )
            if json_schema:
                generate_kwargs["guided_options_request"] = GuidedDecodingRequest(
                    guided_json=json_schema
                )
            inputs = [
                format_request(
                    model=self.model, system_prompt=system_prompt, message=request
                )
                for request in requests
            ]
            request_outputs: list[RequestOutput] = self.llm.generate(
                inputs,
                **generate_kwargs,
            )
        completions = []
        for request_output in request_outputs:
            n_prompt_tokens = len(request_output.prompt_token_ids)
            if self.verbose:
                print("**input**\n", request_output.prompt)
                print("**output**\n", [x.text for x in request_output.outputs])
            n_generation_tokens = sum(
                [len(x.token_ids) for x in request_output.outputs]
            )
            completions.append(
                CompletionOutput(
                    completions=[x.text for x in request_output.outputs],
                    price=(
                        n_prompt_tokens * self.spec.cost_prompt
                        + n_generation_tokens * self.spec.cost_completion
                    )
                    * 1e-6,
                    time=time_model_response.duration / len(request_outputs),
                    n_prompt_token=n_prompt_tokens,
                    n_decoder_token=n_generation_tokens,
                )
            )
        return completions


class OllamaCompletionClient(CompletionClient):
    def __init__(self, model: str = "llama3.1", max_pred_len: int = 2048):
        from ollama import Client

        self.model = model
        self.client = Client(host="http://localhost:11434")
        self.max_pred_len = max_pred_len
        self.engine = "sequential"

    def __str__(self):
        return f'OllamaCompletionClient(model="{self.model}")'

    def to_json(self):
        return {
            "completion_client_cls": self.__class__.__name__,
            "completion_client_kwargs": {
                "model": self.model,
                "max_pred_len": self.max_pred_len,
            },
        }

    @property
    def model_endpoint_name(self):
        return self.model

    def complete_json(
        self,
        requests: list[str],
        system_prompt: str | None = None,
        n: int = 1,
        temperature: float = 0.0,
        **kwargs,
    ) -> list[CompletionOutput]:
        return self._complete_helper(
            requests=requests, system_prompt=system_prompt, n=n, json_format=True
        )

    def complete_text(
        self,
        requests: list[str],
        system_prompt: str | None = None,
        n: int = 1,
        temperature: float = 0.0,
        **kwargs,
    ) -> list[CompletionOutput]:
        return self._complete_helper(
            requests=requests,
            system_prompt=system_prompt,
            n=n,
            temperature=temperature,
            json_format=False,
        )

    def _complete_helper(
        self,
        requests: list[str],
        json_format: bool,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        n: int = 1,
    ):
        def make_messages(request: str, system_prompt: str):
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": request})
            return messages

        def handle_request(messages, model, stream, options, json_format, client):
            with Timeblock(verbose=False) as time_model_response:
                completions = []
                for i in range(n):
                    kwargs = dict(
                        model=model,
                        messages=messages,
                        stream=stream,
                        options=options,
                        # num_thread=32,
                    )
                    if json_format:
                        kwargs["format"] = "json"
                    response = client.chat(**kwargs)
                    print(
                        f'**input**\n{messages}\n\n**output**\n{response["message"]["content"]}'
                    )
                    completions.append(response["message"]["content"])
            return time_model_response.duration, completions

        messages = [make_messages(request, system_prompt) for request in requests]

        common_kwargs = dict(
            model="llama3.1",
            stream=False,
            options={
                "num_predict": self.max_pred_len,
                "temperature": temperature,
            },
            json_format=json_format,
            client=self.client,
        )
        all_completions = parfor(
            handle_request,
            inputs=[{"messages": x} for x in messages],
            context=common_kwargs,
            engine=self.engine,
        )
        request_completions = []
        for time, completions in all_completions:
            request_completions.append(
                CompletionOutput(
                    completions=completions,
                    price=0,
                    time=time,
                )
            )
        return request_completions


class DeterministicCompletionClient(CompletionClient):
    def __init__(self, answers: list[str]):
        self.answers = answers
        self.index = 0
        self.spec = LLMModelSpec(
            name="mockup",
            provider="me",
            cost_prompt=0,
            cost_completion=0,
        )

    def complete_text(
        self,
        requests: list[str],
        temperature: float = 1.0,
        system_prompt: str | None = None,
        n: int = 1,
        **kwargs,
    ) -> list[CompletionOutput]:
        next_completion = [
            self.answers[i % len(self.answers)]
            for i in range(self.index, self.index + n)
        ]
        self.index += 1
        return [CompletionOutput(completions=next_completion, price=0, time=0)]

    @property
    def model_endpoint_name(self):
        return "deterministic"


def create_client(
    model: str,
    # todo rename to max_model_len
    max_pred_len: int = 4096,
    tensor_parallel_size: int = 1,
    dtype: str | None = None,
    **client_kwargs,
) -> CompletionClient:
    assert model in name_to_llm_spec
    if model in name_to_llm_spec:
        provider = name_to_llm_spec[model].provider
        match provider:
            case "openai":
                return OpenAICompletionClient(
                    model=model, max_tokens=max_pred_len, **client_kwargs
                )
            case "togetherai":
                return TogetherAICompletionClient(
                    model=model, max_tokens=max_pred_len, **client_kwargs
                )
            case "vllm":
                return VLLMCompletionClient(
                    model=model,
                    max_pred_len=max_pred_len,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype=dtype,
                    **client_kwargs,
                )
    else:
        return OllamaCompletionClient(
            model=model, max_pred_len=max_pred_len, **client_kwargs
        )


def completion_client_from_json(json_str: str) -> CompletionClient:
    found = False
    for cls in [
        DeterministicCompletionClient,
        OllamaCompletionClient,
        VLLMCompletionClient,
        OpenAICompletionClient,
    ]:
        if cls.__name__ == json_str["completion_client_cls"]:
            found = True
            break
    assert found, f'invalid class {json_str["completion_client_cls"]}'
    return cls(**json_str["completion_client_kwargs"])
