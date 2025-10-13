from typing import Any

import inspect_ai.model._providers.util.tracker
import openai
from inspect_ai._util.constants import DEFAULT_MAX_RETRIES
from inspect_ai.model import GenerateConfig, modelapi
from inspect_ai.model._providers.openai import OpenAIAPI
from inspect_ai.model._providers.util import model_base_url
from inspect_ai.model._providers.openai import OpenAIAPI, OpenAIAsyncHttpxClient
import httpx


@modelapi(name="sgl")
class SglOpenAI(OpenAIAPI):
    """This class is a wrapper around the OpenAIAPI client that allows one to use OpenAI models and Azure OpenAI models simultaneously."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # vLLM hosting setup
        # http_client = OpenAIAsyncHttpxClient(verify=False)
        # vllm serve meta-llama/Llama-3.3-70B-Instruct --port 18443 --dtype bfloat16 --tensor-parallel-size 8
        http_client = httpx.AsyncClient(verify=False)
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=model_base_url(base_url, "OPENAI_BASE_URL"),
            max_retries=(
                config.max_retries if config.max_retries else DEFAULT_MAX_RETRIES
            ),
            http_client=http_client,
            **model_args,
        )

        # Required by inspect
        self._time_tracker = inspect_ai.model._providers.util.tracker.HttpxTimeTracker(
            self.client._client
        )