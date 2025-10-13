from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

from autogen_core.models import LLMMessage, ModelInfo
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models.openai._openai_client import (
    openai_init_kwargs,
    create_kwargs,
    required_create_args,
    disallowed_create_args,
    BaseOpenAIChatCompletionClient
)
from autogen_ext.models.openai.config import BaseOpenAIClientConfiguration
from typing_extensions import Required 
from openai import AsyncOpenAI
from typing_extensions import Unpack

class LocalOpenAIClientConfiguration(BaseOpenAIClientConfiguration, total=False):
    organization: str
    base_url: str
    model_capabilities: Required[ModelInfo]
    # Not required
    max_tokens: Optional[int] = None #defaults to 4096

def _openai_client_from_config(config: Mapping[str, Any]) -> AsyncOpenAI:
    # Shave down the config to just the OpenAI kwargs
    openai_config = {k: v for k, v in config.items() if k in openai_init_kwargs}
    return AsyncOpenAI(**openai_config)


def _create_args_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    create_args = {k: v for k, v in config.items() if k in create_kwargs}
    create_args_keys = set(create_args.keys())
    if not required_create_args.issubset(create_args_keys):
        raise ValueError(f"Required create args are missing: {required_create_args - create_args_keys}")
    if disallowed_create_args.intersection(create_args_keys):
        raise ValueError(f"Disallowed create args are present: {disallowed_create_args.intersection(create_args_keys)}")
    return create_args

class LocalOpenAIChatCompletionClient(BaseOpenAIChatCompletionClient):
    
    
    def __init__(self, **kwargs: Unpack[LocalOpenAIClientConfiguration]):
        if "model" not in kwargs:
            raise ValueError("model is required for LocalOpenAIChatCompletionClient")

         # Set default model capabilities
        
        default_model_capabilities = {
            "vision": False,            # Default: Model has no vision capabilities
            "function_calling": False,    # Default: Model supports function calling
            "json_output": True          # Default: Model supports JSON output
        }

        model_capabilities: Optional[ModelInfo] = None
        copied_args = dict(kwargs).copy()
        
        if "model_capabilities" in kwargs:
            model_capabilities = kwargs["model_capabilities"]
            del copied_args["model_capabilities"]
        else:
            model_capabilities = ModelInfo(**default_model_capabilities)
        client = _openai_client_from_config(copied_args)
        create_args = _create_args_from_config(copied_args)
        self._raw_config = copied_args
              
        super().__init__(client, create_args=create_args, model_capabilities=model_capabilities)
             
        if "token_limit" in kwargs:
            self._token_limit = kwargs["token_limit"]  # or a fallback value
        else:
            self._token_limit = 4096

    ## Overrides the remainining_tokens method in BaseOpenAiChatCompletionClient in order to use a provided value at init, defaults to 4096 if not provided by the user.
    def remaining_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        
        return self._token_limit - self.count_tokens(messages, tools)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = _openai_client_from_config(state["_raw_config"])