import os
import asyncio
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

@lru_cache(maxsize=1)
def initialize_openai_client(api_key=None, base_url=None):
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, APIError)
    ),
)
async def openai_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs
) -> str:
    client = initialize_openai_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    api_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1000),
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
    }
    
    kwargs.pop("hashing_kv", None)    
    api_params.update(kwargs)
    
    response = client.chat.completions.create(**api_params)
    return response.choices[0].message.content.strip()


async def openai_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    model_name = kwargs.pop("model_name", "gpt-4-turbo")
    result = await openai_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
    
    return result