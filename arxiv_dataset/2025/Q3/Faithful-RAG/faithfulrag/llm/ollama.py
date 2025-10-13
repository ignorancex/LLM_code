import sys
import os
import json
import logging
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import httpx
from tenacity import RetryCallState
from util import logger


@lru_cache(maxsize=1)
def get_ollama_client(base_url=None, timeout=300.0):
    base_url = base_url or "http://localhost:11434"
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.ReadTimeout, 
         httpx.RemoteProtocolError, httpx.PoolTimeout)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def ollama_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    **kwargs
) -> str:
    client = get_ollama_client(timeout=kwargs.get("timeout", 300.0))
    
    messages = history_messages.copy() if history_messages else []
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    request_body = {
        "model": model_name,
        "messages": messages,
        "stream": False
    }
    
    options = {}
    if "max_tokens" in kwargs:
        options["num_predict"] = kwargs.pop("max_tokens")
    if "temperature" in kwargs:
        options["temperature"] = kwargs.pop("temperature")
    if "top_p" in kwargs:
        options["top_p"] = kwargs.pop("top_p")
    
    for param in ["format", "template", "keep_alive"]:
        if param in kwargs:
            options[param] = kwargs.pop(param)
    
    if kwargs:
        options.update(kwargs)
    
    if options:
        request_body["options"] = options
    
    try:
        response = await client.post("/api/chat", json=request_body)
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"].strip()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise e

async def ollama_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    model_name = kwargs.pop("model_name", "llama3")
    
    if keyword_extraction:
        kwargs["format"] = "json"
    
    kwargs.pop("hashing_kv", None)
    
    return await ollama_chat_completion(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )