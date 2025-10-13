# llamafactory.py
import os
import asyncio
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import httpx
from util import logger

_cached_llamafactory_client: Optional[httpx.AsyncClient] = None

def initialize_llamafactory_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> httpx.AsyncClient:
    """
    Initialize and reuse a cached LLaMA Factory API client.
    """
    global _cached_llamafactory_client
    if _cached_llamafactory_client is None:
        base_url = base_url or "http://localhost:8000"
        api_key = api_key or ""

        logger.info(f"Initializing LLaMA Factory client for {base_url}")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        _cached_llamafactory_client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=30.0, pool=60.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50)
        )

    return _cached_llamafactory_client

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.TimeoutException)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def llamafactory_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    **generation_params
) -> str:
    """
    Async wrapper for LLaMA Factory chat completion API
    Maintains identical signature to OpenAI and HF versions
    
    Args:
        model_name: Name of the model to use
        prompt: User input prompt
        system_prompt: System-level instruction
        history_messages: Conversation history
        **generation_params: Generation parameters
        
    Returns:
        Generated text response
    """
    client = initialize_llamafactory_client(
        base_url=generation_params.get("base_url"), 
        api_key=generation_params.get("api_key")
    )
    
    # Prepare messages in OpenAI-compatible format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Prepare request body
    request_body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_params.get("max_tokens", 4096),
        "temperature": generation_params.get("temperature", 0.0),
        "top_p": generation_params.get("top_p", 0.9),
        "stream": False
    }
    
    # Add additional parameters
    for key in ["frequency_penalty", "presence_penalty", "stop"]:
        if key in generation_params:
            request_body[key] = generation_params[key]
    
    # Remove special parameters
    generation_params.pop("hashing_kv", None)
    generation_params.pop("keyword_extraction", None)
    
    try:
        # Call LLaMA Factory API
        response = await client.post(
            "/v1/chat/completions",
            json=request_body
        )
        response.raise_for_status()
        
        # Extract response
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"API error: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.PoolTimeout as e:
        logger.error(f"Connection pool timeout: {str(e)}. This usually means too many concurrent requests.")
        raise
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error(f"Network error: {str(e)}")
        raise
    except KeyError as e:
        logger.error(f"Invalid response format: {str(e)}")
        raise ValueError("Invalid API response format")

async def llamafactory_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    model_name: Optional[str] = None,
    **kwargs
) -> Union[str, Dict]:
    """
    Unified interface for LLaMA Factory completions
    Maintains identical signature to OpenAI and HF versions
    
    Args:
        prompt: User input prompt
        system_prompt: System-level instruction
        history_messages: Conversation history
        model_name: Name of the model to use
        keyword_extraction: Flag to enable keyword extraction
        **kwargs: Additional parameters
        
    Returns:
        Generated text or extracted keywords
    """
    if not model_name:
        if "hashing_kv" in kwargs and "global_config" in kwargs["hashing_kv"]:
            model_name = kwargs["hashing_kv"]["global_config"]["llm_model_name"]
        else:
            raise ValueError("Model name not provided and not found in config")
    
    if not model_name:
        raise ValueError("Model name is required")
    
    # Call generation function
    result = await llamafactory_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
    
    return result