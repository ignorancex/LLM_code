import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Callable, Any, Coroutine
from tqdm.asyncio import tqdm_asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import logger


class LLMBackend:
    """Unified interface for multiple LLM backends with consistent API"""
    
    # Mapping of backend types to their completion functions
    BACKENDS = {
        "openai": None,  # Will be set to openai_complete
        "hf": None,      # Will be set to hf_complete
        "llamafactory": None,  # Will be set to llamafactory_complete
        "ollama": None  # Will be set to ollama_complete
    }
    
    def __init__(self, backend_type: str, model_name: str, **backend_config):
        """
        Initialize the LLM backend
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory, ollama)
            model_name: Name of the model to use
            backend_config: Backend-specific configuration parameters
        """
        # Lazy import to avoid dependencies for unused backends
        if LLMBackend.BACKENDS["openai"] is None:
            try:
                from .openai import openai_complete
                LLMBackend.BACKENDS["openai"] = openai_complete
            except ImportError:
                logger.warning("OpenAI module not available")
        
        if LLMBackend.BACKENDS["hf"] is None:
            try:
                from .hf import hf_complete
                LLMBackend.BACKENDS["hf"] = hf_complete
            except ImportError:
                logger.warning("Hugging Face module not available")
        
        if LLMBackend.BACKENDS["llamafactory"] is None:
            try:
                from .llamafactory import llamafactory_complete
                LLMBackend.BACKENDS["llamafactory"] = llamafactory_complete
            except ImportError:
                logger.warning("LLaMA Factory module not available")
        
        if LLMBackend.BACKENDS["ollama"] is None:
            try:
                from .ollama import ollama_complete
                LLMBackend.BACKENDS["ollama"] = ollama_complete
            except ImportError:
                logger.warning("OLLAMA module not available")
        
        # Validate backend type
        if backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend type: {backend_type}. "
                             f"Supported types: {list(self.BACKENDS.keys())}")
        
        if self.BACKENDS[backend_type] is None:
            raise ImportError(f"Required dependencies for {backend_type} backend not installed")
        
        self.backend_type = backend_type
        self.model_name = model_name
        self.backend_config = backend_config
        self.complete_fn: Callable[..., Coroutine[Any, Any, str]] = self.BACKENDS[backend_type]
        
        # Default sampling parameters
        self.default_sampling_params = {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        # Backend-specific adjustments
        if backend_type == "hf":
            self.default_sampling_params["top_k"] = -1
        elif backend_type == "openai":
            self.default_sampling_params["presence_penalty"] = 0.0
            self.default_sampling_params["frequency_penalty"] = 0.0
    

    async def generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **sampling_kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts asynchronously with progress bar
        
        Args:
            prompts: List of user prompts to generate responses for
            system_prompt: System-level instruction (optional)
            history_messages: Conversation history (optional)
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            List of generated text responses
        """
        merged_params = {**self.default_sampling_params, **sampling_kwargs}

        async def run_task(prompt):
            return await self.complete_fn(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                model_name=self.model_name,
                **{**self.backend_config, **merged_params}
            )

        # Wrap each prompt in its own coroutine
        tasks = [run_task(prompt) for prompt in prompts]

        # Use tqdm.asyncio.gather to show progress bar with concurrency limit
        # Limit concurrent requests to avoid overwhelming the server
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_task(task) for task in tasks]
        results = await tqdm_asyncio.gather(*limited_tasks, desc="Generating", total=len(limited_tasks))
        return results
    
    async def single_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **sampling_kwargs
    ) -> str:
        """
        Generate text for a single prompt
        
        Args:
            prompt: User prompt to generate response for
            system_prompt: System-level instruction (optional)
            history_messages: Conversation history (optional)
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            Generated text response
        """
        results = await self.generate(
            [prompt],
            system_prompt=system_prompt,
            history_messages=history_messages,
            **sampling_kwargs
        )
        return results[0]