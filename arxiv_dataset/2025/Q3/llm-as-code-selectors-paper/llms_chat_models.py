from dataclasses import dataclass
import json
import os
import openai
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from typing import Any, ClassVar, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException
import time

import torch

from log import get_logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

load_dotenv()


# Configure logging
logger = get_logger(__name__)

class FireworksDeepSeekR1(BaseChatModel):
    model_name: str = Field('deepseek-r1', alias="model")
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 20480
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2  # Maximum retry attempts
    api_key: str = os.getenv('FIREWORKS_API_KEY')

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """Generate a response with retry logic for API failures."""

        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        payload = {
            "model": "accounts/fireworks/models/deepseek-r1",
            "max_tokens": self.max_tokens,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": self.temperature,
            "messages": [{'role': 'user' if message.type == 'human' else 'assistant', 'content': message.content} for message in messages]
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(self.max_retries + 1):  # Allow max_retries + initial attempt
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                response.raise_for_status()  # Raise HTTP error if status is not 200-299
                
                response_content = response.json()  # Use .json() instead of eval
                message = AIMessage(
                    content=response_content['choices'][0]['message']['content'],
                    additional_kwargs={},  
                    response_metadata=response_content['usage'],
                    usage_metadata={
                        "input_tokens": response_content['usage']['prompt_tokens'],
                        "output_tokens": response_content['usage']['completion_tokens'],
                        "total_tokens": response_content['usage']['total_tokens'],
                    },
                )
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            except RequestException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff (2s, 4s, 8s, ...)
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API request failed after {self.max_retries} retries: {e}")

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "fireworks-deepseek-r1"

class FireworksChatModel(BaseChatModel):
    model_name: str = Field(alias="model")
    temperature: Optional[float] = 1
    max_tokens: Optional[int] = 20480
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2  # Maximum retry attempts
    api_key: str = os.getenv('FIREWORKS_API_KEY')
    top_p: Optional[int] = 1
    top_k: Optional[int] = 40
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    custom_endpoint: Optional[str] = None

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """Generate a response with retry logic for API failures."""

        model_endpoint = self.custom_endpoint if self.custom_endpoint is not None else f"accounts/fireworks/models/{self.model_name}"


        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        payload = {
            "model": model_endpoint,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "temperature": self.temperature,
            "messages": [{'role': 'user' if message.type == 'human' else 'assistant', 'content': message.content} for message in messages]
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(self.max_retries + 1):  # Allow max_retries + initial attempt
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                response.raise_for_status()  # Raise HTTP error if status is not 200-299
                
                response_content = response.json()  
                message = AIMessage(
                    content=response_content['choices'][0]['message']['content'],
                    additional_kwargs={},  
                    response_metadata=response_content['usage'],
                    usage_metadata={
                        "input_tokens": response_content['usage']['prompt_tokens'],
                        "output_tokens": response_content['usage']['completion_tokens'],
                        "total_tokens": response_content['usage']['total_tokens'],
                    },
                )
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            except RequestException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff (2s, 4s, 8s, ...)
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API request failed after {self.max_retries} retries: {e}")

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name


# Gemma 3 4B from Hugging Face running locally ----------------------------------------
@dataclass
class Gemma3ChatModelConfig:
    role_names: ClassVar[Dict[str, str]] = {
        "human": "user",
        "assistant": "model",
        "system": "system",
    }

class Gemma3ChatModel(BaseChatModel):
    model_name: str = Field("gemma-3-4b-it", alias="model")
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 100

    def load_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gemma3_4b_model_id = "google/gemma-3-4b-it"
        self.llm = Gemma3ForConditionalGeneration.from_pretrained(gemma3_4b_model_id, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(gemma3_4b_model_id)
        self.processor = AutoProcessor.from_pretrained(gemma3_4b_model_id)
        return self

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """Generate a response with retry logic for API failures."""

        # Adjust prompt formating
        formatted_messages = []
        for message in messages:
            assert message.type in Gemma3ChatModelConfig.role_names.keys(), f"Invalid message role. Valid roles: {list(Gemma3ChatModelConfig.role_names.keys())}"

            formatted_messages.append({
                'role': 'user',
                'content': [{
                    "type": "text",
                    "text": message.content,
                }]
            })

        inputs = self.processor.apply_chat_template(
            formatted_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.llm.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Run inference
        with torch.inference_mode():
            generation = self.llm.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=True)
            generation = generation[0][input_len:]

        # Usage metadata
        input_tokens = input_len
        output_tokens = generation.shape[-1]
        total_tokens = input_tokens + output_tokens

        response_metadata = {
            'token_usage': 
                {
                    'completion_tokens': output_tokens, 
                    'prompt_tokens': input_tokens, 
                    'total_tokens': total_tokens
                }
            }


        message = AIMessage(
            content=self.processor.decode(generation, skip_special_tokens=True),
            additional_kwargs={},  
            response_metadata=response_metadata,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name
    

# Huggingface local models -------------------------------------------

@dataclass
class HuggingFaceLocalModelConfig:
    role_names: ClassVar[Dict[str, str]] = {
        "human": "user",
        "assistant": "model",
        "system": "system",
    }

class HuggingFaceLocalChatModel(BaseChatModel):
    model_name: str = Field(alias="model")
    tokenizer_name: Optional[str] = None
    processor_name: Optional[str] = None
    temperature: float = 0
    max_tokens: Optional[int] = None
    llm: Any = None
    tokenizer: Any = None
    processor: Any = None
    
    def load_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.llm = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name if self.tokenizer_name is None else self.tokenizer_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name if self.processor_name is None else self.processor_name)
        return self

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """Generate a response with retry logic for API failures."""

        # Adjust prompt formating
        formatted_messages = []
        for message in messages:
            assert message.type in HuggingFaceLocalModelConfig.role_names.keys(), f"Invalid message role. Valid roles: {list(HuggingFaceLocalModelConfig.role_names.keys())}"

            formatted_messages.append({
                'role': HuggingFaceLocalModelConfig.role_names[message.type],
                'content': message.content,
            })

        inputs = self.processor.apply_chat_template(
            formatted_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.llm.device)

        input_len = inputs["input_ids"].shape[-1]

        # Run inference
        with torch.inference_mode():
            generation = self.llm.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=True)
            generation = generation[0][input_len:]

        # Usage metadata
        input_tokens = input_len
        output_tokens = generation.shape[-1]
        total_tokens = input_tokens + output_tokens

        response_metadata = {
            'token_usage': 
                {
                    'completion_tokens': output_tokens, 
                    'prompt_tokens': input_tokens, 
                    'total_tokens': total_tokens
                }
            }

        message = AIMessage(
            content=self.processor.decode(generation, skip_special_tokens=True),
            additional_kwargs={},  
            response_metadata=response_metadata,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name
    
class MaritacaChatModel(BaseChatModel):
    model_name: str = Field(alias="model")
    temperature: Optional[float] = 1
    max_tokens: Optional[int] = 20480
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2  # Maximum retry attempts
    api_key: str = os.getenv('MARITACA_API_KEY')
    top_p: Optional[int] = 1
    top_k: Optional[int] = 40
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    custom_endpoint: Optional[str] = None

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """Generate a response with retry logic for API failures."""

        client = openai.OpenAI(
            api_key=os.environ.get("MARITACA_API_KEY"),
            base_url="https://chat.maritaca.ai/api",
        )

        for attempt in range(self.max_retries + 1):  # Allow max_retries + initial attempt
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user' if message.type == 'human' else 'system', 'content': message.content} for message in messages],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    stop=self.stop,
                    )
                                
                message = AIMessage(
                    content=response.choices[0].message.content,
                    additional_kwargs={},  
                    response_metadata=response.usage.model_dump(),
                    usage_metadata={
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                )
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff (2s, 4s, 8s, ...)
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API request failed after {self.max_retries} retries: {e}")

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name