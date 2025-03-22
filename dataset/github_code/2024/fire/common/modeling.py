import re
import torch
import transformers
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from common import utils

SYS_PROMPT = 'You are a fact-checking agent responsible for verifying the accuracy of claims.'


class Model:
    """Class for managing language models, currently supporting OpenAI and Hugging Face models."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        show_responses: bool = False,
        show_prompts: bool = False,
    ) -> None:
        """
        Initializes the model instance with given parameters.
        
        Args:
            model_name (str): Model name in the format 'organization:model_id'.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens for the generated output.
            show_responses (bool): Whether to print responses after generation.
            show_prompts (bool): Whether to print prompts before generation.
        """
        self.organization, self.model_id = model_name.split(':')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_responses = show_responses
        self.show_prompts = show_prompts
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the appropriate model based on the organization type.

        Returns:
            Model instance based on organization type.
        """
        if self.organization == 'openai':
            return self._load_openai_model()
        elif self.organization == 'anthropic':
            return self._load_anthropic_model()
        elif self.organization == 'hf':
            return self._load_huggingface_model()
        else:
            raise ValueError(f"Unsupported organization: {self.organization}")

    def _load_openai_model(self):
        """
        Loads the appropriate OpenAI model.
        
        Returns:
            OpenAI model instance.
        """
        if self.model_id.startswith('o1'):
            print('Loading o1 series...')
            return ChatOpenAI(model=self.model_id, temperature=1)
        else:
            print('Loading OpenAI model...')
            return ChatOpenAI(
                model=self.model_id, 
                temperature=self.temperature, 
                max_tokens=self.max_tokens
            )

    def _load_anthropic_model(self):
        """
        Loads the Anthropic model.
        
        Returns:
            Anthropic model instance.
        """
        print('Loading Anthropic model...')
        return ChatAnthropic(
            model=self.model_id, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens
        )

    def _load_huggingface_model(self):
        """
        Loads the Hugging Face model.
        
        Returns:
            Hugging Face model instance.
        """
        print('Loading Hugging Face model...')
        return transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=self.max_tokens,
        )

    def generate(self, context: str) -> tuple[str, dict | None]:
        """
        Generates a response to the provided prompt.
        
        Args:
            context (str): Input text context.
        
        Returns:
            tuple:
                str: Generated response from the model.
                dict | None: The LLM usage if it's an API call, None if from Hugging Face.
        """
        if self.organization == 'openai':
            return self._generate_openai_response(context)
        elif self.organization == 'anthropic':
            return self._generate_anthropic_response(context)
        elif self.organization == 'hf':
            return self._generate_huggingface_response(context)
        else:
            raise ValueError(f"Unsupported organization: {self.organization}")

    def _generate_openai_response(self, context: str) -> tuple[str, dict | None]:
        """
        Generates a response from an OpenAI model.
        
        Args:
            context (str): Input text context.

        Returns:
            tuple:
                str: Generated response.
                dict: API usage metadata.
        """
        if self.model_id.startswith('o1'):
            response = self.model.invoke(context)
        else:
            messages = [
                SystemMessage(content=SYS_PROMPT),
                HumanMessage(content=context),
            ]
            response = self.model.invoke(messages)
        return response.content, response.usage_metadata

    def _generate_anthropic_response(self, context: str) -> tuple[str, dict | None]:
        """
        Generates a response from an Anthropic model.
        
        Args:
            context (str): Input text context.

        Returns:
            tuple:
                str: Generated response.
                dict: API usage metadata.
        """
        messages = [
            ("system", SYS_PROMPT),
            ("human", context),
        ]
        
        response = self.model.invoke(messages)
        return response.content, response.response_metadata['usage']

    def _generate_huggingface_response(self, context: str) -> tuple[str, dict | None]:
        """
        Generates a response from a Hugging Face model.
        
        Args:
            context (str): Input text context.

        Returns:
            tuple:
                str: Generated response.
                None: No usage metadata available for Hugging Face models.
        """
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": context},
        ]
        
        outputs = self.model(
            messages,
            pad_token_id=self.model.tokenizer.eos_token_id,
        )
        
        return outputs[0]["generated_text"][-1]['content'], None


    def print_config(self) -> None:
        """Prints the model configuration in a readable JSON format."""
        settings = {
            'organization': self.organization,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'show_responses': self.show_responses,
            'show_prompts': self.show_prompts,
        }
        print(utils.to_readable_json(settings))





