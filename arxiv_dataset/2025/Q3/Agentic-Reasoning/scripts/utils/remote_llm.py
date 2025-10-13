import os
from typing import List, Optional
from dataclasses import dataclass
import dspy

@dataclass
class SamplingParams:
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.8
    stop: Optional[List[str]] = None

class RemoteAPILLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if 'gpt' in model_name:
            from openai import OpenAI
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set via environment variable 'OPENAI_API_KEY'.")
            self.client = OpenAI(api_key=self.api_key)
            self.provider = 'openai'
        elif 'claude' in model_name:
            from anthropic import Anthropic
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key required. Set via environment variable 'ANTHROPIC_API_KEY'.")
            self.anthropic = Anthropic(api_key=self.api_key)
            self.provider = 'anthropic'
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, **kwargs) -> List:
        if sampling_params is None:
            sampling_params = SamplingParams()

        responses = []
        for prompt in prompts:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=sampling_params.max_tokens,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    n=1,
                    stop=sampling_params.stop,
                )
                output_text = response.choices[0].message.content
            else:  # anthropic
                response = self.anthropic.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=sampling_params.max_tokens,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    stop_sequences=sampling_params.stop,
                )
                output_text = response.content[0].text

            # Create object to mimic local LLM structure
            class Temp:
                pass
            temp = Temp()
            temp.outputs = [type("Output", (object,), {"text": output_text})]
            responses.append(temp)
        return responses

    def invoke(self, prompt: str) -> type:
        """Method for code generation compatibility"""
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            return type("Response", (), {"content": response.choices[0].message.content})
        else:  # anthropic
            response = self.anthropic.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )
            return type("Response", (), {"content": response.content[0].text}) 


def setup_dspy_model(model_name: str) -> dspy.LM:
    """
    Configure and return a DSPy language model based on the specified model name.
    
    Args:
        model_name (str): Name of the model to configure ('gpt-4', 'deepseek', etc.)
    
    Returns:
        dspy.LM: Configured language model
    """
    model_configs = {
        'o1': {
            'model_path': 'openai/o1',
            'kwargs': {
                'api_key': os.getenv("OPENAI_API_KEY"),
                'temperature': 1.0,
                'top_p': 0.9,
                'max_tokens': 8192
            }
        },
        'gpt-4o': {
            'model_path': 'openai/gpt-4o',
            'kwargs': {
                'api_key': os.getenv("OPENAI_API_KEY"),
                'temperature': 1.0,
                'top_p': 0.9
            }
        },
        'deepseek': {
            'model_path': 'openai/deepseek-reasoner',
            'kwargs': {
                'api_key': os.getenv("DEEPSEEK_API_KEY"),
                'temperature': 1.0,
                'top_p': 0.9
            }
        },
        'deepseek-v3': {
            'model_path': 'openai/DeepSeek-V3',
            'kwargs': {
                'api_key': os.getenv("DEEPSEEK_API_KEY"),
                'temperature': 1.0,
                'top_p': 0.9
            }
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    config = model_configs[model_name]
    return dspy.LM(config['model_path'], **config['kwargs'])