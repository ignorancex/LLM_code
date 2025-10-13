import openai
import backoff
from openai import AzureOpenAI, OpenAIError, RateLimitError, APITimeoutError

class AzureOpenAIClient:
    def __init__(self, api_key, endpoint, deployment_name, api_version='2024-12-01-preview'):
        """
        Initialize the AzureOpenAIClient with the provided API key, endpoint, and deployment name.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
        self.total_cost = 0  # Track total cost

    @backoff.on_exception(backoff.expo, (RateLimitError, APITimeoutError))
    def send_request(self, model_name, prompt, max_tokens=9000, temperature=1, top_p=1.0, **kwargs):
        """
        Send a prompt to the specified deployment (model) via the Azure OpenAI API.
        Returns:
          str: The generated response from the model.
        """
        try:
            # Convert the prompt string into a chat message
            messages = [{"role": "user", "content": prompt}]

            # Use the chat completion endpoint with an increased timeout
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=1,
                timeout=60,  # Increased timeout in seconds
            )

            generated_text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            # Calculate and accumulate cost
            cost = self.calculate_price(prompt_tokens, completion_tokens)
            self.total_cost += cost

            return generated_text

        except RateLimitError as e:
            # Backoff will handle retries for RateLimitError
            raise e
        except OpenAIError as e:
            raise Exception(f"An error occurred: {str(e)}")

    def calculate_price(self, prompt_tokens, completion_tokens):
        """
        Calculate the price based on the token usage and model pricing.
        Returns:
          float: Approximate cost in USD.
        """
        model_pricing = {
            'z-gpt-4o-2024-08-0': {
                'input': 5.00 / 1000,
                'output': 15.00 / 1000,
            },
            'z-gpt-4o-mini-2024-07-18': {
                'input': 0.15 / 1000,
                'output': 0.6 / 1000,
            },
            'z-gpt-o1-mini-2024-09-12': {
                'input': 3 / 1000,
                'output': 12 / 1000,
            },
            'z-gpt-o1-preview-2024-09-12': {
                'input': 15 / 1000,  
                'output': 60 / 1000,  
            },
            'z-gpt-o3-mini-2025-01-31' : {
                'input': 1.10 / 1000,
                'output': 4.40 / 1000,
            },
        }

        model_name = self.deployment_name
        if model_name in model_pricing:
            pricing = model_pricing[model_name]
            input_price_per_1k = pricing['input']
            output_price_per_1k = pricing['output']
        else:
            input_price_per_1k = 0.0
            output_price_per_1k = 0.0

        input_cost = (prompt_tokens / 1000) * input_price_per_1k
        output_cost = (completion_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost
        return total_cost

    def get_total_cost(self):
        """Get the total accumulated cost of all API calls."""
        return self.total_cost
