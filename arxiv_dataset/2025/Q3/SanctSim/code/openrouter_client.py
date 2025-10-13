# import openai
# import backoff
# from openai import OpenAI, OpenAIError, RateLimitError

# class OpenRouterClient:
#     def __init__(self, api_key, base_url="https://openrouter.ai/api/v1", model_name=None, site_url=None, site_name=None):
#         """
#         Initialize the OpenRouterClient with the provided API key and base URL.

#         Parameters:
#         - api_key (str): Your OpenRouter API key.
#         - base_url (str): The base URL for OpenRouter API. Default is 'https://openrouter.ai/api/v1'.
#         - model_name (str): The name of the model to use (e.g., 'openai/gpt-3.5-turbo').
#         - site_url (str): Optional. Site URL for rankings on openrouter.ai.
#         - site_name (str): Optional. Site title for rankings on openrouter.ai.
#         """
#         # Initialize the OpenAI client for OpenRouter
#         self.client = OpenAI(
#             api_key=api_key,
#             base_url=base_url,
#         )
#         self.deployment_name = model_name

#         # Optional headers for OpenRouter rankings
#         self.extra_headers = {}
#         if site_url:
#             self.extra_headers["HTTP-Referer"] = site_url
#         if site_name:
#             self.extra_headers["X-Title"] = site_name

#     @backoff.on_exception(backoff.expo, (RateLimitError, OpenAIError, TypeError, ValueError), max_tries=5)
#     def send_request(self, model_name, prompt, max_tokens=30000, temperature=1, top_p=1.0, **kwargs):
#         """
#         Send a prompt to the specified model via the OpenRouter API.

#         Parameters:
#         - model_name (str): The name of the model to use (e.g., 'openai/gpt-3.5-turbo').
#         - prompt (str): The prompt to send to the model.
#         - max_tokens (int): The maximum number of tokens to generate.
#         - temperature (float): Sampling temperature.
#         - top_p (float): Nucleus sampling parameter.
#         - **kwargs: Additional parameters for the OpenAI API.

#         Returns:
#         - str: The generated response from the model.
#         """
#         try:
#             # Convert the prompt string into a chat message
#             messages = [{"role": "user", "content": prompt}]

#             # Use the chat completion endpoint
#             response = self.client.chat.completions.create(
#                 model=self.deployment_name,
#                 messages=messages,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 n=1,
#                 extra_headers=self.extra_headers,
#                 **kwargs
#             )

#             # Check if response is valid
#             if not response or not response.choices or not response.choices[0].message.content:
#                 raise ValueError("Invalid response received from the API")

#             generated_text = response.choices[0].message.content.strip()
#             return generated_text

#         except RateLimitError as e:
#             # The backoff decorator will handle retries
#             raise e

#         except (OpenAIError, TypeError, ValueError) as e:
#             # Raise exception to trigger retry
#             raise e
        

import requests
import backoff
from requests.exceptions import HTTPError, ConnectionError, Timeout

class OpenRouterClient:
    # Define a mapping of models to their specific provider preferences
    MODEL_PROVIDER_MAPPING = {
        "deepseek/deepseek-r1": {
            "order": ["Fireworks", "Together"],
            "allow_fallbacks": False
        },
        # Add other models and their provider configurations here if needed
    }

    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1",model_name=None, site_url=None, site_name=None):
        """
        Initialize the OpenRouterClient with the provided API key and base URL.
        
        Parameters:
        - api_key (str): Your OpenRouter API key.
        - base_url (str): The base URL for OpenRouter API. Default is 'https://openrouter.ai/api/v1'.
        - site_url (str): Optional. Site URL for rankings on openrouter.ai.
        - site_name (str): Optional. Site title for rankings on openrouter.ai.
        """
        self.deployment_name = model_name
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # Ensure no trailing slash

        # Optional headers for OpenRouter rankings
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if site_name:
            self.headers["X-Title"] = site_name

    @backoff.on_exception(
        backoff.expo,
        (HTTPError, ConnectionError, Timeout),
        max_tries=5,
        jitter=backoff.full_jitter
    )
    def send_request(self, model_name, prompt, max_tokens=30000, temperature=1, top_p=1.0, **kwargs):
        """
        Send a prompt to the specified model via the OpenRouter API.
        
        Parameters:
        - model_name (str): The name of the model to use (e.g., 'openai/gpt-3.5-turbo').
        - prompt (str): The prompt to send to the model.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): Sampling temperature.
        - top_p (float): Nucleus sampling parameter.
        - **kwargs: Additional parameters for the OpenRouter API.
        
        Returns:
        - str: The generated response from the model.
        """

        model_name = model_name or self.deployment_name

        # Prepare the payload
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1  # Number of completions to generate
        }

        # Inject provider preferences based on the model using the mapping
        provider_preferences = self.MODEL_PROVIDER_MAPPING.get(model_name)
        if provider_preferences:
            payload["provider"] = provider_preferences

        # Include any additional kwargs into the payload
        if kwargs:
            payload.update(kwargs)

        try:
            # Send the POST request to the OpenRouter API
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30  # Set a timeout for the request
            )

            # Raise an exception for HTTP error codes
            response.raise_for_status()

            # Parse the JSON response
            data = response.json()

            # Validate the response structure
            if not data or "choices" not in data or not data["choices"]:
                raise ValueError("Invalid response structure received from the API.")

            generated_text = data["choices"][0].get("message", {}).get("content", "").strip()
            if not generated_text:
                raise ValueError("Empty response content received from the API.")

            return generated_text

        except HTTPError as http_err:
            # Handle HTTP errors
            print(f"HTTP error occurred: {http_err}")  # You can use logging instead of print
            raise
        except ConnectionError as conn_err:
            # Handle connection errors
            print(f"Connection error occurred: {conn_err}")  # You can use logging instead of print
            raise
        except Timeout as timeout_err:
            # Handle timeout errors
            print(f"Timeout error occurred: {timeout_err}")  # You can use logging instead of print
            raise
        except ValueError as val_err:
            # Handle JSON decoding errors or invalid response structure
            print(f"Value error: {val_err}")  # You can use logging instead of print
            raise
        except Exception as err:
            # Handle any other exceptions
            print(f"An unexpected error occurred: {err}")  # You can use logging instead of print
            raise