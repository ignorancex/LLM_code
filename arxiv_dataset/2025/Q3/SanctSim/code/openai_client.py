import openai
import backoff
from openai import OpenAIError, RateLimitError

class OpenAIClient:
    def __init__(self, api_key, model_name, reasoning_effort=None):
        """
        Initialize the OpenAIClient with your API key and desired model.
        
        Parameters:
        - api_key (str): Your OpenAI API key.
        - model_name (str): The model to use (e.g., "o1-mini" or "o3-mini").
        - reasoning_effort (str, optional): Reasoning effort (only used with O3-mini models).
        """
        openai.api_key = api_key
        self.model_name = model_name
        self.deployment_name = model_name
        self.total_cost = 0.0  # Accumulate cost across calls
        self.reasoning_effort = reasoning_effort

    def is_o3_mini_model(self, model_name):
        """
        Check if the model is an O3-mini variant.
        
        Parameters:
        - model_name (str): The model name to check.
        
        Returns:
        - bool: True if the model is an O3-mini variant, False otherwise.
        """
        return model_name.lower().startswith("o3-mini")

    @backoff.on_exception(backoff.expo, RateLimitError)
    def send_request(self, model_name, prompt, max_tokens=30000, temperature=1, top_p=1.0, reasoning_effort='medium', **kwargs):
        """
        Send a prompt to the OpenAI ChatCompletion API.
        
        Parameters:
        - model_name (str): The model to use.
        - prompt (str): The prompt to send to the model.
        - max_tokens (int): Maximum number of tokens to generate.
        - temperature (float): Sampling temperature.
        - top_p (float): Nucleus sampling parameter.
        - reasoning_effort (str, optional): Reasoning effort (only used with O3-mini models).
        - **kwargs: Additional keyword arguments for the API call.
        
        Returns:
        - str: The generated response from the model.
        """
        # Use the provided reasoning_effort or fall back to the instance-level one
        effective_reasoning_effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort

        try:
            # Build the messages payload
            messages = [{"role": "user", "content": prompt}]
            params = {
                "model": model_name,  # Use the provided model_name parameter
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "n": 1,
            }
            
            # Include max tokens parameter - use max_completion_tokens instead of max_tokens
            # as it's required for newer models
            if max_tokens:
                params["max_completion_tokens"] = max_tokens
                
            # Only include reasoning_effort for O3-mini models
            if self.is_o3_mini_model(model_name) and effective_reasoning_effort is not None:
                params["reasoning_effort"] = effective_reasoning_effort

            # Include any additional parameters
            params.update(kwargs)

            # Call the ChatCompletion endpoint
            response = openai.chat.completions.create(**params)

            generated_text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            #print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")

            # Calculate and accumulate cost
            cost = self.calculate_price(prompt_tokens, completion_tokens)
            self.total_cost += cost

            return generated_text

        except RateLimitError as e:
            # backoff will retry
            raise e
        except OpenAIError as e:
            raise Exception(f"An error occurred: {str(e)}")

    def calculate_price(self, prompt_tokens, completion_tokens):
        """
        Calculate the estimated cost for the API call based on token usage.
        
        Parameters:
        - prompt_tokens (int): Number of tokens in the prompt.
        - completion_tokens (int): Number of tokens in the completion.
        
        Returns:
        - float: Approximate cost in USD.
        
        Note: Prices are defined per 1,000 tokens.
        """
        # Pricing in USD per 1,000 tokens for each model.
        # Updated so that both "o1-mini" and "o3-mini" use the same pricing.
        model_pricing = {
            "o1-mini": {
                "input": 1.10 / 1000,   # $1.10 per 1M tokens => $0.0011 per 1,000 tokens
                "output": 4.40 / 1000,  # $4.40 per 1M tokens => $0.0044 per 1,000 tokens
            },
            "o3-mini": {
                "input": 1.10 / 1000,
                "output": 4.40 / 1000,
            },
            "o3-mini-2025-01-31": {
                "input": 1.10 / 1000,
                "output": 4.40 / 1000,
            },
            # Add additional models here as needed.
        }

        # Retrieve pricing for the current model
        if self.model_name in model_pricing:
            pricing = model_pricing[self.model_name]
            input_price = pricing["input"]
            output_price = pricing["output"]
        else:
            input_price = 0.0
            output_price = 0.0

        input_cost = (prompt_tokens / 1000) * input_price
        output_cost = (completion_tokens / 1000) * output_price
        return input_cost + output_cost

    def get_total_cost(self):
        """
        Return the total accumulated cost of API calls made via this client.
        
        Returns:
        - float: Total cost in USD.
        """
        return self.total_cost

# Example usage:
if __name__ == "__main__":
    # Replace with your API key and desired model ("o1-mini" or "o3-mini")
    client = OpenAIClient(api_key="YOUR_API_KEY", model_name="o3-mini")
    
    # Example with O3-mini (will include reasoning_effort)
    response_text = client.send_request(
        model_name="o3-mini",
        prompt="Explain quantum entanglement in simple terms.",
        max_tokens=500,
        temperature=0.7,
        reasoning_effort="high"  # Will be included only for O3-mini models
    )
    
    print("Response:", response_text)
    
    # Example with O1-mini (will NOT include reasoning_effort)
    response_text = client.send_request(
        model_name="o1-mini",
        prompt="Explain quantum entanglement in simple terms.",
        max_tokens=500,
        temperature=0.7,
        reasoning_effort="high"  # Will be ignored for non-O3-mini models
    )
    
    print("Total cost so far: $", client.get_total_cost())