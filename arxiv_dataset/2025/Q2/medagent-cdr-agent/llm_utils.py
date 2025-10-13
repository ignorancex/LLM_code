import openai
from openai import OpenAI


class OpenaiEngine():
    def __init__(self, config):
        self.api_key = config["api_key"]
        self.temperature = config["temperature"]
        self.model = config["model"]

    def generate(self, prompt, max_new_tokens=2000, **kwargs):
        openai.api_key = self.api_key
        if isinstance(prompt, str):
            # Assume one turn dialogue
            prompt = [
                {"role": "user", "content": prompt},
            ]
        client = OpenAI(
            api_key=self.api_key
        )
        if self.model == 'o1-preview':
            response = client.chat.completions.create(
                model=self.model,
                messages=prompt,
                **kwargs,
            )
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=prompt,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                **kwargs,
            )
        return [response.choices[0].message.content.strip()]

    def embed(self, text):
        openai.api_key = self.api_key
        client = OpenAI(
            api_key=self.api_key
        )
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

