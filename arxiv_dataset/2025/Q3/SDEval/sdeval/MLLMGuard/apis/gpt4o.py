import base64
from openai import OpenAI
from models.base import Mllm
api = ""
client = OpenAI(
    base_url = "",
    api_key= api 
)





def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



class GPT4o(Mllm):
    def __init__(self, api_key, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time


    def evaluate(self, prompt, filepath):


        image_path = filepath

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="gemini-2.5-pro-preview-06-05",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output

class o4mini(Mllm):
    def __init__(self, api_key, engine="o4-mini", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time


    def evaluate(self, prompt, filepath):


        image_path = filepath

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output


class claude4(Mllm):
    def __init__(self, api_key, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time


    def evaluate(self, prompt, filepath):


        image_path = filepath

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output


class grok(Mllm):
    def __init__(self, api_key, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
        self.client = client
        self.temperature = temperature
        self.sleep_time = sleep_time


    def evaluate(self, prompt, filepath):


        image_path = filepath

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature # 自行修改温度等参数
        )
        # print(response)
        output = response.choices[0].message.content

        return output