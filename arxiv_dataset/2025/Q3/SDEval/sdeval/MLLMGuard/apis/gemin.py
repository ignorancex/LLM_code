import requests
api = "sk-B0gWjAvaUkrgEHNsrHQmAwyBPDdUOEgTXOmJHXtnbxTeEBjb"


headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

data = {
    'model': 'gemini-1.5-flash-preview-0514',
    'messages': [{'role': 'user', "content": [
                {
                    "type": "text",
                    "text": "Who are you and 这张图片的图标是个什么动物？"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/dianping/cat/raw/master/cat-home/src/main/webapp/images/logo/cat_logo03.png"
                    }
                }
            ]}],
}

response = requests.post(url, headers=headers, json=data)

print(response.json())

class gemini(Mllm):
    def __init__(self, api_key, engine="gpt-4o", temperature=0, sleep_time=10) -> None:
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