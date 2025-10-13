import os
import argparse
from openai import OpenAI 
from PIL import Image
import base64
import io

def _encode_image_(image_path):
    
    image = Image.open(image_path)

    # 计算缩放后的尺寸
    width, height = image.size
    max_size = 512
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_image = image
    
    # 确定保存格式（优先使用原图格式，默认JPEG）
    img_format = resized_image.format if resized_image.format else 'JPEG'
    img_format = img_format.upper()
    
    # 处理图像模式以适应格式要求
    if img_format in ('JPEG', 'JPG'):
        if resized_image.mode in ('RGBA', 'LA', 'P'):
            # 透明背景转白色
            background = Image.new('RGB', resized_image.size, (255, 255, 255))
            mask = resized_image.split()[3] if resized_image.mode == 'RGBA' else None
            background.paste(resized_image, mask=mask)
            resized_image = background
        elif resized_image.mode != 'RGB':
            resized_image = resized_image.convert('RGB')
    
    # 将图像保存到内存中的字节流
    byte_stream = io.BytesIO()
    save_params = {'format': img_format}
    if img_format == 'JPEG':
        save_params['quality'] = 85  # 优化JPEG质量
    resized_image.save(byte_stream, **save_params)
    byte_data = byte_stream.getvalue()
    
    # 返回Base64编码结果
    return base64.b64encode(byte_data).decode('utf-8')


class OpenAI_API:

    def __init__(self, args):
        self.client = OpenAI( 
        api_key="",
        base_url="",
        )
        print(f"OpenAI API initialized with url: {self.client.base_url}")

        self.model_name = args.model_path.split("/")[-1]

        print(f"Model name: {self.model_name}")
        self.args = args

    def inference(self, prompt_batch, image_path_batch):
        messages = []
        for text, image_path in zip(prompt_batch, image_path_batch):
            base64_image = _encode_image_(image_path)
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                },
                {"type": "text", "text": text},
            ]
            me=[
                # {"role": "system", "content": "You are a medical specialist with expertise in medical knowledge across various diseases."},
                {"role": "user", "content": content}
        ]
        messages.append(me)

        responses = []
        for mes in messages:
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=mes,
                    temperature=0,
                    max_tokens=8192,
                )
            except Exception as e:
                responses.append(f"Error: {str(e)}")
                continue
            responses.append(res.choices[0].message.content)

        return responses
