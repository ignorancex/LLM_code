import base64
import csv
import openai
from openai import OpenAI
import os
import base64
import json


api_key = 'sk-8t6eT7HAW18ozkkzD7Cc25F86cEc46C1Be8aEe53CfB5F701'
# os.environ["OPENAI_BASE_URL"] = "https://api.yesapikey.com/v1"
client = OpenAI(
    base_url = 'https://api.yesapikey.com/v1',
    api_key = api_key
)

MODEL='gpt-4o'

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def ChatGpt(prompt,base64_image):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Analyze the image and describe what you see."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )
    return response.choices[0].message.content
# prompt= "Tell me the main instance in the image, only few words, format like: 'main instance: .'. And describe this image containing the main instance and its style to generate a succinct yet informative description. The description should be useful for AI to re-generate the image. The description should be no more than four sentences. Remember do not exceed 4 sentences."
prompt= "Tell me the main instance in the image, only one or two words, format like: 'main instance: .'.  And Describe this image containing the main instance and its style to generate a succinct yet informative description in one or two sentences. The description should be useful for AI to re-generate the image. The description should be no more than 25 words. Remember do not exceed 25 words and must contain the main instance in the sentence."
# with open('prompt_0.txt', 'r', encoding='utf-8') as file:
#     prompt = file.read()
    #print(prompt)
    
# json_file_path = 'output.json'
 
# 使用with语句确保文件正确关闭
# with open(json_file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
 

# 现在data是一个字典，你可以遍历它
# for key,item in data.items():
output_file = "output_2.csv"
root = "/private/task/linyijing/dataset/val2017"
with open(output_file, 'a', newline='') as f: 
    writer = csv.writer(f)
    # writer.writerow(["path", "text"])    
    i = 0
    for img in os.listdir(root):
        i += 1
        if i < 2696:
            continue
        print(i)
        img_path = os.path.join(root, img)
        # img_path = "/private/task/linyijing/dataset/Coco2017/images_512_512/0.png"
        # 检查是否是文件，排除非文件项（例如文件夹）
        if os.path.isfile(img_path):
            base64_image = encode_image(img_path)  # 将图片编码为base64
            response = ChatGpt(prompt, base64_image)  # 调用ChatGPT接口生成响应

            # 写入数据并立即刷新到文件
            writer.writerow([img_path, response])
            f.flush()