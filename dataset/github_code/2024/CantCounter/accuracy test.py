import os
import openai

# 设置代理端口，否则连接不上api
os.environ["http_proxy"] = "127.0.0.1:7890"  # os.environ["http_proxy"] = "http://<代理ip>:<代理端口>"，这里的127.0.0.1就是本机电脑ip，1080是vpn的端口
os.environ["https_proxy"] = "127.0.0.1:7890"  # os.environ["https_proxy"] = "http://<代理ip>:<代理端口>"

openai.api_key = "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"   # 将自己的api_key填进去，获取地址：https://platform.openai.com/account/api-keys

while True:
  # 获取用户输入
  user_input = input("User：")

  # 检查是否输入了 “exit”，如果是则退出聊天
  if user_input.lower() == "exit":
    print("AI：好的，再见！")
    break

  # 获得ChatGPT的回复
  completion = openai.Completion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user",
       "content": user_input}
    ]
  )

  # 将回复进行输出
  print("AI:", completion.choices[0].message["content"])
