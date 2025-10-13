from openai import OpenAI
openai_client = None

# 目前建议简单任务4o-mini，复杂任务4o，老模型价格高效果一般
def chat_openai(prompt, model="gpt-4o-mini", temperature=0.7):
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(base_url="http://47.74.22.128:35040/openai/v1/", api_key="aaa")
    
    completion = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    message = completion.choices[0].message.content
    
    # 统计输入和输出token
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    
    # 计算资费
    if model == "gpt-4o":
        input_cost = input_tokens * 5.00 / 1000000
        output_cost = output_tokens * 15.00 / 1000000
    elif model == "gpt-4o-mini":
        input_cost = input_tokens * 0.150 / 1000000
        output_cost = output_tokens * 0.600 / 1000000
    else:
        input_cost = 0
        output_cost = 0
    
    cost = input_cost + output_cost
    return message, input_tokens, output_tokens, cost


import anthropic
anthropic_client = None

# 目前仅建议调用claude3.5 sonnet,年内3.5小杯haiku和大杯opus会更新
def chat_anthropic(prompt, model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=4096):
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = anthropic.Anthropic(base_url = "http://47.74.22.128:35040/anthropic/",api_key="aaa")
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response = message.content[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    
    if model == "claude-3-5-sonnet-20240620":
        input_cost = input_tokens * 3.00 / 1000000
        output_cost = output_tokens * 15.00 / 1000000
    else:
        input_cost = 0
        output_cost = 0
    cost = input_cost + output_cost
    return response, input_tokens, output_tokens, cost



if __name__ == "__main__":
    response, input_tokens, output_tokens, cost = chat_anthropic("hello")
    print(f"response: {response}")
    print(f"input_tokens: {input_tokens}")
    print(f"output_tokens: {output_tokens}")
    print(f"cost: {cost} 美元")
