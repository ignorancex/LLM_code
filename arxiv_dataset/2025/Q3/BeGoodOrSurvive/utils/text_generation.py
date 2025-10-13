from openai import OpenAI

client = OpenAI()

model = "gpt-4o"

def generate_text(prompt, system_message, max_length=512, temperature=1.5, top_p=.95):
    # Generate text
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": str(system_message)},
            {
            "role": "user",
            "content": prompt
            }
        ],
        max_tokens = max_length,
        temperature = temperature,
        top_p = top_p
    )
    # Decode and return generated text
    response_text = response.choices[0].message.content
    return response_text
