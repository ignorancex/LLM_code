import openai
import argparse
import os

from global_config import OPENAI_KEY


# Initialize the OpenAI API client
openai.api_key = OPENAI_KEY


def main(args):
    num_form = args.num_form
    theme = args.theme

    # [CUSTOM] add your own theme here
    if theme == 'government':
        input_to_cont = """\
1. 內政部戶政司身分證遺失申請表
2. 勞動部勞動基準法違反檢舉表
3. 交通部公路總局車輛異動登記表
4. 經濟部工業局新創企業補助申請表
5. 台北市政府社會局低收入戶補助申請表"""
    else:
        print("Invalid theme")
        exit()

    # [CUSTOM] add your own theme here
    if theme == 'government':
        theme_input = "政府機構"
    else:
        print("Invalid theme")
        exit()


    prompt = f"""\
請創造出 {num_form} 個可能在台灣出現的{theme_input}相關表單標題，標題需要明確表達該表單的用途，並確保標題的真實性。
直接接續以下內容:

{input_to_cont}
"""

    response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
    response = response.choices[0].message.content
    print(response)
    
    # write to txt file
    save_dir = "headers/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, args.file_name)
    with open(save_path, 'w', encoding="utf8") as f:
        f.write(response)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_form", type=int, default=10)
    parser.add_argument("--theme", type=str, default='government')
    parser.add_argument("--file_name", type=str, default='example_TC.txt')
    args = parser.parse_args()

    main(args)