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
1. 2020年少数民族高层次骨干人才计划考生登记表
2. 职业介绍所执照申请表
3. 医药费报销表
4. 入职登记表
5. 华东师范大学家庭经济困难学生调查及认定申请表"""
    else:
        print("Invalid theme")
        exit()

    # [CUSTOM] add your own theme here
    if theme == 'government':
        theme_input = "政府机构"
    else:
        print("Invalid theme")
        exit()


    prompt = f"""\
请创造出 {num_form} 个可能在中国出现的{theme_input}相关表单标题，标题需要明确表达该表单的用途，并确保标题的真实性。
请直接接续以下标题，不必再重复朗读:

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
    parser.add_argument("--file_name", type=str, default='example_SC.txt')
    args = parser.parse_args()

    main(args)