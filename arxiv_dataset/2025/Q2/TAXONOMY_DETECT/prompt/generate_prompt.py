import os


def load_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = f.read()
    except:
        with open(file_path, "rb") as f:
            data = f.read().decode()
    return data


def generate_prompt(curr_input, prompt_lib_file):
    current_file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(current_file_path)
    prompt = load_file(os.path.join(dir_path, prompt_lib_file))
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()
