import os
import yaml
import glob
import platform




def get_sub_dir_list(home_dir):
    host_platform = platform.system()
    sub_dir_list = []
    if host_platform == 'Windows':
        split_char = '\\'
    else:
        split_char = '/'

    for sub_dir in glob.glob(os.path.join(home_dir, '*')):
        sub_dir_name = sub_dir.split(split_char)
        sub_dir_list.append(sub_dir_name[-1])
    return sub_dir_list


def get_filename_list(dir,regexp):

    host_platform = platform.system()
    file_name_list = []

    if os.path.exists(dir)==False:
        print(f'Error, directory {dir} not exist, force to exit')
        exit()

    if host_platform=='Windows':
        split_char = '\\'
    else:
        split_char = '/'

    for file_path in glob.glob(os.path.join(dir,regexp)):

        filename = file_path.split(split_char)
        file_name_list.append(filename[-1])

    return file_name_list


def load_config(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_api_key(config_path, LLM):
    """
        LLM: 
            openai -> chatgpt
            anthropic -> claude
            google -> gemini
    """
    config = load_config(config_path)

    api_key_path = config[f'{LLM}-key']
    with open(api_key_path, 'r') as f:
        api_key = f.readline()
    return api_key

# EOF