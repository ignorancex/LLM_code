"""
generate CuLP-style image prompts with ChatGPT
"""


# CuPL - Food-101
# ----------------------------------------------------------------------- #
def generate_cupl_1_prompt_for_food101(name):
    return f"Describe what {name} looks like"


def generate_cupl_2_prompt_for_food101(name):
    return f"Visually describe {name}"


def generate_cupl_3_prompt_for_food101(name):
    return f"How can you tell that the food in this photo is {name}?"


# CuPL - UCF-101
# ----------------------------------------------------------------------- #
def generate_cupl_1_prompt_for_ucf101(name):
    return f"Describe the action of {name}"


def generate_cupl_2_prompt_for_ucf101(name):
    return f"What does the act of {name} look like?"


def generate_cupl_3_prompt_for_ucf101(name):
    return f"What does a person doing {name} look like?"


def generate_cupl_4_prompt_for_ucf101(name):
    return f'Describe "{name}"'


def generate_cupl_5_prompt_for_ucf101(name):
    return f'Describe the action "{name}"'


# CuPL - CUB-200 / Birdsnap
# ----------------------------------------------------------------------- #
def generate_cupl_1_prompt_for_cub200(name):
    return f"Describe what the bird {name} looks like:"


def generate_cupl_2_prompt_for_cub200(name):
    return f"Describe the bird {name}:"


def generate_cupl_3_prompt_for_cub200(name):
    return f"What are the identifying characteristics of the bird {name}?"


# CuPL - SUN-324 / SUN-397
# ----------------------------------------------------------------------- #
def generate_cupl_1_prompt_for_sun324(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return f"Describle what {article} {name} looks like"


def generate_cupl_2_prompt_for_sun324(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return f"How can you identify {article} {name}?"


def generate_cupl_3_prompt_for_sun324(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return f"Describle a photo of {article} {name}"


# CuPL - ImageNet
# ----------------------------------------------------------------------- #
def generate_cupl_1_prompt_for_imagenet(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return "Describe what " + article + " " + name + " looks like"


def generate_cupl_2_prompt_for_imagenet(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return "How can you identify " + article + " " + name + "?"


def generate_cupl_3_prompt_for_imagenet(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return "What does " + article + " " + name + " look like?"


def generate_cupl_4_prompt_for_imagenet(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return "Describe an image from the internet of " + article + " "  + name


def generate_cupl_5_prompt_for_imagenet(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'a'
    if name[0] in vowels: article = 'an'
    return "A caption of an image of "  + article + " "  + name + ":"


CuPL_LANG_PROMPTS = {
    'food-101': [
        generate_cupl_1_prompt_for_food101,
        generate_cupl_2_prompt_for_food101,
        generate_cupl_3_prompt_for_food101,
        ],
    'ucf-101': [
        generate_cupl_1_prompt_for_ucf101,
        generate_cupl_2_prompt_for_ucf101,
        generate_cupl_3_prompt_for_ucf101,
        generate_cupl_4_prompt_for_ucf101,
        generate_cupl_5_prompt_for_ucf101,
    ],
    'cub-200': [
        generate_cupl_1_prompt_for_cub200,
        generate_cupl_2_prompt_for_cub200,
        generate_cupl_3_prompt_for_cub200,
    ],
    'sun-324': [
        generate_cupl_1_prompt_for_sun324,
        generate_cupl_2_prompt_for_sun324,
        generate_cupl_3_prompt_for_sun324,
    ],
    'imagenet': [
        generate_cupl_1_prompt_for_imagenet,
        generate_cupl_2_prompt_for_imagenet,
        generate_cupl_3_prompt_for_imagenet,
        generate_cupl_4_prompt_for_imagenet,
        generate_cupl_5_prompt_for_imagenet,
    ]
}