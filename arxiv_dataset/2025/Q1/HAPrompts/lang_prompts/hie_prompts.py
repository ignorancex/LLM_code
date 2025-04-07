"""
generate HIE (cluster & tree) prompts with chatgpt using
(1) in-context learning 
(2) grouping of classes via 
    a. k-means clustering 
    b. given label hierarchy
"""

import json
from collections import defaultdict




def generate_prompt_summary(category_name):
    return f"""Q:summarize the following categories with one sentence: Salmon, Goldfish, Piranha, Zebra Shark, Whale Shark, Snapper, Swordfish, Bass, Trout?
A: this is a dataset of various fishes

Q: summarize the following categories with one sentence: Smartphone, Laptop, Piranha, Scanner, Refrigerator, Tiger, Bluetooth Speaker, Projector, Printer?
A: this dataset includes different electronic devices

Q: summarize the following categories with one sentence: Scott Oriole, Baird Sparrow, Black-throated Sparrow, Chipping Sparrow, House Sparrow, Grasshopper Sparrow
A: most categories in this dataset are sparrow

Q: summarize the following categories with one sentence: {category_name}?
A: """

def generate_prompt_given_overall_feature(category_name, over_all):
    return f"""Q: What are useful visual features for distinguishing a Clay Colored Sparrow in a photo in a dataset: This dataset consists of various sparrows?
A: There are several useful visual features to tell there is a Clay Colored Sparrow in a photo:
- a distinct pale crown stripe or central crown patch
- a dark eyeline and a pale stripe above the eye
- brownish-gray upperparts
- conical-shaped bill

Q: What are useful visual features for distinguishing a Zebra Shark in a photo in a dataset: Most categories in this dataset are sharks?
A: There are several useful visual features to tell there is a Zebra Shark in a photo:
- prominent dark vertical stripes or bands
- a sleek and slender body with a long, flattened snout and a distinctive appearance
- a tan or light brown base color on their body
- a long, slender tail with a pattern of dark spots and bands that extend to the tail fin
- dark edges of both dorsal fins

Q: What are useful features for distinguishing a {category_name} in a photo: {over_all}?
A: There are several useful visual features to tell there is a {category_name} in a photo:
- """


def generate_prompt_compare(categories_group: str, to_compare: str):
    return f"""Q: What are useful visual features for distinguishing Hooded Oriole from Scott Oriole, Baltimore Oriole in a photo
A: There are several useful visual features to tell there is a Hooded Oriole in a photo:
- distinctive bright orange or yellow and black coloration
- orange or yellow body and underparts
- noticeably curved downwards bill
- a black bib or "hood" that extends up over the head and down the back

Q: What are useful visual features for distinguishing a smartphone from television, laptop, scanner, printer in a photo?
A: There are several useful visual features to tell there is a smartphone in a photo:
- rectangular and much thinner shape
- a touchscreen, lacking the buttons and dials
- manufacturer's logo or name visible on the front or back of the device
- one or more visible camera lenses on the back

Q: What are useful features for distinguishing a {categories_group} from {to_compare} in a photo?
A: There are several useful visual features to tell there is a {categories_group} in a photo:
- """


def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    

def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]


def compose_hie_image_prompts(descriptors, name):
    before_text = ''
    between_text = ', '
    after_text = '.'
    image_prompts = []
    for des in descriptors:
        modify_descriptor = make_descriptor_sentence(des)
        image_prompts.append(f"{before_text}{name}{between_text}{modify_descriptor}{after_text}")
    return image_prompts


def load_initial_vcd_img_prompts(fpath):
    with open(fpath, 'r') as f:
        prompts = json.load(f)
    hie_prompts = defaultdict(list)
    for k, v in prompts.items():
        # {classname: [[vcd_prompts], ]}
        hie_prompts[k].append(v)
    return hie_prompts