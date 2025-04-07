"""
generate VCD-style image promtps with ChatGPT
using their in-context learning language prompts
"""

def generate_vcd_llm_prompt(category_name):
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""


def vcd_stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]


def make_descriptor_sentence(descriptor):
    # remove leading & trailing spaces in descriptors
    descriptor = descriptor.strip()
    # make descriptor and classname into a natural sentence
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def compose_vcd_image_prompts(descriptors, name):
    between_text = ', '
    after_text = '.'
    image_prompts = []
    for des in descriptors:
        modify_descriptor = make_descriptor_sentence(des)
        image_prompts.append(f"{name}{between_text}{modify_descriptor}{after_text}")
    return image_prompts


