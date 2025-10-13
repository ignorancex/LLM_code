import json

from tqdm.auto import tqdm


all_captions = {}
attribute_captions = {
    "5_o_Clock_Shadow": (2, ' with a five o\'clock shadow'),
    "Arched_Eyebrows": (2, ' with arched eyebrows'),
    "Attractive": (1, 'n attractive '),
    "Bags_Under_Eyes": (2, ' with bags under their eyes'),
    "Bald": (1, ' bald '),
    "Bangs": (2, ' wearing bangs'),
    "Big_Lips": (2, ' with big lips'),
    "Big_Nose": (2, ' with a big nose'),
    "Black_Hair": (2, ' with black hair'),
    "Blond_Hair": (2, ' with blond hair'),
    "Blurry": (0, ' blurry '),
    "Brown_Hair": (2, ' with brown hair'),
    "Bushy_Eyebrows": (2, ' with bushy eyebrows'),
    "Chubby": (1, ' chubby '),
    "Double_Chin": (2, ' with a double chin'),
    "Eyeglasses": (2, ' wearing eyeglasses'),
    "Goatee": (2, ' wearing a goatee'),
    "Gray_Hair": (2, ' with gray hair'),
    "Heavy_Makeup": (2, ' wearing heavy makeup'),
    "High_Cheekbones": (2, ' with high cheekbones'),
    "Male": None,
    "Mouth_Slightly_Open": (2, ' with their mouth slightly open'),
    "Mustache": (2, ' wearing a mustache'),
    "Narrow_Eyes": (2, ' with narrow eyes'),
    "No_Beard": (1, ' beardless '),
    "Oval_Face": (2, ' with an oval face'),
    "Pale_Skin": (2, ' with pale skin'),
    "Pointy_Nose": (2, ' with a pointy nose'),
    "Receding_Hairline": (2, ' with a receding hairline'),
    "Rosy_Cheeks": (2, ' with rosy cheeks'),
    "Sideburns": (2, ' wearing sideburns'),
    "Smiling": (2, ' smiling'),
    "Straight_Hair": (2, ' with straight hair'),
    "Wavy_Hair": (2, ' with wavy hair'),
    "Wearing_Earrings": (2, ' wearing earrings'),
    "Wearing_Hat": (2, ' wearing a hat'),
    "Wearing_Lipstick": (2, ' wearing lipstick'),
    "Wearing_Necklace": (2, ' wearing a necklace'),
    "Wearing_Necktie": (2, ' wearing a necktie'),
    "Young": (1, ' young ')
}

target_attributes = list(attribute_captions.keys())

order_1 = {'Attractive': 0, 'Young': 1, 'Chubby': 2, 'No_Beard': 3, 'Bald': 4}

for ta1, caption_instructions_1 in tqdm(attribute_captions.items()):
    all_captions[ta1] = {}

    if isinstance(caption_instructions_1, tuple):
        base_caption = 'A{}photo of a{}person{}.'
        dummy_texts = [' ', ' ', '']
        text_pos, additional_text = caption_instructions_1
        dummy_texts[text_pos] = additional_text
        base_caption = base_caption.format(*dummy_texts)
    else:
        assert ta1 == 'Male'
        base_caption = 'A photo of a man.'

    for ta2, caption_instructions_2 in attribute_captions.items():
        if ta1 == ta2:
            continue

        if isinstance(caption_instructions_2, tuple):
            text_pos, additional_text = caption_instructions_2
            if text_pos == 0:
                assert ta2 == 'Blurry'
                base_caption_words = base_caption.split(' ')
                caption = ' '.join(base_caption_words[:1] + ['blurry'] + base_caption_words[1:])
            elif text_pos == 1:
                base_caption_words = base_caption.split(' ')
                if caption_instructions_1 is not None and caption_instructions_1[0] == 1:
                    if order_1[ta1] > order_1[ta2]:
                        caption = ' '.join(base_caption_words[:4] + [additional_text.strip()] + base_caption_words[4:])
                    else:
                        caption = ' '.join(base_caption_words[:5] + [additional_text.strip()] + base_caption_words[5:])
                else:
                    caption = ' '.join(base_caption_words[:4] + [additional_text.strip()] + base_caption_words[4:])
                if ta2 == 'Attractive':
                    caption = caption.replace(' a n ', ' an ')
            elif text_pos == 2:
                if caption_instructions_1 is not None and caption_instructions_1[0] == 2:
                    ta_1_first_word = caption_instructions_1[1].strip().split(' ')[0]
                    ta_2_first_word = additional_text.strip().split(' ')[0]
                    if ta_1_first_word == 'with':
                        if ta_2_first_word == 'with':
                            caption = base_caption[:-1] + additional_text.replace('with', 'and') + '.'
                        elif ta_2_first_word in ['wearing', 'smiling']:
                            caption = base_caption[:-1] + additional_text + '.'
                        else:
                            raise ValueError('Unexpected attribute text first word.')
                    elif ta_1_first_word == 'wearing':
                        if ta_2_first_word == 'with':
                            base_caption_words = base_caption.split(' ')
                            caption = ' '.join(base_caption_words[:5] + [additional_text.strip()] + base_caption_words[5:])
                        elif ta_2_first_word == 'wearing':
                            caption = base_caption[:-1] + additional_text.replace('wearing', 'and') + '.'
                        else:
                            assert ta2 == 'Smiling'
                            base_caption_words = base_caption.split(' ')
                            caption = ' '.join(base_caption_words[:5] + [additional_text.strip() + ' and'] + base_caption_words[5:])
                    else:
                        assert ta1 == 'Smiling'
                        if ta_2_first_word == 'with':
                            base_caption_words = base_caption.split(' ')
                            caption = ' '.join(base_caption_words[:5] + [additional_text.strip()] + base_caption_words[5:])
                        elif ta_2_first_word == 'wearing':
                            caption = base_caption[:-1] + ' and' + additional_text + '.'
                        else:
                            raise ValueError('Unexpected attribute text first word.')
                else:
                    caption = base_caption[:-1] + additional_text + '.'
            else:
                raise ValueError('Unexpected attribute position.')
        else:
            assert ta2 == 'Male'
            caption = base_caption.replace('person', 'man')

        all_captions[ta1][ta2] = caption

with open('gt_captions.json', 'w') as f:
    json.dump(all_captions, f, indent='  ')
