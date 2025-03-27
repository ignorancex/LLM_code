import os
import replicate

os.environ["REPLICATE_API_TOKEN"]= 'key-for-replicate'

def assessment_def_block(
        target_word,
        sentences,
        lang,
        headwords,
        lm,
        prompt_placeholder="",
        response_placeholder="",
):

    if lang == 'Russian':
        requirements = "Создаваемое определение должно быть кратким. Разрешается максимум десять слов."
        instructions = [f'Представьте, что вы лексикограф, учитывая заглавное слово {target_word} на {lang}, напишите словарное определение его значения в следующих предложениях.']
        sentences_list = [f'{i+1}. {sen}.' for i, sen in enumerate([sentences[0]])]
    else:
        requirements = "Luodun määritelmän tulee olla ytimekäs. Enintään kymmenen sanaa on sallittu."        
        instructions = [f'Kuva, jonka olet sanakirjailija, saa otsikkosanan {target_word} kielellä {lang}, kirjoita sanakirjamäärittely sen merkityksestä seuraaviin lauseisiin.']
        sentences_list = [f'{i+1}. {sen}, jossa {w} on pääsana.' for i, (sen, w) in enumerate(zip([sentences[0]], [headwords[0]]))]

    prompt_pieces = instructions + sentences_list + [requirements]
    print('\n'.join(prompt_pieces))


    lm = replicate.run(
    "meta/llama-2-7b",
    stream=False,
    input={
        "top_k": 0,
        "top_p": 0.9,
        "prompt": '\n'.join(instructions + sentences_list),
        "system_prompt": requirements,
        "temperature": 0,
        "length_penalty": 1,
        "max_new_tokens": 30,
        "prompt_template": "{prompt}",
        "presence_penalty": 1.15
        },
    )

    return ''.join(lm).strip()
