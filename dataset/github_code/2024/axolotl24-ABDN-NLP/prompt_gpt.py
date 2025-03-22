import os
from guidance import models, system, user, assistant, gen

os.environ["OPENAI_API_KEY"]= 'key-for-openai'

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
    if lang == 'Finnish':
        requirements = "Luodun määritelmän tulee olla ytimekäs. Enintään kymmenen sanaa on sallittu."        
        instructions = [f'Kuva, jonka olet sanakirjailija, saa otsikkosanan {target_word} kielellä {lang}, kirjoita sanakirjamäärittely sen merkityksestä seuraaviin lauseisiin.']
        sentences_list = [f'{i+1}. {sen}, jossa {w} on pääsana.' for i, (sen, w) in enumerate(zip([sentences[0]], [headwords[0]]))]
    if lang == 'German':
        requirements = "Das erstellte Definition sollte kurz sein. Maximal zehn Wörter erlaubt."
        instructions = [f'Stellen Sie sich vor, Sie sind ein Lexikograf und geben das Schlüsselwort {target_word} auf {lang} an. Schreiben Sie eine Wörterbuchdefinition seines Bedeutung in den folgenden Sätzen.']
        sentences_list = [f'{i+1}. {sen}.' for i, sen in enumerate([sentences[0]])]

    prompt_pieces = instructions + sentences_list + [requirements] + ["gen('definition', max_tokens=50, stop='.')"]
    print('\n'.join(prompt_pieces))

    with system():
        lm += requirements

    with user():
        lm += '\n'.join(instructions + sentences_list)
                    
    with assistant():
        lm += gen('definition', max_tokens=50, stop='.')
    
    return lm["definition"].strip()

