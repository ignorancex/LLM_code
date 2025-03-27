# from googletrans import Translator
# translator = Translator()
# ret = translator.translate('pakistan ka ghra tauq he Pakistan Israel ko tasleem nahein kerta Isko Palestine kehta he', dest='en')

# print(ret.text)
# print(gs.translate('pakistan ka ghra tauq he Pakistan Israel ko tasleem nahein kerta Isko Palestine kehta he', 'en'))


from googletrans import Translator
translator = Translator()

translations = translator.translate(['The quick brown fox', 'jumps over', 'the lazy dog'], dest='ko')
for translation in translations:
    print(translation.origin, ' -> ', translation.text)