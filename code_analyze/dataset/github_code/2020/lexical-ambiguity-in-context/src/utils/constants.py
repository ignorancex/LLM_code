import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scripts = {
    'mr': ['Devanagari'],
    'ba': ['Cyrillic', 'Arabic'],
    'af': ['Latin', 'Arabic'],
    'is': ['Latin'],
    'ceb': ['Latin'],
    'sw': ['Latin', 'Arabic'],
    'jv': ['Latin', 'Javanese'],
    'ne': ['Devanagari'],
    'yo': ['Latin', 'Arabic'],
    'cy': ['Latin'],
    'ta': ['Tamil'],
    'vo': ['Latin'],
    'ja': None,
    'pt': ['Latin'],
    'ka': ['Georgian'],
    'az': None,
    'et': ['Latin'],
    'my': ['Myanmar'],
    'fa': ['Cyrillic', 'Arabic'],
    'he': ['Hebrew'],

    'bn': ['Bengali'],
    'eu': ['Latin'],
    'id': ['Latin'],
    'kn': ['Kannada', 'Brahmi'],
    'ml': ['Malayalam'],
    'tl': ['Latin'],
    'tt': ['Latin', 'Cyrillic', 'Arabic'],
}


LANG_CODE = {
    'en': 'eng',
    'fi': 'fin',
    'ja': 'jpn',
    'pt': 'por',
    'es': 'spa',

    'fr': 'fra',
    'th': 'tha',
    'nb': 'nob',
    'nn': 'nnb',
    'gl': 'glg',
    'id': 'ind',
    'fa': 'fas',
    'eu': 'eus',
    'ar': 'arb',
    'ca': 'cat',

    'it': 'ita',
    'he': 'heb',
    'sq': 'als',
    'zh': 'cmn',
    'da': 'dan',
    'pl': 'pol',
    'ms': 'zsm',
}
