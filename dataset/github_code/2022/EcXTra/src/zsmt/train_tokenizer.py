import json
import os
from argparse import ArgumentParser
from typing import Optional, List

from zsmt.lang_info import get_langs_d
from zsmt.textprocessor import TextProcessor


def get_tokenizer(train_paths: Optional[List[str]] = None,
                  model_path: Optional[str] = None, vocab_size: Optional[int] = None,
                  lang_path: Optional[str] = None) -> TextProcessor:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    langs = {}
    if lang_path:
        print('Loading languages dict...')
        langs = get_langs_d(lang_path)
    else:
        print('Not using languages dict')

    print("Training Tokenizer...")
    text_processor = TextProcessor()
    text_processor.train_tokenizer(
        paths=train_paths, vocab_size=vocab_size, to_save_dir=model_path, languages=langs)
    print("done!")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data", dest="data_paths", help="Path to the data folder",
                        nargs='+')
    parser.add_argument("--vocab", dest="vocab_size",
                      help="Vocabulary size", type=int, default=30000)
    parser.add_argument("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_argument("--lang-info", dest="lang_info_path", help="path to language info file", default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    tokenizer = get_tokenizer(train_paths=args.data_paths,
                              model_path=args.model_path, vocab_size=args.vocab_size,
                              lang_path=args.lang_info_path)
