import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

from transformers import logging as hf_logging
from tokenizers import Encoding
from tokenizers import SentencePieceBPETokenizer
from tokenizers.normalizers import BertNormalizer
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from zsmt.lang_info import get_langs_d

def add_lang_special_toks(tokenizer):
    with open(os.path.join('models/tok_srct_lang', "langs"), "rb") as fp:
        languages: Dict[str, int] = pickle.load(fp)
        langs = sorted(set(languages.values()))
        special_tokens_dict = {'additional_special_tokens': langs}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

class TextProcessor:
    def __init__(self, tok_model_path: Optional[Union[str, Path]] = None,
                pretrained_name: Optional[Union[str, Path]] = None,
                src_lang_fam=False):
        self.languages = {}
        self.test = src_lang_fam
        tok_model_path = Path(tok_model_path) if tok_model_path else None
        if tok_model_path and pretrained_name:
            print('ERROR: cannot use both pretrained tokenizer and a specified path')
            os._exit(-1)
        if tok_model_path is not None:
            self.tokenizer = SentencePieceBPETokenizer(
                str(tok_model_path / "vocab.json"),
                str(tok_model_path / "merges.txt"),
            )
            with (tok_model_path / "langs").open("rb") as fp:
                self.languages: Dict[str, int] = pickle.load(fp)

            self.init_properties(self.languages)
        elif pretrained_name is not None:
            hf_logging.set_verbosity_error()

            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_name
            )

            if src_lang_fam:
                add_lang_special_toks(self.tokenizer)
            self.init_properties(self.languages)

            # monkeypatch some functions for compatability across `tokenizers` and `transformers`
            # version of tokenizers
            self.tokenizer.token_to_id = self.tokenizer.convert_tokens_to_ids
            self.tokenizer.id_to_token = self.tokenizer.convert_ids_to_tokens
            self.tokenizer.encode_batch = \
                lambda x: self.tokenizer.batch_encode_plus(x, add_special_tokens=False).encodings
            self.tokenizer.get_vocab_size = lambda: self.tokenizer.vocab_size
        else:
            self.tokenizer = None


    def init_properties(self, languages: Dict[str, str] = None):
        self.max_len = 512
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"
        self.sep_token = "</s>"
        self.bos = "<s>"
        self.languages = languages
        if languages:
            lang_families = set(languages.values())
            self.special_tokens = [self.pad_token, self.bos, self.unk_token, self.mask_token,
                                self.sep_token] + sorted(lang_families)
        else:
            self.special_tokens = [self.pad_token, self.bos, self.unk_token, self.mask_token,
                                self.sep_token]

    def train_tokenizer(self, paths: List[str], vocab_size: int, to_save_dir: str, languages: Dict[str, int]):
        self.tokenizer = SentencePieceBPETokenizer()
        bert_normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=False, lowercase=False)
        self.tokenizer._tokenizer.normalizer = bert_normalizer
        self.init_properties(languages)
        self.tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=5, special_tokens=self.special_tokens)
        self.save(directory=to_save_dir)

    def _tokenize(self, line) -> Encoding:
        return self.tokenizer.encode(line)

    def save(self, directory):
        self.tokenizer.save_model(directory)
        with open(os.path.join(directory, "langs"), "wb") as fp:
            pickle.dump(self.languages, fp)

    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.pad_token)

    def mask_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.mask_token)

    def unk_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.unk_token)

    def bos_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.bos)

    def sep_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.sep_token)

    def token_id(self, token: str) -> int:
        tok_id = self.tokenizer.token_to_id(token)
        if tok_id is None:
            return 0
        return tok_id

    def id2token(self, id: int) -> str:
        return self.tokenizer.id_to_token(id)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
