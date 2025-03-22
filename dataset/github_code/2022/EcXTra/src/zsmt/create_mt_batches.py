import datetime
import marshal
import os
import sys
import time

from argparse import ArgumentParser
from tokenizers import SentencePieceBPETokenizer
from typing import List, Optional, Union
from tqdm import tqdm

from zsmt.utils import get_token_id
from zsmt.textprocessor import TextProcessor

def filter_on_len(list_ids, min_seq_len, max_seq_len):
    valid_ids = []
    for i, ids in enumerate(list_ids):
        if min_seq_len < len(ids) < max_seq_len:
            valid_ids.append(i)
    return set(valid_ids)

def batch_encode(lines, tokenizer, batch_size=2000000):
    # regular encoding call uses too much memory when lines is large -- call it in batches
    encoded_ids = []
    if isinstance(tokenizer, SentencePieceBPETokenizer):
        bos = tokenizer.token_to_id('<s>')
        eos = tokenizer.token_to_id('</s>')

    for i in range(0, len(lines), batch_size):
        if isinstance(tokenizer, SentencePieceBPETokenizer):
            ids_curr = [[bos] + encoding.ids + [eos] for encoding in \
                tokenizer.encode_batch(lines[i:i+batch_size])]
        else:
            ids_curr = tokenizer(lines[i:i+batch_size])['input_ids']
        encoded_ids.extend(ids_curr)
        print(f'encoded {len(ids_curr)} lines')
    return encoded_ids


def write(output_file: str, src_txt_file: Union[str, List[str]],
          dst_txt_file : Optional[Union[str, List[str]]] = None, srct_txt_file: str = None,
          tp_dst: Optional[TextProcessor] = None, tp_srct : Optional[TextProcessor] = None,
          shallow: bool = False, lang_lines_path: Optional[str] = None,
          min_seq_len=-1, max_seq_len=-1, max_sents=-1,
          sort=True, src_lang_fam=False, quiet=False):
    """
    There are scenarios for which the input comes from two streams such as original text and
    transliterated text, or we want to use different encoders. In these cases, srct_txt_file serves
    as the second file.
    NOTE: We have up to 3 tokenizers -- one for src, srct, and dst.

    Parameters:
        src_lang_fam - if True, and lang_lines_path is specified, adds lang fam tokens to source sentences
    """
    if quiet:
        old_stdout = sys.stdout
        f_null = open(os.devnull, 'w')
        sys.stdout = f_null

    # if only 1 TextProcessor was passed in, use it for both srct and dst
    tp_dst = tp_dst or tp_srct
    tp_srct = tp_srct or tp_dst

    dst_tokenizer = tp_dst.tokenizer

    if srct_txt_file is not None:
        srct_tokenizer = tp_srct.tokenizer

    # set src_tokenizer
    if shallow:
        src_tokenizer = tp_dst.tokenizer
    else: # using XLM-Roberta encoder
        tp_src = TextProcessor(pretrained_name='xlm-roberta-base',
                               src_lang_fam=src_lang_fam)
        src_tokenizer = tp_src.tokenizer

    if isinstance(src_txt_file, list):
        # TODO: also support taking srct_lines directly
        src_lines = src_txt_file
    else:
        print(datetime.datetime.now(), "Reading source lines!")
        with open(src_txt_file, "r") as s_fp:
            src_lines = list(map(lambda x: x.strip(), s_fp))
    if max_sents != -1:
        print(f'using {max_sents}/{len(src_lines)} lines')
        src_lines = src_lines[:max_sents]
    print(datetime.datetime.now(), "Reading target lines!")

    dst_lines = None
    if isinstance(dst_txt_file, list):
        dst_lines = dst_txt_file
    elif dst_txt_file is not None:
        with open(dst_txt_file, "r") as d_fp:
            dst_lines = list(map(lambda x: x.strip(), d_fp))
        print("Number of parallel sentences:", len(dst_lines))

        assert len(src_lines) == len(dst_lines)

    lang_lines = None
    if lang_lines_path:
        print(datetime.datetime.now(), 'Reading language lines!')
        if lang_lines_path.startswith('<') and lang_lines_path.endswith('>'):
            lang_lines = [lang_lines_path] * len(src_lines)
        else:
            with open(lang_lines_path, 'r') as l_fp:
                lang_lines = list(map(lambda x: x.strip(), l_fp))
        lang2id = {}
        srct_bos_ids = [get_token_id(x, tp_srct, lang2id) for x in lang_lines]
        assert len(lang_lines) == len(src_lines)
    else:
        print('Not using language lines!')
        bos_id = tp_srct.bos_token_id()
        srct_bos_ids = [bos_id] * len(src_lines)

    if srct_txt_file is not None:
        print(datetime.datetime.now(), "Reading source-transliterated lines!")
        with open(srct_txt_file, "r") as st_fp:
            srct_lines = list(map(lambda x: x.strip(), st_fp))
        assert len(src_lines) == len(srct_lines)

    print(datetime.datetime.now(), "Encoding source lines!")
    start = time.time()
    if shallow:
        src_ids = [encoding.ids for encoding in src_tokenizer.encode_batch(src_lines)]
    else:
        if src_lang_fam:
            assert lang_lines is not None # TODO: move this check to main()
            lang2id = {}
            src_bos_ids = [get_token_id(x, tp_src, lang2id) for x in lang_lines]
            # TODO: replace all instances of tokenizer.encode_batch() with batch_encode()
            src_ids = [[src_bos_id] + encoding.ids + [tp_src.sep_token_id()] for src_bos_id, encoding in
                        zip(src_bos_ids, src_tokenizer.encode_batch(src_lines))]
        else:
            src_ids = batch_encode(src_lines, src_tokenizer)

    print(f'took {time.time() - start}s')
    print('EX 1')
    print(src_lines[0])
    print(src_tokenizer.decode(src_ids[0]))

    # for debugging bidirectional TODO: remove
    print('EX 2')
    idx_mid = len(src_lines)//2
    print(src_lines[idx_mid])
    print(src_tokenizer.decode(src_ids[idx_mid]))

    if dst_txt_file is not None:
        print(datetime.datetime.now(), "Encoding dest lines!")
        start = time.time()
        dst_ids = batch_encode(dst_lines, tp_dst.tokenizer)
        print(f'took {time.time() - start}s')
        print('DEST EX 1')
        print(dst_lines[0])
        print(dst_tokenizer.decode(dst_ids[0]))
        print('DEST EX 2')
        print(dst_lines[idx_mid])
        print(dst_tokenizer.decode(dst_ids[idx_mid]))
    else:
        fixed_output = [tp_dst.token_id(tp_dst.bos)]
        dst_ids = [fixed_output] * len(src_ids)

    if srct_txt_file is not None:
        print(datetime.datetime.now(), "Encoding source-translitered lines!")
        srct_ids = [[srct_bos_id] + encoding.ids + [tp_srct.sep_token_id()] for srct_bos_id, encoding in
                    zip(srct_bos_ids, tqdm(srct_tokenizer.encode_batch(srct_lines)))]
        print(srct_lines[0])
        print(srct_tokenizer.decode(srct_ids[0]))

    if max_seq_len != -1 or min_seq_len != -1:
        print(datetime.datetime.now(), f'filtering to lengths ({min_seq_len}, {max_seq_len})')
        orig_len = len(src_ids)
        filtered_ids = filter_on_len(src_ids, min_seq_len, max_seq_len)
        if srct_txt_file is not None:
            filtered_ids_srct = filter_on_len(srct_ids, min_seq_len, max_seq_len)
            filtered_ids = filtered_ids & filtered_ids_srct
            srct_ids = [x for i, x in enumerate(srct_ids) if i in filtered_ids]
        src_ids = [x for i, x in enumerate(src_ids) if i in filtered_ids]
        if dst_txt_file is not None:
            dst_ids = [x for i, x in enumerate(dst_ids) if i in filtered_ids]
            assert len(src_ids) == len(dst_ids)
        if srct_txt_file is not None:
            assert len(src_ids) == len(srct_ids)
        print(f'filtered to {len(src_ids)} (from {orig_len})')

    print(datetime.datetime.now(), "Getting example lengths!")
    example_length = dict(map(lambda e: (e[0], len(e[1])), enumerate(src_ids)))

    if sort:
        print(datetime.datetime.now(), "Sorting example lengths!")
        sorted_lens = sorted(example_length.items(), key=lambda item: item[1])
        print(datetime.datetime.now(), "Getting sorted examples!")
        if srct_txt_file is not None:
            examples = list(map(lambda i: (src_ids[i[0]], dst_ids[i[0]], srct_ids[i[0]]), sorted_lens))
        else:
            examples = list(map(lambda i: (src_ids[i[0]], dst_ids[i[0]]), sorted_lens))
    else:
        examples = list(zip(src_ids, dst_ids, srct_ids)) if srct_txt_file is not None else \
                   list(zip(src_ids, dst_ids))

    print(datetime.datetime.now(), "Dumping examples!")
    with open(output_file, "wb") as fw:
        marshal.dump(examples, fw)

    print(datetime.datetime.now(), "Finished!")

    if quiet:
        f_null.close()
        sys.stdout = old_stdout

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src_data_path", help="Path to the source txt file for xlm tokenizer",
                      metavar="FILE", default=None)
    parser.add_argument("--srct", dest="srct_data_path",
                      help="Path to the source txt file for second tokenizer (shallow encoder) ", metavar="FILE",
                      default=None)
    parser.add_argument("--dst", dest="dst_data_path", help="Path to the target txt file", metavar="FILE", default=None)
    parser.add_argument("--output", dest="output_path", help="Output marshal file ", metavar="FILE", default=None)
    parser.add_argument("--tok", dest="tokenizer_path", help="Path to the tokenizer folder for srct", metavar="FILE", default=None)

    parser.add_argument("--max_seq_len", dest="max_seq_len", help="Max sequence length", type=int, default=175)
    parser.add_argument("--min_seq_len", dest="min_seq_len", help="Max sequence length", type=int, default=1)
    parser.add_argument("--shallow", action="store_true", dest="shallow_encoder",
                      help="Use shallow encoder instead of XLM", default=False)
    parser.add_argument("--lang-lines", dest="lang_lines_path", default='',
            help="path to file with language family IDs for each example in --dst")

    parser.add_argument("--pt-tok", dest='pretrained_tokenizer', choices=('', 'roberta-base', 'roberta-large', 'xlm-roberta-large'),
                      help="Use a pretrained tokenizer for dst", default='')
    parser.add_argument('--no-sort', dest='sort', action='store_false',
                      help='sort examples by length')
    parser.add_argument('--src-lang-fam', dest='src_lang_fam', action='store_true',
                      help='add lang fam tags to src as well')
    parser.add_argument('--no-eval-every-epoch', dest='eval_every_epoch', action='store_false',
                      help='do not evaluate at end of every epoch')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    srct_tokenizer = None

    if args.tokenizer_path:
        srct_tokenizer = TextProcessor(args.tokenizer_path)

    dst_tokenizer = None
    if args.pretrained_tokenizer:
        dst_tokenizer = TextProcessor(pretrained_name=args.pretrained_tokenizer)

    if dst_tokenizer is None and srct_tokenizer is None:
        print(f'ERROR: must pass in at least one of --pt-tok or --tok')
        sys.exit(-1)

    write(output_file=args.output_path, src_txt_file=args.src_data_path,
          dst_txt_file=args.dst_data_path, srct_txt_file=args.srct_data_path,
          tp_dst=dst_tokenizer, tp_srct=srct_tokenizer,
          shallow=args.shallow_encoder, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len,
          lang_lines_path=args.lang_lines_path, sort=args.sort, src_lang_fam=args.src_lang_fam)
