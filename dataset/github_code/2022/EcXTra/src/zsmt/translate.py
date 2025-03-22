import datetime
import tempfile
from argparse import ArgumentParser
from os import devnull
from pathlib import Path

import torch
import torch.utils.data as data_utils
from torch.cuda.amp import autocast

from zsmt import dataset
from zsmt.seq2seq import Seq2Seq
from zsmt.seq_gen import BeamDecoder, get_outputs_until_eos
from zsmt.create_mt_batches import write as write_batches
# from zsmt.train_mt import TO_FOREIGN_TOK

def get_lm_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input_path", type=Path, default=None)
    parser.add_argument("--input2", dest="second_input_path", type=Path, default=None)

    parser.add_argument("--batch-file", dest="batch_file", type=Path, default=None, help='path to binary file with input')

    parser.add_argument("--src", dest="src_lang", type=str, default=None)
    parser.add_argument("--target", dest="target_lang", type=str, default=None)
    parser.add_argument("--output", dest="output_path", type=Path, default=None)
    parser.add_argument("--batch", dest="batch", help="Batch size", type=int, default=4000)
    parser.add_argument("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", type=Path, default=None)
    parser.add_argument("--cache_size", dest="cache_size", help="Number of blocks in cache", type=int, default=300)
    parser.add_argument("--model", dest="model_path", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true", dest="verbose", help="Include input!", default=False)
    parser.add_argument("--beam", dest="beam_width", type=int, default=4)
    parser.add_argument("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type=float, default=1.3)
    parser.add_argument("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type=int, default=5)
    parser.add_argument("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type=float, default=0.8)
    parser.add_argument("--max_seq_len", dest="max_seq_len", help="Max sequence length", type=int, default=175)
    parser.add_argument("--capacity", dest="total_capacity", help="Batch capacity", type=int, default=600)
    parser.add_argument("--shallow", action="store_true", dest="shallow", default=False)
    parser.add_argument("--lang", dest="lang_lines_path", default='',
                      help="path to file with language family IDs for each input example, or one tag")
    parser.add_argument("--pt-dec", dest='pretrained_decoder', choices=('', 'roberta-base', 'roberta-large'),
                      help="Use a pretrained decoder for dst", default='')
    parser.add_argument('--src-lang-fam', dest='src_lang_fam', action='store_true',
                      help='add lang fam tags to src as well')
    parser.add_argument('--foreign-tok', dest='foreign_tok', default=None,
                      help='token to indicate translation to foreign lang')
    parser.add_argument('--xlm-name', dest='xlm_name', default='xlm-roberta-base', choices=('xlm-roberta-base', 'xlm-roberta-large'))
    parser.add_argument('--max-sents', default=-1, type=int)
    parser.add_argument('--xlmr-dst-tok', action='store_true', help='use XLM-R tokenizer for dest')
    parser.add_argument('--cleanup', dest='cleanup', action='store_true',
                      help='cleanup tokenization spaces (set True only if using untokenized files)')

    return parser


def translate_batch(batch, generator, text_processor, verbose=False, foreign_tok=None, cleanup=True):
    src_inputs = batch["src_texts"].squeeze(0)
    src_mask = batch["src_pad_mask"].squeeze(0)
    tgt_inputs = batch["dst_texts"].squeeze(0)
    src_pad_idx = batch["src_pad_idx"].squeeze(0)
    src_text = None
    srct_text = None
    if generator.seq2seq_model.multi_stream:
        srct_inputs = batch["srct_texts"].squeeze(0)
        srct_mask = batch["srct_pad_mask"].squeeze(0)

        tp_src = generator.seq2seq_model.srct_text_processor
    else:
        srct_inputs = None
        srct_mask = None
    if verbose:
        src_ids = get_outputs_until_eos(generator.seq2seq_model.src_eos_id(), src_inputs, remove_first_token=True)
        src_text = list(map(lambda src: generator.seq2seq_model.decode_src(src, cleanup=cleanup), src_ids))
        if generator.seq2seq_model.multi_stream:
            srct_ids = get_outputs_until_eos(tp_src.token_id(tp_src.sep_token), srct_inputs, remove_first_token=False)
            srct_text = list(map(lambda src: tp_src.tokenizer.decode(src.numpy()), srct_ids))
        src_text = [x.lstrip(foreign_tok) for x in src_text]
    with autocast():
        outputs = generator(src_inputs=src_inputs, src_sizes=src_pad_idx,
                            first_tokens=tgt_inputs[:, 0], srct_inputs=srct_inputs,
                            src_mask=src_mask, srct_mask=srct_mask,
                            pad_idx=text_processor.pad_token_id())
    mt_output = list(map(lambda x: text_processor.tokenizer.decode(x[1:].numpy(), clean_up_tokenization_spaces=cleanup), outputs))

    return mt_output, src_text, srct_text

def build_model(args):
    model = Seq2Seq.load(Seq2Seq, args.model_path, tok_dir=args.tokenizer_path,
                         pretrained_decoder=args.pretrained_decoder, src_lang_fam=args.src_lang_fam,
                         xlm_name=args.xlm_name, xlmr_dst_tok=args.xlmr_dst_tok)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    generator = BeamDecoder(model, beam_width=args.beam_width, max_len_a=args.max_len_a,
                            max_len_b=args.max_len_b, len_penalty_ratio=args.len_penalty_ratio)
    generator.eval()
    return model, generator


def get_mt_data(mt_model, args, pin_memory, srct_pad_token_id):
    mt_data = dataset.MTDataset(batch_pickle_dir=args.batch_file,
                                max_batch_capacity=args.total_capacity, keep_src_pad_idx=True,
                                max_batch=args.batch,
                                src_pad_idx=mt_model.src_pad_id(),
                                srct_pad_idx=srct_pad_token_id,
                                dst_pad_idx=mt_model.text_processor.pad_token_id())
    dl = data_utils.DataLoader(mt_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
    return dl

def translate(args):
    print('building model...')
    model, generator = build_model(args)
    dst_text_processor, srct_text_processor = model.text_processor, model.srct_text_processor

    print('building dataloader...')
    pin_memory = torch.cuda.is_available()
    srct_pad_token_id = srct_text_processor.pad_token_id() if srct_text_processor else None
    f_temp = None

    foreign_tok = ''
    if args.foreign_tok: # add foreign tok to input
        foreign_tok = args.foreign_tok.encode().decode('unicode-escape')
        if isinstance(args.input_path, list):
            args.input_path = [f'{foreign_tok} {line}' for line in args.input_path]
        else:
            f_temp = tempfile.NamedTemporaryFile('w+')
            with open(args.input_path, 'r') as f_in:
                for line in f_in:
                    f_temp.write(f'{foreign_tok} {line}')
            f_temp.flush()
            args.input_path = f_temp.name

    if args.batch_file:
        test_loader = get_mt_data(model, args, pin_memory, srct_pad_token_id)
    else:
        with tempfile.NamedTemporaryFile('w+b') as f:
            args.batch_file = f.name
            write_batches(output_file=f.name, src_txt_file=args.input_path,
                        srct_txt_file=args.second_input_path,
                        tp_dst=dst_text_processor, tp_srct=srct_text_processor,
                        shallow=args.shallow, min_seq_len=-1, max_seq_len=-1,
                        lang_lines_path=args.lang_lines_path, sort=False,
                        src_lang_fam=args.src_lang_fam, max_sents=args.max_sents)

            test_loader = get_mt_data(model, args, pin_memory, srct_pad_token_id)
        args.batch_file = None

    sen_count = 0
    predictions = []
    args.output_path = args.output_path or devnull
    with open(args.output_path, "w") as writer:
        with torch.no_grad():
            for batch in test_loader:
                try:
                    mt_output, src_text, srct_text = translate_batch(batch, generator, dst_text_processor,
                        args.verbose, foreign_tok, args.cleanup)
                    sen_count += len(mt_output)
                    print(datetime.datetime.now(), "Translated", sen_count, "sentences", end="\r")
                    predictions.extend(mt_output)
                    if not args.verbose:
                        writer.write("\n".join(mt_output))
                    elif args.verbose and not srct_text:
                        writer.write("\n".join([y + " ||| " + x for x, y in zip(mt_output, src_text)]))
                    elif args.verbose and srct_text[0]:
                        writer.write("\n".join([y + " ||| " + z + " ||| " + x for x, y, z in zip(mt_output, src_text, srct_text)]))
                    else:
                        print('ERROR: should not reach here!')
                    writer.write("\n")
                except RuntimeError as err:
                    print("\n", repr(err))

        print(datetime.datetime.now(), "Translated", sen_count, "sentences")
        print(datetime.datetime.now(), "Done!")

    if f_temp:
        f_temp.close()
    return predictions

if __name__ == "__main__":
    parser = get_lm_args_parser()
    args = parser.parse_args()
    translate(args)
