from argparse import ArgumentParser


def get_mt_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", default=None)
    parser.add_argument("--cache_size", dest="cache_size", help="Number of blocks in cache", type=int, default=300)
    parser.add_argument("--model", dest="model_path", help="Directory path to save the best model",
                      default=None)
    parser.add_argument("--pretrained", dest="pretrained_path", help="Directory of pretrained model",
                      default=None)
    parser.add_argument("--epoch", dest="num_epochs", help="Number of training epochs", type=int, default=100)
    parser.add_argument("--clip", dest="clip", help="For gradient clipping", type=int, default=1)
    parser.add_argument("--batch", dest="batch", help="Batch size for train", type=int, default=12500)
    parser.add_argument("--mask", dest="mask_prob", help="Random masking probability", type=float, default=0.5)
    parser.add_argument("--lr", dest="learning_rate", help="Learning rate", type=float, default=0.0001)
    parser.add_argument("--warmup", dest="warmup", help="Number of warmup steps", type=int, default=12500)
    parser.add_argument("--step", dest="step", help="Number of training steps", type=int, default=125000)
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type=float, default=1.0)
    parser.add_argument("--cont", action="store_true", dest="continue_train",
                      help="Continue training from pretrained model", default=False)
    parser.add_argument("--dropout", dest="dropout", help="Dropout probability", type=float, default=0.1)
    parser.add_argument("--embed", dest="embed_dim", help="Embedding dimension", type=int, default=768)
    parser.add_argument("--intermediate", dest="intermediate_layer_dim", type=int, default=3072)
    parser.add_argument("--local_rank", dest="local_rank", type=int, default=-1)
    parser.add_argument("--capacity", dest="total_capacity", help="Batch capacity", type=int, default=600)
    parser.add_argument("--dict", dest="dict_path", help="External lexical dictionary", default=None)
    parser.add_argument("--beam", dest="beam_width", help="Beam width", type=int, default=5)
    parser.add_argument("--bt-beam", dest="bt_beam_width", help="Beam width for back-translation loss", type=int,
                      default=1)
    parser.add_argument("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type=float, default=1.3)
    parser.add_argument("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type=int, default=5)
    parser.add_argument("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type=float, default=0.8)
    parser.add_argument("--max_seq_len", dest="max_seq_len", help="Max sequence length", type=int, default=175)
    parser.add_argument("--ldec", action="store_true", dest="lang_decoder", help="Lang-specific decoder", default=False)
    parser.add_argument("--nll", action="store_true", dest="nll_loss", help="Use NLL loss instead of smoothed NLL loss",
                      default=False)
    parser.add_argument("--dev", dest="mt_dev_path",
                      help="Path to the MT dev data pickle files (SHOULD NOT BE USED IN UNSUPERVISED SETTING)",
                      default=None)
    parser.add_argument("--train", dest="mt_train_path",
                      help="Path to the MT train data pickle files (SHOULD NOT BE USED IN PURELY UNSUPERVISED SETTING)",
                      nargs='*', default=None)

    parser.add_argument("--dec", dest="decoder_layer", help="# decoder layers", type=int, default=6)

    parser.add_argument("--pt-dec", dest='pretrained_decoder', choices=('', 'roberta-base', 'roberta-large'),
                      help="specify pretrained decoder (if '', uses shallow encoder)", default='')
    parser.add_argument("--lang-info", dest="lang_info_path", default='./lang_info.json',
                      help="path to lang info file, only used with `--pt-dec`")

    parser.add_argument("--output", dest="output", help="Output file (for simiality)", default=None)
    parser.add_argument("--save-opt", action="store_true", dest="save_opt", default=False)
    parser.add_argument("--acc", dest="accum", help="Gradient accumulation", type=int, default=1)
    parser.add_argument("--freeze-enc", action="store_true", dest="freeze_encoder", default=False)
    parser.add_argument("--freeze-dec", action="store_true", dest="freeze_decoder", default=False)
    parser.add_argument("--shallow", action="store_true", dest="shallow_encoder",
                      help="Use shallow encoder instead of XLM", default=False)

    parser.add_argument("--multi", action="store_true", dest="multi_stream",
                      help="Using multi-stream model (the batches should be built via multi-stream)", default=False)
    parser.add_argument("--load-separate-train", action="store_true", dest="load_separate_train", default=False)
    parser.add_argument("--eval-steps", dest="eval_steps", help='number of steps before running evaluation', type=int, default=5000)
    parser.add_argument("--early-stop", dest="early_stop", help="Num of epochs before early stop", type=int, default=0)
    parser.add_argument('--src-lang-fam', dest='src_lang_fam', action='store_true',
                      help='add lang fam tags to src as well')
    parser.add_argument("--redo-output", action="store_false", dest="load_output",
                      help="When loading pretrained model, do not load output layer and decoder embedding"
                           "weights, initialize from scratch", default=True)
    parser.add_argument('--no-eval-every-epoch', dest='eval_every_epoch', action='store_false',
                      help='do not eval every epoch')
    parser.add_argument('--bidi', dest='bidi', action='store_true',
                      help='bidirectional model (used for back-translation only)')
    parser.add_argument('--xlm-name', dest='xlm_name', default='xlm-roberta-base', choices=('xlm-roberta-base', 'xlm-roberta-large'))
    parser.add_argument('--xlmr-dst-tok', action='store_true', help='use XLM-R tokenizer for dest')
    parser.add_argument('--round-trip-bleu', '-rtb', action='store_true', help='for UNMT, use round-trip BLEU')
    parser.add_argument('--foreign-tok-d', '-ftd', help='load foreign token dict')
    return parser
