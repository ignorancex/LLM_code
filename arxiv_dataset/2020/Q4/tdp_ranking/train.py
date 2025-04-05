import argparse

from data_preparation import make_training_data, max_padded_candidate_length, merge_vocab


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--train_file", help="training data")
    arg_parser.add_argument("--silver_train_file", help="additional low quality training data",
                            default=None)
    arg_parser.add_argument("--dev_file", help="dev data",
                            default=None)
    arg_parser.add_argument("--model_file", help="where to store the model")

    arg_parser.add_argument("--TE_label_set",
                            help="which span label set to use: none, timex_event, time_ml, or full",
                            choices=["none", "timex_event", "full", "time_ml"],
                            default="full")

    arg_parser.add_argument("--classifier", help="which classifier to use",
                            choices=["bilstm", "glove_bilstm", "bert_bilstm", "bert_as_classifier",
                                     "bert_as_classifier_alt"])

    arg_parser.add_argument("--iter", help="number of iterations", type=int)

    arg_parser.add_argument("--labeled", help="train a model to predict labels", action="store_true",
                            default=False)

    arg_parser.add_argument("--edge_label_set", help="which edge label set to use: full or time_ml",
                            choices=["full", "time_ml"],
                            default="full")

    # arguments for the neural model
    arg_parser.add_argument("--size_embed", help="word embedding size (bilstm model)",
                            default=8, type=int)
    arg_parser.add_argument("--size_TE_label_embed", help="timex/event label embedding size (bilstm model)",
                            default=4, type=int)
    arg_parser.add_argument("--size_lstm", help="single lstm vector size (bilstm model)",
                            default=8, type=int)
    arg_parser.add_argument("--size_hidden", help="feed-forward neural network's hidden layer size (bilstm model)",
                            default=8, type=int)
    arg_parser.add_argument("--max_sequence_length", help="maximum words per sequence (1-2 sentences, for BERT)",
                            default=128, type=int)
    arg_parser.add_argument("--max_sentence_span", help="maximum sentences to use between nodes for "
                                                        "BERT-as-classifier-alt",
                            default=12, type=int)
    arg_parser.add_argument("--max_words_per_node", help="maximum words to consider per node",
                            default=5, type=int)
    arg_parser.add_argument("--no_handcrafted_features", help="disable hand-crafted features",
                            action="store_true", default=False)

    # arguments for training the TF neural model
    arg_parser.add_argument("--early_stopping_warmup", help="iterations before starting early stopping on dev loss",
                            default=30, type=int)
    arg_parser.add_argument("--early_stopping_threshold", help="iterations with increasing dev loss before stopping",
                            default=5, type=int)
    arg_parser.add_argument("--blending_init", help="iterations to train on silver data only",
                            default=25, type=int)
    arg_parser.add_argument("--blending_epochs", help="iterations to train on blended gold and silver data",
                            default=15, type=int)
    arg_parser.add_argument("--blend_factor", help="fraction of silver data to use "
                                                   "(blend a^k of silver data in kth blending iteration)",
                            default=0.75, type=float)
    arg_parser.add_argument("--lr", help="initial learning rate",
                            default=0.001, type=float)

    return arg_parser


def main(raw_args=None):  # Optionally take arguments to method instead of from script call
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args(raw_args)

    # Validate args
    edge_label_set = 'unlabeled' if not args.labeled else args.edge_label_set

    # Sanity check blending epochs and early stopping
    blending_init = args.blending_init if args.blending_init <= args.iter else 0
    blending_epochs = args.blending_epochs if blending_init + args.blending_epochs <= args.iter else 0
    early_stopping_warmup = min(args.early_stopping_warmup, args.iter)

    # Build data
    print('Building gold training data...')
    gold_training_data, vocab = make_training_data(args.train_file, pad_candidates=True)

    silver_training_data = None
    if args.silver_train_file:
        print('Building silver training data...')
        silver_training_data, silver_vocab = make_training_data(args.silver_train_file, pad_candidates=True)
        vocab = merge_vocab(vocab, silver_vocab)

    if args.dev_file:
        dev_data, _ = make_training_data(args.dev_file, pad_candidates=True)
    else:
        dev_data = None

    # Create and train classifier
    if args.classifier == 'bilstm':
        from bilstm_classifier import BilstmClassifier

        classifier = BilstmClassifier(vocab, args.size_embed, args.size_lstm,
                                      args.size_hidden, args.TE_label_set,
                                      args.size_TE_label_embed, edge_label_set,
                                      args.max_words_per_node, max_padded_candidate_length,
                                      args.no_handcrafted_features)

    elif args.classifier == 'glove_bilstm':
        from glove_bilstm_classifier import GloVeBilstmClassifier

        classifier = GloVeBilstmClassifier(args.size_lstm,
                                           args.size_hidden, args.TE_label_set,
                                           args.size_TE_label_embed, edge_label_set,
                                           args.max_words_per_node, max_padded_candidate_length,
                                           args.no_handcrafted_features)

    elif args.classifier == 'bert_bilstm':
        from bert_bilstm_classifier import BertBilstmClassifier

        classifier = BertBilstmClassifier(args.TE_label_set, args.size_TE_label_embed, args.size_lstm, args.size_hidden,
                                          edge_label_set, args.max_sequence_length, args.max_words_per_node,
                                          max_padded_candidate_length, args.no_handcrafted_features)

    elif args.classifier == 'bert_as_classifier':
        from bert_classifier import BertClassifier

        classifier = BertClassifier(args.TE_label_set, edge_label_set, args.max_sequence_length,
                                    max_padded_candidate_length, args.no_handcrafted_features)

    elif args.classifier == 'bert_as_classifier_alt':
        from bert_alt_classifier import BertAltClassifier

        classifier = BertAltClassifier(args.TE_label_set, edge_label_set, args.max_sequence_length,
                                       max_padded_candidate_length, args.max_sentence_span,
                                       args.no_handcrafted_features)

    else:
        raise ValueError('Invalid classifier argument')

    classifier.train(gold_training_data, silver_training_data, dev_data, args.model_file, args.labeled,
                     args.iter, early_stopping_warmup, args.early_stopping_threshold,
                     blending_init, blending_epochs, args.blend_factor, args.lr)


if __name__ == '__main__':
    main()
