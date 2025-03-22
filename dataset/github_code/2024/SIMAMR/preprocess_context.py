# coding:utf-8
import os
import torch
import numpy as np
import argparse
import config_utils
# from dataset import AMRDialogSetNew
from dataset2Context import AMRDialogSetNew
from dataset_utils import load_vocab_new, save_vocab_new, generate_vocab_new, tokenize, bpe_tokenize, load_pretrained_emb
from torch.utils import data
# from bpemb import BPEmb
import time

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, help="Configuration file.")
    FLAGS, unparsed = argparser.parse_known_args()

    if FLAGS.config_path is not None:
        print("Loading hyperparameters from " + FLAGS.config_path)
        FLAGS = config_utils.load_config(FLAGS.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if not os.path.exists(FLAGS.save_data):
        os.makedirs(FLAGS.save_data)

    print('Building vocabularies ... ')
    # words, word2id, concepts, concept2id, relations, relation2id, word_rels, word_rel2id = generate_vocab_new(
    #     FLAGS.train_path, relation_vocab_size=5000, shared_word_concept=FLAGS.share_con_vocab
    # )
    words, word2id, concepts, concept2id, relations, relation2id = generate_vocab_new(
        FLAGS.train_path, relation_vocab_size=5000, shared_word_concept=FLAGS.share_con_vocab
    )


    print('Preprocess.py word2id:', len(word2id))
    print('Preprocess.py concept2id:', len(concept2id))
    print('Preprocess.py relation2id:', len(relation2id))
    print('Saving vocabularies ... ')
    # save_vocab_new(FLAGS.save_data, words, concepts, relations, word_rels)
    save_vocab_new(FLAGS.save_data, words, concepts, relations)


    tokenize_fn = tokenize
    # word_vocab, concept_vocab, relation_vocab, word_rel_vocab = word2id, concept2id, relation2id, word_rel2id
    word_vocab, concept_vocab, relation_vocab = word2id, concept2id, relation2id

    print('word_vocab size: ', len(word_vocab))
    print('concept_vocab: ', len(concept_vocab))
    print('relation_vocab size: ', len(relation_vocab))

    print("Loading train data ...")
    s_time = time.time()
     
    train_set = AMRDialogSetNew(FLAGS.train_path, tokenize_fn, word_vocab, concept_vocab, relation_vocab, use_bert=False)
    dev_set = AMRDialogSetNew(FLAGS.dev_path, tokenize_fn, word_vocab, concept_vocab, relation_vocab, use_bert=False)
    test_set = AMRDialogSetNew(FLAGS.test_path, tokenize_fn, word_vocab, concept_vocab, relation_vocab, use_bert=False)

     
    
    print("Num train examples = {}".format(len(train_set.instance)))
    print("Train examples:", train_set.instance[0])
    print("Num valid examples = {}".format(len(dev_set.instance)))
    print("Num test examples = {}".format(len(test_set.instance)))
    print("Test examples:", test_set.instance[0])
    print("Saving dataset ...")
    
    train_pt_file = "{:s}/train_data.pt".format(FLAGS.save_data)
    torch.save(train_set, train_pt_file)
    dev_pt_file = "{:s}/dev_data.pt".format(FLAGS.save_data)
    torch.save(dev_set, dev_pt_file)
    test_pt_file = "{:s}/test_data.pt".format(FLAGS.save_data)
    torch.save(test_set, test_pt_file)
    

    print("Preprocessing ended.")

