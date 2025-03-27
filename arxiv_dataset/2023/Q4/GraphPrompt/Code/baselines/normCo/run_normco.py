import nltk
# #
# nltk.download('stopwords', ".")
# nltk.download('punkt',".")
# nltk.data.path.append('.')
import numpy as np
import random
import os
import argparse
from data_process import load_data,data_split
import pickle
import logging
from normco_trainer import NormCoTrainer
# from edit_distance import EditDistance_Classifier
from normco.data.data_generator import DataGenerator
from normco.data.datasets import PreprocessedDataset
from torch.utils.data import DataLoader
from data_process import *

#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
# set up logger
def setup_logger(root,name,log_file_nm,level=logging.INFO):
    log_dir = os.path.join(root,name)
    """To setup as many loggers as you want"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir,"{}.log".format(log_file_nm))
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(log_dir)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data processing arguments
    parser = argparse.ArgumentParser(description="Generate data and datasets for training and evaluation")
    parser.add_argument('--data_dir',type=str,help='direct to dataset file')
    parser.add_argument('--require_download',type=bool,default=False,help='whether to download data from directory file')
    parser.add_argument('--directory_file',type=str,default=None,help='the location of the directory file based upon which we retrieve data')
    parser.add_argument('--use_unk_concept', action='store_true',
                        help='Whether or not to use a special concept for "UNKNOWN"', default=False)
    parser.add_argument('--init_embedding', type=bool, default=False, help='if need initial embeddings')
    #,max_depth,max_nodes,search_method
    parser.add_argument('--max_depth',type=int,default=2,help='number of maximum depth')
    parser.add_argument('--max_nodes', type=int, default=3, help='number of maximum nodes')
    parser.add_argument('--search_method', type=str, default='bfs', help='algorithm used to traverse the graph to generate context information for each node')
    # training arguments
    parser.add_argument('--model', type=str, help='The RNN type for coherence',
                            default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('--num_epochs_mention_only', type=int, help='The number of epochs to run', default=1)
    parser.add_argument('--num_epochs_with_coherence', type=int, help='The number of epochs to run', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size for mini batching', default=32)
    parser.add_argument('--sequence_len', type=int, help='The sequence length for phrases', default=20)
    parser.add_argument('--embedding_dim', type=int, help='embedding dimension', default=128)
    parser.add_argument('--num_neg', type=int, help='The number of negative examples', default=5)
    parser.add_argument('--output_dim', type=int, help='The output dimensionality', default=128)
    parser.add_argument('--lr', type=float, help='The starting learning rate', default=0.001)
    parser.add_argument('--l2reg', type=float, help='L2 weight decay', default=0.0)
    parser.add_argument('--dropout_prob', type=float, help='Dropout probability', default=0.0)
    parser.add_argument('--scoring_type', type=str, help='The type of scoring function to use', default="euclidean",
                        choices=['euclidean', 'bilinear', 'cosine'])
    parser.add_argument('--weight_init', type=str, help='Weights file to initialize the model', default=None)
    parser.add_argument('--threads', type=int, help='Number of parallel threads to run', default=0)
    parser.add_argument('--save_every', type=int, help='Number of epochs between each model save', default=1)
    parser.add_argument('--save_file_name', type=str, help='Name of file to save model to', default='model.pth')
    parser.add_argument('--optimizer', type=str, help='Which optimizer to use', default='adam',
                            choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam'])
    # Using SGD gives errors https://github.com/pytorch/pytorch/issues/30402
    parser.add_argument('--loss', type=str, help='Which loss function to use', default='maxmargin',
                        choices=['maxmargin', 'xent'])
    parser.add_argument('--eval_every', type=int, help='Number of epochs between each evaluation', default=2)
    parser.add_argument('--train_loss_log_interval', type=int, help='Number of epochs between each log of train losses', default=50)
    parser.add_argument('--use_features', action='store_true', help='Whether or not to use hand crafted features',
                            default=False)
    parser.add_argument('--mention_only', action='store_true', help='Whether or not to use mentions only',
                            default=False)
    parser.add_argument('--logfile', type=str, help='File to log evaluation in', default=None)
    parser.add_argument('--start', type=int, help='from which to start', default=-1)
    parser.add_argument('--end', type=int, help='till which to stop', default=-1)
    parser.add_argument('--is_unseen', action='store_true', help='whether to use unseen entities during evaluation', default=False)
    parser.add_argument('--dataset_nm',type=str,default="all",help='specify a dataset to run the experiment')
    parser.add_argument('--save_only',default=False,action='store_true')
    parser.add_argument('--legacy_features_only', default=False, action='store_true')
    parser.add_argument('--folds',type=int, default=5)
    parser.add_argument('--test_bsz', type=int, default=1)
    parser.add_argument('--log_root', type=str, default='../log')
    args = parser.parse_args()
    setup_seed(0)
    if args.require_download:
        assert directory_file is not None
        get_all_data(args.directory_file,args.data_dir)
        print("data downloading done")

    if args.dataset_nm=="all":

        files = list(json.load(open('./rel2desc.json')).keys())
        if args.start!=-1:
            if args.end!=-1:
                files = files[args.start:args.end]
            else:
                files = files[args.start:]
        if True:
            for filename in files:
                # for d,_,all_files in os.walk(args.data_dir):
                #     files = all_files[args.start:args.end] if args.end!=-1 else all_files[args.start:]
                nm = "file_nm_{}_train_m_{}_epochs_c_{}_epochs_is_unseen_{}_valid_and_test_0903_1".format(filename,
                                                                                                     args.num_epochs_mention_only,
                                                                                                     args.num_epochs_with_coherence,
                                                                                                     args.is_unseen)
                type = ("unseen" if args.is_unseen else "seen")
                logger = setup_logger(args.log_root,type, nm)
                logger.info("file name is {}".format(filename))
                for arg_nm in vars(args):
                    logger.info("LOGGING ARGS | {} : {}".format(arg_nm, getattr(args, arg_nm)))
                vtop1,vtop10,ttop1,ttop10 = [],[],[],[]
                for fold in range(args.folds):
                    logger.info("current experiment: fold {}".format(fold))
                    if filename[-4:]==".obo":
                        #if filename=='envo.obo':continue# envo.obo has a specific problem and I am not sure why.
                        # concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph(os.path.join(dir,filename))
                        #np.array(name_array), np.array(query_id_array), mention2id, edge_index,edges
                        concept_array,query2id_array,mention2id,_,edges,triples = load_data(os.path.join(args.data_dir,filename),True)
                        concept_sz = len(concept_array)
                        logger.info("concept size: {}".format(concept_sz))
                        logger.info('number of synonym pairs: {}'.format(len(mention2id)))
                        if len(query2id_array)>10:
                            # each pair contains queries(mentions) and their ids
                            queries_train,queries_valid,queries_test =  data_split(query2id_array,is_unseen=args.is_unseen,test_size=0.33,seed=fold)
                            logger.info("train_size：{}".format(len(queries_train)))
                            logger.info("valid_size：{}".format(len(queries_valid)))
                            logger.info("test_size：{}".format(len(queries_test)))
                            data_generator = DataGenerator(args)
                            #def prepare_data(self,paired_data,tree,concept2id,max_depth,max_nodes,search_method):
                            mentions = {
                                    'train':queries_train,
                                    'valid':queries_valid,
                                    'test':queries_test

                                }
                            concept_ids = {
                                    'train': [mention2id[i] for i in queries_train],
                                    'valid': [mention2id[i] for i in queries_valid],
                                    'test':[mention2id[i] for i in queries_test]

                                }
                            num_neg = args.num_neg
                            if args.legacy_features_only:
                                logger.info("Number of Edges in the relationship graph: {} ".format(len(edges)))
                                data_dicts, vocab = data_generator.prepare_data(concept_ids, mentions, query2id_array, edges, mention2id)
                            else:
                                logger.info("Number of Edges in the relationship graph: {} ".format(len(triples)))
                                data_dicts, vocab = data_generator.prepare_data(concept_ids, mentions, query2id_array, triples, mention2id)
                            if args.save_only:  # only saved directory to pickle files to save memory during process
                                with open(data_dicts['train'], 'rb') as tf:
                                    train_dat = pickle.load(tf)
                                mention_data_train = PreprocessedDataset(mention2id, train_dat['mentions'], num_neg,
                                                                         vocab, False,train_subset=concept_ids['train'])
                                coherence_data_train = PreprocessedDataset(mention2id, train_dat['hierarchy'],
                                                                           num_neg, vocab,
                                                                           False,train_subset=concept_ids['train'])
                                with open(data_dicts['valid'], 'rb') as tf:
                                    valid_dat = pickle.load(tf)
                                mention_data_valid = PreprocessedDataset(mention2id, valid_dat['mentions'], concept_sz - 1,
                                                                         vocab, False)
                                coherence_data_valid = PreprocessedDataset(mention2id, valid_dat['hierarchy'],
                                                                           concept_sz - 1, vocab,
                                                                           False)
                                with open(data_dicts['test'], 'rb') as tf:
                                    test_dat = pickle.load(tf)
                                mention_data_test = PreprocessedDataset(mention2id, test_dat['mentions'], concept_sz - 1,
                                                                        vocab, False)
                                coherence_data_test = PreprocessedDataset(mention2id, test_dat['hierarchy'],
                                                                          concept_sz - 1, vocab,
                                                                          False)
                            else:
                                # import dataset from def __init__(self, concept_dict,data_dict, num_neg, vocab_dict=None, use_features=False):

                                mention_data_train = PreprocessedDataset(mention2id, data_dicts['train']['mentions'],
                                                                         num_neg,
                                                                         vocab, False)
                                coherence_data_train = PreprocessedDataset(mention2id, data_dicts['train']['hierarchy'],
                                                                           num_neg, vocab,
                                                                           False)
                                mention_data_valid = PreprocessedDataset(mention2id, data_dicts['valid']['mentions'],
                                                                         concept_sz - 1,
                                                                         vocab, False)
                                coherence_data_valid = PreprocessedDataset(mention2id, data_dicts['valid']['hierarchy'],
                                                                           concept_sz - 1, vocab,
                                                                           False)
                                # Negative Examples cardinality equals total number of concepts
                                mention_data_test = PreprocessedDataset(mention2id, data_dicts['test']['mentions'],
                                                                        concept_sz - 1,
                                                                        vocab, False)
                                coherence_data_test = PreprocessedDataset(mention2id, data_dicts['test']['hierarchy'],
                                                                          concept_sz - 1,
                                                                          vocab,
                                                                          False)

                            trainer = NormCoTrainer(args,logger)
                            n_concepts = len(mention2id.keys())
                            n_vocab = len(vocab.keys())
                            logger.info("FOLD NUM {} | vocabulary size  {}".format(fold,n_vocab))
                            val_res = trainer.train(mention_data_train,coherence_data_train,mention_data_valid,coherence_data_valid,n_concepts,n_vocab,logger)
                            test_res = trainer.evaluate(mention_data_test,coherence_data_test,logger)
                            for i in val_res:
                                logger.info("FOLD {} | Valid result on {} is {:6f},{:6f}".format(fold,i,val_res[i][0],val_res[i][1]))
                            for i in test_res:
                                logger.info("FOLD {} |Test result on {} is {:6f},{:6f}".format(fold,i,test_res[i][0],test_res[i][1]))
                            vtop1.append(val_res['all'][0])
                            vtop10.append(val_res['all'][1])
                            ttop1.append(test_res['all'][0])
                            ttop10.append(test_res['all'][1])
                        else:
                            logger.warning("Unable to run the experiment for file, too few items for  {} with file number {}".format(filename,len(mention2id)))
                    vmean1,vstd1 = np.mean(vtop1),np.std(vtop1)
                    logger.info("FILE {} | valid top 1 mean {:6f} std {:6f}".format(filename,vmean1,vstd1))
                    vmean10,vstd10 = np.mean(vtop10), np.std(vtop10)
                    logger.info("FILE {} | valid top 10 mean {:6f} std {:6f}".format(filename,vmean1,vstd1))
                    tmean1,tstd1 = np.mean(ttop1), np.std(ttop1)
                    logger.info("FILE {} | test top 1 mean {:6f} std {:6f}".format(filename, tmean1, tstd1))
                    tmean10,tstd10 = np.mean(ttop10), np.std(ttop10)
                    logger.info("FILE {} | test top 10 mean {:6f} std {:6f}".format(filename, tmean10, tstd10))

                else: logger.info("Invalid file encountered, file name is {}".format(filename))
    else:
        # setattr(args, "is_unseen", False)
        type = ("unseen" if args.is_unseen else "seen")
        nm = "file_nm_{}_train_m_{}_epochs_c_{}_epochs_is_unseen_{}_valid_and_test_2".format(args.dataset_nm,
                                                                               args.num_epochs_mention_only,
                                                                               args.num_epochs_with_coherence,
                                                                               args.is_unseen)
        
        logger = setup_logger(type, nm)
        for arg_nm in vars(args):
            logger.info("LOGGING ARGS | {} : {}".format(arg_nm, getattr(args, arg_nm)))
        filename = args.dataset_nm + ".obo"
        logger.info("file name is {}".format(filename))
        if filename[-4:] == ".obo":
            # if filename=='envo.obo':continue# envo.obo has a specific problem and I am not sure why.
            # concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph(os.path.join(dir,filename))
            # np.array(name_array), np.array(query_id_array), mention2id, edge_index,edges
            concept_array, query2id_array, mention2id, _, edges,triples = load_data(
                os.path.join(args.data_dir, filename), True)
            concept_sz = concept_array.shape[0]
            logger.info("concept size: {}".format(concept_sz))
            logger.info('number of synonym pairs: {}'.format(len(mention2id)))
            if len(query2id_array) > 10:
                # each pair contains queries(mentions) and their ids
                queries_train, queries_valid, queries_test = data_split(query2id_array,
                                                                        is_unseen=args.is_unseen,
                                                                        test_size=0.33)
                logger.info("train_size：{}".format(len(queries_train)))
                logger.info("valid_size：{}".format(len(queries_valid)))
                logger.info("test_size：{}".format(len(queries_test)))
                data_generator = DataGenerator(args)
                # def prepare_data(self,paired_data,tree,concept2id,max_depth,max_nodes,search_method):
                mentions = {
                    'train': queries_train,
                    'valid': queries_valid,
                    'test': queries_test

                }
                concept_ids = {
                    'train': [mention2id[i] for i in queries_train],
                    'valid': [mention2id[i] for i in queries_valid],
                    'test': [mention2id[i] for i in queries_test]

                }
                num_neg = args.num_neg
                if args.legacy_features_only:

                    data_dicts, vocab = data_generator.prepare_data(concept_ids, mentions, query2id_array, edges,
                                                                    mention2id)
                else:
                    data_dicts, vocab = data_generator.prepare_data(concept_ids, mentions, query2id_array, triples,
                                                                    mention2id)
                if args.save_only:#only saved directory to pickle files to save memory during process
                    with open(data_dicts['train'],'rb') as tf:
                        dat = pickle.load(tf)
                    mention_data_train = PreprocessedDataset(mention2id, dat['mentions'], num_neg,
                                                             vocab, False)
                    coherence_data_train = PreprocessedDataset(mention2id, dat['hierarchy'],
                                                               num_neg, vocab,
                                                               False)
                    with open(data_dicts['valid'],'rb') as tf:
                        dat = pickle.load(tf)
                    mention_data_valid = PreprocessedDataset(mention2id, dat['mentions'], concept_sz - 1,
                                                                 vocab, False)
                    coherence_data_valid = PreprocessedDataset(mention2id, dat['hierarchy'],
                                                                   concept_sz - 1,vocab,
                                                                   False)
                    with open(data_dicts['test'],'rb') as tf:
                        dat = pickle.load(tf)
                    mention_data_test = PreprocessedDataset(mention2id, dat['mentions'], concept_sz - 1,
                                                                 vocab, False)
                    coherence_data_test = PreprocessedDataset(mention2id, dat['hierarchy'],
                                                                   concept_sz - 1, vocab,
                                                                   False)
                else:
                    # import dataset from def __init__(self, concept_dict,data_dict, num_neg, vocab_dict=None, use_features=False):

                        mention_data_train = PreprocessedDataset(mention2id, data_dicts['train']['mentions'], num_neg,
                                                                 vocab, False)
                        coherence_data_train = PreprocessedDataset(mention2id, data_dicts['train']['hierarchy'],
                                                                   num_neg, vocab,
                                                                   False)
                        mention_data_valid = PreprocessedDataset(mention2id, data_dicts['valid']['mentions'], concept_sz - 1,
                                                                 vocab, False)
                        coherence_data_valid = PreprocessedDataset(mention2id, data_dicts['valid']['hierarchy'],
                                                                   concept_sz - 1, vocab,
                                                                   False)
                        # Negative Examples cardinality equals total number of concepts
                        mention_data_test = PreprocessedDataset(mention2id, data_dicts['test']['mentions'], concept_sz - 1,
                                                                vocab, False)
                        coherence_data_test = PreprocessedDataset(mention2id, data_dicts['test']['hierarchy'], concept_sz - 1,
                                                                  vocab,
                                                                  False)

                trainer = NormCoTrainer(args, logger)
                n_concepts = len(mention2id.keys())
                n_vocab = len(vocab.keys())
                logger.info("vocabulary size {}".format(n_vocab))
                trainer.train(mention_data_train, coherence_data_train, mention_data_valid,
                              coherence_data_valid, n_concepts, n_vocab, logger)
                res = trainer.evaluate(mention_data_test, coherence_data_test, logger)
                for i in res:
                    logger.info("Test result on {} is {:6f},{:6f}".format(i, res[i][0], res[i][1]))
            else:
                logger.warning(
                    "Unable to run the experiment for file, too few items for  {} with file number {}".format(
                        filename, len(mention2id)))
        else:
            logger.info("Invalid file encountered, file name is {}".format(filename))