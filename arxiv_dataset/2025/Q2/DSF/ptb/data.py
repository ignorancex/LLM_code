import os
import torch as th
import sys
import numpy as np


##################################################################
class Dictionary(object):
    """
    Simple interface for Vocabulary
    With special tokens.
    """
    def __init__(self, add_specials = True):
        self.word2idx = {}
        self.idx2word = []
        if add_specials:
            self.add_word("<pad>")
            self.add_word("<bos>")
            self.add_word("<eos>")
            self.add_word("<unk>")

    def __str__(self):
        return str(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


##################################################################
class Corpus(object):
    """
    Build a corpus from :
    - path
    - train, valid and test filename
    """
    def __init__(self, path, train, valid, test):
        self.dictionary = Dictionary()
        if train is not None:
            self.train, self.maxtrain = self.sentence_tokenizer(
                os.path.join(path, train)
            )
        else:
            self.train = None
        if valid is not None:
            self.valid, self.maxvalid = self.sentence_tokenizer(
                os.path.join(path, valid)
            )
        else:
            self.valid = None

        if test is not None:
            self.test, self.maxtest = self.sentence_tokenizer(
                os.path.join(path, test)
            )
        else:
            self.test = None

    def tokenize(self, filename):
        """Tokenizes a text file."""
        assert os.path.exists(filename)
        # Add words to the dictionary
        with open(filename, 'r', encoding="utf8") as f:
            tokens = 0
            wids = []
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = line.strip().split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                    wids.append(self.dictionary.word2idx[word])

        ids = th.LongTensor(wids)
        return ids

    def raw_tokenizer(self, filename):
        """Tokenizes a text file."""
        assert os.path.exists(filename)
        # Add words to the dictionary
        with open(filename, 'r', encoding="utf8") as f:
            tokens = 0
            wids = []
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = line.strip().split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                    wids.append(self.dictionary.word2idx[word])

        ids = th.LongTensor(wids)
        return ids

    def sentence_tokenizer(self, filename):
        """Tokenizes a text file per sentence
        The output is a list of LongTensor.
        Note: the sentences are wrapped with <bos> and <eos>.
        """
        assert os.path.exists(filename)
        maxl = 0
        with open(filename, 'r', encoding="utf8") as f:
            ntoks = 0
            nsents = 0
            tensorlist = []
            for line in f:
                words = ['<bos>'] + line.strip().split() + ['<eos>']
                lsent = len(words) 
                if len(words) == 1:
                    continue
                nsents+=1
                ntoks += lsent-2 # don't count <bos> and <eos>
                maxl = max(lsent,maxl)
                sent_tensor = th.zeros(lsent, dtype=th.long)
                cpt = 0
                for word in words:
                    self.dictionary.add_word(word)
                    sent_tensor[cpt] = self.dictionary.word2idx[word]
                    cpt+=1
                tensorlist.append(sent_tensor)
        sys.stdout.write("Load from "+filename+"\n")
        sys.stdout.write("#sent        = "+str(nsents)+"\n" )
        sys.stdout.write("#toks        = "+str(ntoks)+"\n" )
        sys.stdout.write("#max length  = "+str(maxl)+"\n" )
        return tensorlist, maxl


##################################################################
def raw_batchify(dataset, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = dataset.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    dataset = dataset.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    dataset = dataset.view(bsz, -1).t().contiguous()
    return dataset




##################################################################
def build_sentence_batch(dataset, maxl, sentence_indices,
                         pad_index=0, mask_index=0
                         ):
    """Build a batch of sentences.

    In inputs:
    - dataset : a list of torch.LongTensor i.e a list of sentences seen as word
      indices
    - sentence_indices: the sentences you want to pick to build the minibatch,
      a 1D array

    And optional :
    - pad_index and mask index (default 0)

    The function compute bsize = len(sentence_indices) and outputs:
    - the tensor for input  bsize,maxl
    - the tensor for output bsize,maxl
    - the np.array with the length

    Note that the sentences are sorted along their length (in
    decreasing order) to be further packed and unpacked.
    """
    bsize = len(sentence_indices)
    assert(bsize > 0)
    # look at the lengths
    lengths = np.zeros(bsize, dtype=np.int32)
    brange = range(bsize)
    for i,j in zip(brange, sentence_indices):
        lsent = len(dataset[j])
        seqlen = min(lsent, maxl)
        lengths[i] = seqlen
    # if the longest is shorter than maxl,
    # don't waste place and narrow the output tensors
    longest = max(lengths)
    if longest < maxl:
        maxl=longest
    XmB = th.ones(bsize,maxl, dtype=th.long)*pad_index 
    YmB = th.ones(bsize,maxl, dtype=th.long)*mask_index
    for i,j in zip(brange,sentence_indices): 
        lsent = len(dataset[j])
        seqlen = lengths[i]
        XmB[i,0:seqlen] = dataset[j][:seqlen]
        YmB[i,0:seqlen-1]= dataset[j][1:seqlen]
        if lsent > maxl : 
            YmB[i,-1] =  dataset[j][maxl]
    # sort the minibatch along sentence length 
    outidx = np.argsort(-lengths)
    return XmB[outidx,:] , YmB[outidx,:] , th.IntTensor(lengths[outidx])


####################################################################################
# Basic test
#  wc ./ptb/ptb.test.txt 
#  3761  78669 449945 ./ptb/ptb.test.txt
# The total number of tokens in the test should be 82430
####################################################################################
if __name__ == "__main__":
    corpus = Corpus("./ptb/", "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
    print("a sample : ",corpus.test[0])
    assert(len(corpus.test) == 3761)
    xmb, ymb ,lmb = build_sentence_batch(corpus.test,20, np.arange(3))
    print("A minibatch: ------ ")
    print(xmb)
    print(ymb)
    print(lmb)


    
