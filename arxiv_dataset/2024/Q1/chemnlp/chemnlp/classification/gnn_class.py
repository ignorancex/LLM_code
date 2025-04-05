# https://github.com/zshicode/GNN-for-text-classification
import dgl
from dgl.dataloading.pytorch import GraphDataLoader
from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv, GATConv, DotGatConv

# from dgl.nn.pytorch import GraphConv, GATConv, GatedGraphConv, DotGatConv
from dgl.nn import AvgPooling

# , MaxPooling
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import operator

import pandas as pd
import pickle
import seaborn as sns
import string
import sklearn

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import TruncatedSVD
# from sklearn.manifold import TSNE
# import seaborn as sns
# from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp

from tqdm import tqdm
import torch

# from torch.utils.data import DataLoader

# from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn


# from wordcloud import WordCloud
# import wandb
import warnings


warnings.filterwarnings("ignore")

sns.set_theme()
sns.set_context("talk")

stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

seed = 42

# filename = "/wrk/knc6/AtomNLP/chemnlp/chemnlp/sample_data/cond_mat_small.csv"
filename = "/wrk/knc6/AtomNLP/Summarize/cond_mat.csv"
key = "categories"
value = "title_abstract"
GLOVE_EMBEDDING_PATH = "/wrk/knc6/AtomNLP/GNN_GAT/glove.6B.50d.txt"
ratio = 0.8


class args:
    max_epochs = 50
    lr = 1e-3
    batch_size = 64
    embedding_dim = 50
    hidden_dim = 256
    num_heads = 8  # used for attention model
    n_folds = 1
    window_size = 3


df_csv = pd.read_csv(filename, dtype="str")


def filter_text(text):
    """
    Returns the lowercase of input text by removing punctuations, stopwords

        Parameters:
            text (str): input string

        Returns:
            string
    """

    tokenized_words = word_tokenize(text)
    filtered_words = [
        word.strip().strip(".").lower()
        for word in tokenized_words
        if (
            (word.lower() not in string.punctuation)
            & (word.lower() not in stopwords)
        )
    ]

    return " ".join(filtered_words)


def load_embeddings(path):
    with open(path, "rb") as f:
        emb_arr = pickle.load(f)
    return emb_arr


def check_coverage(vocab, embeddings_index):
    """
    Returns list of tuples. The first element of each tuple specifies
    the word present in the description but not in the embeddings
    and the second element
    specifies the count of that word in the descriptions.
    The tuples are sorted in the descending order of their count.

        Parameters:
            vocab (dict): Dictionary with keys as words and
            values as their count of occurence
            embeddings_index (dict): Dictionary with keys as
            words and values as their embeddings

        Returns:
            List of tuples
    """
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except Exception:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print("Found embeddings for {:.2%} of vocab".format(len(a) / len(vocab)))
    print("Found embeddings for  {:.2%} of all text".format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def build_vocab(sentences, verbose=True):
    """
    Returns dictionary with keys as words in the
     sentences and values as their count of occurence

        Parameters:
            sentences (list of list): List of lists of descriptions
            verbose (bool): whether to show the progress bar

        Returns:
            dictionary
    """
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def build_graph(
    start, end, truncate=False, weighted_graph=True, MAX_TRUNC_LEN=100
):
    """
    Returns list of adjacency matrix and list of node matrix

        Parameters:
            start (int): start index of list
            end (int): end index of list
            truncate (bool): whether to truncate the text
            weighted_graph (bool): whether to use word pair
            count as the weight in adjacency matrix or just 1.0

        Returns:
            list of adjacency matrices, list of node matrices


    """
    x_adj = []
    x_feature = []
    doc_len_list = []
    vocab_set = set()

    for i in tqdm(range(start, end)):

        doc_words = shuffle_doc_words_list[i].split()
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j : j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.0
                    else:
                        word_pair_count[word_pair_key] = 1.0
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.0
                    else:
                        word_pair_count[word_pair_key] = 1.0

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.0)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(
                word_embeddings[k] if k in word_embeddings else oov[k]
            )

        x_adj.append(adj)
        x_feature.append(features)

    return x_adj, x_feature


class GraphDataset(DGLDataset):
    """
    A dataset class

    ...

    Attributes
    ----------
    x_adj (list): list of scipy sparse adjacency matrices

    x_feature (list): list of node matrices

    targets (list): list of industry tags


    """

    def __init__(self, x_adj, x_feature, targets=None):

        self.adj_matrix = x_adj
        self.node_matrix = x_feature
        self.targets = targets

    def __len__(self):
        return len(self.adj_matrix)

    def __getitem__(self, idx):
        """
        Returns a Graph and tensor of label

        """

        scipy_adj = self.adj_matrix[idx]
        G = dgl.from_scipy(scipy_adj)
        # feat = torch.zeros((len(self.node_matrix[idx]), 50))
        # self.n = self.node_matrix[idx]
        # for item in self.n:
        #     feat[int(item[0])] = torch.tensor(item[1], dtype = torch.float)

        # G.ndata['feat'] = feat
        G.ndata["feat"] = torch.stack(
            [torch.tensor(x, dtype=torch.float) for x in self.node_matrix[idx]]
        )

        G = dgl.add_self_loop(G)
        if self.targets is not None:
            label = self.targets[idx]

            return G, torch.tensor(label, dtype=torch.long)

        return G


# Graph Neural Network with normal Convolutional Layers
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.avgpooling = AvgPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = self.avgpooling(g, h)

        return self.classify(h)


# Graph Neural Network with Attention Layers where
# the node features are concatenated for attention
class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GATClassifier, self).__init__()
        self.hid_dim = hidden_dim
        self.gat1 = GATConv(in_dim, hidden_dim, num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, 1)
        self.avgpooling = AvgPooling()
        self.drop = nn.Dropout(p=0.3)
        #         self.maxpooling = MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        bs = h.shape[0]
        h = F.relu(self.gat1(g, h))
        h = h.reshape(bs, -1)
        h = F.relu(self.gat2(g, h))
        h = h.reshape(bs, -1)
        h = self.drop(h)
        h = self.avgpooling(g, h)
        #         hmax = self.maxpooling(g, h)
        #         h = torch.cat([havg, hmax], 1)

        return self.classify(h)


# Graph Neural Network with Attention
# Layers where a dot product is performed between node features
class GATDotClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GATDotClassifier, self).__init__()
        self.hid_dim = hidden_dim
        self.gat1 = DotGatConv(in_dim, hidden_dim, num_heads)
        self.gat2 = DotGatConv(hidden_dim * num_heads, hidden_dim, 1)
        self.avgpooling = AvgPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        bs = h.shape[0]
        h = F.relu(self.gat1(g, h))
        h = h.reshape(bs, -1)
        h = F.relu(self.gat2(g, h))
        h = h.reshape(bs, -1)
        h = self.avgpooling(g, h)

        return self.classify(h)


def train_fold(args, adj_list, node_list, fold=0):
    """
    Returns dictionary with loss, f1, auc and mrr as the
    keys and list containing their epochwise scores as values.
    This function trains and validates a model over a fold of dataset

        Parameters:
            args (class): Class containing variables specifying
            values necessary for training model
            adj_list (list): list of adjacency matrices
            node_list (list): list of node matrices
            fold (int): fold to validate model on.

        Returns:
            dictionary


    """

    train_idx = list(
        traindf.index
    )  # list(traindf[traindf['fold']!=fold].index)
    val_idx = list(testdf.index)  # list(traindf[traindf['fold']==fold].index)

    print("Num train samples ", len(train_idx))
    print("Num val samples ", len(val_idx))

    num_classes = traindf["target"].nunique()

    train = traindf  # traindf[traindf['fold']!=fold].reset_index(drop = True)
    val = testdf  # traindf[traindf['fold']==fold].reset_index(drop = True)

    train_adj_list, val_adj_list = [adj_list[i] for i in train_idx], [
        adj_list[i] for i in val_idx
    ]
    train_node_list, val_node_list = [node_list[i] for i in train_idx], [
        node_list[i] for i in val_idx
    ]
    train_label_list, val_label_list = (
        train["target"].values,
        val["target"].values,
    )

    traindataset = GraphDataset(
        train_adj_list, train_node_list, train_label_list
    )
    valdataset = GraphDataset(val_adj_list, val_node_list, val_label_list)

    trainloader = GraphDataLoader(
        traindataset, batch_size=args.batch_size, shuffle=True
    )
    valloader = GraphDataLoader(
        valdataset, batch_size=args.batch_size, shuffle=False
    )

    model = GATClassifier(
        args.embedding_dim, args.hidden_dim, args.num_heads, num_classes
    )
    criterion = CrossEntropyLoss()  # weight = weights
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    # best_val_mrr = 0
    best_val_acc = 0

    loss = []
    f1 = []
    auc = []
    mrr = []

    for idx in range(args.max_epochs):
        print(f"Epoch {idx + 1}/{args.max_epochs}")

        (
            train_loss,
            train_f1,
            train_auc,
            train_mrr,
            train_acc,
        ) = train_one_epoch(
            trainloader, model, criterion, optimizer, scheduler, num_classes
        )
        (
            val_loss,
            val_f1,
            val_auc,
            val_mrr,
            val_acc,
            one_hot_labels,
            all_logits,
        ) = validate(valloader, model, criterion, num_classes)

        print("train acc, val_acc", train_acc, val_acc)
        log_results(
            train_loss,
            train_f1,
            train_auc,
            train_mrr,
            val_loss,
            val_f1,
            val_auc,
            val_mrr,
            idx,
        )

        loss.append((train_loss, val_loss))
        f1.append((train_f1, val_f1))
        auc.append((train_auc, val_auc))
        mrr.append((train_mrr, val_mrr))

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "model.pt")
            # torch.save(model.state_dict(), f"model.pt")
            best_val_acc = val_acc

            plt.rcParams.update({"font.size": 20})
            conf_mat = confusion_matrix(one_hot_labels, all_logits)
            fig, ax = plt.subplots(figsize=(16, 16))
            sns.heatmap(
                conf_mat / conf_mat.sum(axis=1)[:, np.newaxis],
                annot=True,
                # fmt="d",
                fmt=".1%",
                cbar=False,
                square=True,
                cmap=sns.diverging_palette(20, 220, n=200),
                # xticklabels=category_id_df[key].values,
                # yticklabels=category_id_df[key].values,
            )
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig("conf.pdf")
            plt.close()
            f = open("accuracy", "w")
            f.write(str(accuracy_score(one_hot_labels, all_logits)))
            f.close()

    return {"loss": loss, "f1": f1, "auc": auc, "mrr": mrr}


def train_one_epoch(
    trainloader, model, criterion, optimizer, scheduler, num_classes
):
    """
    Returns training loss, f1, roc_auc and mrr scores over 1 epoch
    This function trains model for 1 epoch

    Parameters:
        trainloader (DataLoader/Iterable): dataloader that yields a
         batch for training
        model (nn.Module): model used for training
        criterion (nn.Module): loss function
        optimizer (Optimizer): used to optimize the loss function
        scheduler (Scheduler): used to change the learning rate over epochs
        num_classes (int): number of classes

    Returns:
        loss, f1, roc_auc and mrr floats


    """
    train_loss = 0
    train_f1 = 0
    train_auc = 0

    all_labels = []
    all_logits = []
    all_logits_argmax = []

    total = len(trainloader)
    model.train()
    for idx, (G, label) in tqdm(enumerate(trainloader), total=total):

        h = G.ndata["feat"].float()
        logit = model(G, h)
        loss = criterion(logit, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        label_numpy = label.detach().cpu().numpy()
        logit_numpy = logit.softmax(-1).detach().cpu().numpy()

        train_loss += loss.item() / total
        train_f1 += (
            sklearn.metrics.f1_score(
                label_numpy, logit_numpy.argmax(-1), average="micro"
            )
            / total
        )

        all_labels.append(label_numpy)
        all_logits.append(logit_numpy)
        all_logits_argmax.append(logit_numpy.argmax(-1))

    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    all_logits_argmax = np.concatenate(all_logits_argmax)
    one_hot_labels = np.zeros((len(all_labels), num_classes))
    one_hot_labels[np.arange(len(all_labels)), all_labels] = 1.0

    train_auc = sklearn.metrics.roc_auc_score(
        all_labels,
        all_logits,
        multi_class="ovo",
        labels=np.array([int(i) for i in range(num_classes)]),
    )
    train_mrr = sklearn.metrics.label_ranking_average_precision_score(
        one_hot_labels, all_logits
    )
    train_acc = sklearn.metrics.accuracy_score(all_labels, all_logits_argmax)
    # print ('train_acc',train_acc)

    return train_loss, train_f1, train_auc, train_mrr, train_acc


def validate(valloader, model, criterion, num_classes):
    """
    Returns validation loss, f1, roc_auc and mrr scores over 1 epoch
    This function validates the model

        Parameters:
            valloader (DataLoader/Iterable):
            dataloader that yields a batch for validating
            model (nn.Module): model to be used for validation
            criterion (nn.Module): loss function
            num_classes (int): number of classes

        Returns:
            loss, f1, roc_auc and mrr floats


    """

    val_loss = 0
    val_f1 = 0
    val_auc = 0

    all_labels = []
    all_logits = []
    all_logits_argmax = []
    total = len(valloader)
    model.eval()

    with torch.no_grad():
        for idx, (G, label) in tqdm(enumerate(valloader), total=total):

            h = G.ndata["feat"].float()
            logit = model(G, h)
            loss = criterion(logit, label)

            label_numpy = label.detach().cpu().numpy()
            logit_numpy = logit.softmax(-1).detach().cpu().numpy()

            val_loss += loss.item() / total
            val_f1 += (
                sklearn.metrics.f1_score(
                    label_numpy, logit_numpy.argmax(-1), average="micro"
                )
                / total
            )

            all_labels.append(label_numpy)
            all_logits.append(logit_numpy)
            all_logits_argmax.append(logit_numpy.argmax(-1))
        all_labels = np.concatenate(all_labels)
        all_logits = np.concatenate(all_logits)
        all_logits_argmax = np.concatenate(all_logits_argmax)

        one_hot_labels = np.zeros((len(all_labels), num_classes))
        one_hot_labels[np.arange(len(all_labels)), all_labels] = 1.0

        val_auc = sklearn.metrics.roc_auc_score(
            all_labels,
            all_logits,
            multi_class="ovo",
            labels=np.array([int(i) for i in range(num_classes)]),
        )
        val_mrr = sklearn.metrics.label_ranking_average_precision_score(
            one_hot_labels, all_logits
        )
        # train_acc = sklearn.metrics.accuracy_score(all_labels, all_logits)
        # print ('train_acc',train_acc)
        val_acc = sklearn.metrics.accuracy_score(all_labels, all_logits_argmax)
        # print ('val_acc',val_acc)

    return (
        val_loss,
        val_f1,
        val_auc,
        val_mrr,
        val_acc,
        all_labels,
        all_logits_argmax,
    )


def log_results(
    train_loss,
    train_f1,
    train_auc,
    train_mrr,
    val_loss,
    val_f1,
    val_auc,
    val_mrr,
    idx,
):
    """
    This function logs all the metric values to wandb project

        Parameters:
            ints/floats of values to be logged by wandb logger

        Returns:
            Nothing


    """

    metric_dict = {
        "train_loss": train_loss,
        "train_f1": train_f1,
        "train_auc": train_auc,
        "train_mrr": train_mrr,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "val_mrr": val_mrr,
        "epoch": idx,
    }

    print(metric_dict)


def test(args, n_classes):
    """
    Returns test dataframe with 'preds_list' and 'preds' as two new columns.
    'preds_list' has a list for each description with
    predictions sorted in descending order of their softmax score.
    It can be used for test mrr evaluation. 'preds' column has the first entry
    of the 'preds_list' list for each sample.

        Parameters:
            args (class): Class containing variables specifying
            values necessary for training model
            n_classes (int): number of classes

        Returns:
            dataframe


    """

    num_classes = n_classes
    # window_size = args.window_size

    print("building graphs for training")
    x_adj, x_feature = build_graph(
        start=len(traindf), end=len(traindf) + len(testdf), weighted_graph=True
    )

    testdataset = GraphDataset(x_adj, x_feature)
    testloader = GraphDataLoader(
        testdataset, batch_size=args.batch_size, shuffle=False
    )

    model = GATClassifier(
        args.embedding_dim, args.hidden_dim, args.num_heads, num_classes
    )
    model_list = [model]  # load_models(model, args.n_folds)

    pred_list = []

    with torch.no_grad():
        for idx, G in enumerate(tqdm(testloader)):
            h = G.ndata["feat"].float()
            logits = 0
            for mod in model_list:
                log = mod(G, h)
                # blending of logits from all 5 models. i
                # This helps in getting more robust predictions.
                logits += log.softmax(-1) / args.n_folds

            pred_soft = logits.detach().cpu().numpy()
            pred_list.append(pred_soft)

        preds = np.concatenate(pred_list)

    tags = []

    for sample in preds:
        sample = sample.argsort(-1)[::-1]
        x = [idx2label[i] for i in sample]
        tags.append(f"{x}")

    testdf["preds_list"] = tags

    preds = preds.argmax(-1)
    preds = [idx2label[i] for i in preds]

    testdf["preds"] = preds

    return testdf

    # wandb.log(metric_dict)


labels = df_csv[key].unique()
print("Number of unique categories tags: ", len(labels))
label2idx = {l: i for i, l in enumerate(sorted(labels))}
idx2label = {v: k for k, v in label2idx.items()}

df_csv["target"] = df_csv[key].apply(lambda x: label2idx[x])
df_csv["text"] = df_csv[value].apply(lambda x: filter_text(x))
# df_csv['text'] = df_csv['title'].apply(lambda x: filter_text(x))
df_csv["text_len"] = df_csv[value].apply(lambda x: len(x.split()))
df_csv["unique_text_len"] = df_csv[value].apply(lambda x: len(set(x.split())))

n_train = int(len(df_csv) * ratio)
n_test = len(df_csv) - n_train

traindf = df_csv[0:n_train]
testdf = df_csv[n_train:-1]
# traindf.columns = ['company', 'text', 'target']


print("Checking and removing rows with na values in the dataset \n")
print(traindf.isna().sum())
traindf = traindf.dropna(axis=0).reset_index(drop=True)
print("*" * 50)
print("After processing \n")
print(traindf.isna().sum())

print(
    "Max of mean text length: ",
    traindf.groupby(by="target")["text_len"].mean().max(),
)
print(
    "Tag of max of mean text length: ",
    idx2label[traindf.groupby(by="target")["text_len"].mean().argmax()],
)

print(
    "Min of mean text length: ",
    traindf.groupby(by="target")["text_len"].mean().min(),
)


word_embeddings = {}

with open(GLOVE_EMBEDDING_PATH, "r") as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float, data[1:]))
vocab = build_vocab(list(traindf["text"].apply(lambda x: x.split())))
oov = check_coverage(vocab, word_embeddings)
traindf["text"] = traindf["text"].apply(lambda x: " ".join(x.split("-")))
vocab = build_vocab(list(traindf["text"].apply(lambda x: x.split())))
oov = check_coverage(vocab, word_embeddings)


word_embeddings_dim = args.embedding_dim

shuffle_doc_words_list = list(traindf["text"].values) + list(
    testdf["text"].values
)

word_set = set()

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)

vocab = list(word_set)
vocab_size = len(vocab)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.1, 0.1, word_embeddings_dim)


window_size = args.window_size

print("building graphs for training")
# x_adj, x_feature =
# build_graph(start=0, end=len(traindf), weighted_graph = True)
x_adj, x_feature = build_graph(
    start=0, end=len(traindf) + len(testdf), weighted_graph=True
)
result_list = []
# for i in range(args.n_folds):
#     result = train_fold(args, x_adj, x_feature, fold = i)
#     result_list.append(result)
result = train_fold(args, x_adj, x_feature, fold=0)
result_list.append(result)
