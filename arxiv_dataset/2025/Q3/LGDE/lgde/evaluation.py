"""Code for dictionary evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


def evaluate_prediction(kw_list, y_true, document_term_matrix, with_report=True):
    """Evaluate list of keywords through performance of simple classifier.

    Parameters:
    -----------
    kw_list : list[str]
        List of keywords to use for prediction of document class.

    y_true : array of shape (n_documents, 1)
        True document classes (0 or 1).

    document_term_matrix : pandas DataFrame of shape (n_documents, n_words)
        Document term matrix with column indices corresponding to all words considered.
        All keywords in kw_list should correspond to columns in matrix.

    with_report : bool
        Print classification report if True.as_integer_ratio

    Returns:
    --------
    precision : float
        Macro precision score.

    recall : float
        Macro recall score.

    f1 : float
        Macro F1 score.
    --------

    """
    # predict document classes through simple keyword matching
    y_pred = np.sum(np.asarray(document_term_matrix[kw_list]), axis=1) > 0

    if with_report:
        # print classification report
        print(classification_report(y_true, y_pred, digits=3, labels=np.unique(y_pred)))

    # compute macro scores
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=np.unique(y_pred)
    )

    return precision, recall, f1


def error_analysis(lgde, lr, return_df=False):
    """Detailed error analysis based on likelihood ratios.

    Parameters:
    -----------
    lgde: LGDE object
        LGDE object after dictionary expansion.
    lr: pd.Series
        Likelihood ratios (LR) for all words in vocabulary. LR is positive
        when word appears more often in positive than negative class.
    """
    # for error analysis retrieve LGDE discovered words with LR < 1
    lr_small = lr[lgde.discovered_dict_][lr[lgde.discovered_dict_] <= 1]
    lr_small = pd.DataFrame(lr_small.sort_index(), columns=["LR"])
    lr_small["Seed (sim)"] = ""

    # iterate through discovered words to find seed keywords and similarities
    for disc_word in lr_small.index:
        word_seeds = []
        for seed_kw, community in lgde.semantic_communities_.items():
            if disc_word in community:
                similarity = np.round(
                    lgde.pairwise_word_similarity(seed_kw, disc_word), 3
                )
                lr_small.loc[disc_word, "Seed (sim)"] += f"{seed_kw} ({similarity}) "

    print("ERROR ANALYSIS DISCOVERED KWs:")
    display(lr_small)

    # compute statistics for each seed keyword
    seed_kw_error = pd.DataFrame(index=lgde.seed_dict)
    seed_kw_error["Seed LR"] = 0
    seed_kw_error["# community"] = 0
    seed_kw_error["LR <= 1 (in %)"] = 0
    seed_kw_error["Min LR"] = 0
    seed_kw_error["Median LR"] = 0
    seed_kw_error["Mean LR"] = 0
    seed_kw_error["Max LR"] = 0
    seed_kw_error["# Inf LR"] = 0

    # iterate through seed keywords to compute community stats
    for seed_kw, community in lgde.semantic_communities_.items():
        seed_kw_error.loc[seed_kw, "Seed LR"] = lr[seed_kw]
        # get finite likelihood ratios
        lr_c = lr[community][np.isfinite(lr[community])]
        seed_kw_error.loc[seed_kw, "LR <= 1 (in %)"] = round(np.mean(lr_c < 1) * 100, 2)
        seed_kw_error.loc[seed_kw, "# community"] = len(community)
        seed_kw_error.loc[seed_kw, "Min LR"] = np.min(lr_c)
        seed_kw_error.loc[seed_kw, "Median LR"] = np.median(lr_c)
        seed_kw_error.loc[seed_kw, "Mean LR"] = np.mean(lr_c)
        seed_kw_error.loc[seed_kw, "Max LR"] = np.max(lr_c)
        seed_kw_error.loc[seed_kw, "# Inf LR"] = len(community) - len(lr_c)

    print("ERROR ANALYSIS SEED KWs:")
    display(seed_kw_error)

    if return_df:
        return lr_small, seed_kw_error
