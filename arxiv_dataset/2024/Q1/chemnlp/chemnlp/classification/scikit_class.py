"""Module for classification tasks."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import sys

# from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from jarvis.db.figshare import data
from jarvis.db.jsonutils import dumpjson
import pickle
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="ChemNLP package.")
parser.add_argument(
    "--csv_path",
    default="filename.csv",
    help="Path for comma separated file.",
)
parser.add_argument(
    "--test_ratio",
    default=0.2,
    help="Test split ratio e.g. 0.2",
)

parser.add_argument(
    "--key_column",
    default="categories",
    help="Column name for classes in csv file",
)

parser.add_argument(
    "--value_column",
    default="title_abstract",
    help="Column name for text in csv file",
)

parser.add_argument(
    "--min_df",
    default=5,
    help="Minimum numbers of documents a word must be present in to be kept",
)
parser.add_argument(
    "--shuffle",
    default=False,
    help="Whether or not shuffle the rows in csv before splitting",
)


def generate_data(n_entries=None, nentry_each=None, filename="cond_mat.csv"):
    """Generate cond-mat dataset."""
    d = data("arXiv")
    df = pd.DataFrame(d)
    cond_mat_topics = [
        "cond-mat.mtrl-sci",
        "cond-mat.mes-hall",
        "cond-mat.str-el",
        "cond-mat.stat-mech",
        "cond-mat.supr-con",
        "cond-mat.soft",
        "cond-mat.quant-gas",
        "cond-mat.other",
        "cond-mat.dis-nn",
    ]
    df_cond_mat = df[
        df["categories"].apply(
            lambda x: "cond-mat" in x and x in cond_mat_topics
        )
    ]
    if n_entries is not None:
        df_cond_mat = df_cond_mat[0:n_entries]
    if nentry_each is not None:
        x = []
        for i in cond_mat_topics:
            x.extend(
                df_cond_mat[df_cond_mat["categories"] == i][0:nentry_each][
                    "id"
                ].values
            )
        df_cond_mat = df_cond_mat[df_cond_mat["id"].isin(x)]

    dff = df_cond_mat[["id", "title", "abstract", "categories"]]
    dff["title_abstract"] = dff["title"] + " " + df["abstract"]
    dff = dff.sample(frac=1, replace=True, random_state=1)
    dff.to_csv(filename)
    # print(len(set(dff.categories)))


def sk_class(
    csv_path=None,
    key="categories",
    value="title_abstract",
    test_size=0.2,
    min_df=5,
    ngram_range=[1, 2],
    print_common=False,
    model=None,
    categorize=True,  # False is buggy, need to check
    shuffle=False,
):
    """Classifcy data using scikit-learn library algos."""
    df = pd.read_csv(csv_path, dtype="str")
    # df=pd.read_csv(csv_path,dtype={'id':'str'})
    if categorize:
        df["category_id"] = df[key].factorize()[0]
        category_id_df = (
            df[[key, "category_id"]].sort_values("category_id")
            # df[[key, "category_id"]].drop_duplicates()
            # .sort_values("category_id")
        )

        category_to_id = dict(category_id_df.values)
        print("category_to_id", category_to_id)
    print(df)
    print(df[key].value_counts())
    title_tfidf = TfidfVectorizer(
        sublinear_tf=True,
        # logarithmic form for frequency
        min_df=min_df,
        # minimum numbers of documents a word must be present in to be kept
        norm="l2",
        # norm is set to l2, to ensure all our feature
        # vectors have a euclidian norm of 1
        encoding="latin-1",  # utf-8 possible
        ngram_range=ngram_range,
        # (1, 2) to indicate that we want to consider both unigrams and bigrams
        stop_words="english",
        # remove common pronouns ("a", "the", ...)
        # to reduce the number of noisy features
    )
    # Title features:
    title_features = title_tfidf.fit_transform(
        df[value]
    )  # .toarray()  # title
    if categorize:
        title_labels = df.category_id  # categories (cond-mat)
    else:
        title_labels = df[key]  # categories (cond-mat)
    # title_labels = df.category_id  # categories (cond-mat)
    print(
        "title_features.shape", title_features.shape
    )  # titles represented by features,
    # representing tf-idf score for different unigrams and bigrams
    if model is None:
        model = LogisticRegression(random_state=0)  # LinearSVC()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(
        title_features,
        title_labels,
        df.index,
        test_size=test_size,
        random_state=0,
        shuffle=shuffle,
    )
    if not categorize:
        category_to_id = list(set(df[key].values))
    id_to_category = dict((v, k) for k, v in category_to_id.items())
    info = {}
    train_info = {}
    test_info = {}
    for i, j in zip(df.id[indices_train], y_train):
        train_info[i] = id_to_category[int(j)]  # j
    for i, j in zip(df.id[indices_test], y_test):
        test_info[i] = id_to_category[int(j)]  # j
    info["train"] = train_info
    info["test"] = test_info
    dumpjson(data=info, filename="arXiv_categories.json")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("pred_test.csv", "w")
    # f.write(str(accuracy_score(y_test, y_pred)))
    # f.close()
    line = "id,target,prediction\n"
    f.write(line)
    # pickle.dump(model, open("log.pk", "wb"))
    for i, j, k in zip(df.id[indices_test], y_test, y_pred):
        # print(i,j,k)
        line = (
            str(i)
            + ","
            + str(id_to_category[int(j)])
            + ","
            + str(id_to_category[int(k)])
            + "\n"
        )
        f.write(line)
    f.close()
    # print ('indices_train',df.id[indices_train]
    # .astype(str),len(indices_train))
    # print('indices_test',df.id[indices_test].astype(str),len(indices_test))
    print("Logistic model accuracy", accuracy_score(y_test, y_pred))
    plt.rcParams.update({"font.size": 20})
    conf_mat = confusion_matrix(y_test, y_pred)  # ,labels=category_to_id)
    fig, ax = plt.subplots(figsize=(16, 16))
    # print('category_id_df[key].values',category_id_df[key].values)
    sns.heatmap(
        conf_mat / conf_mat.sum(axis=1)[:, np.newaxis],
        annot=True,
        # fmt="d",
        fmt=".1%",
        cbar=False,
        square=True,
        cmap=sns.diverging_palette(20, 220, n=200),
        # xticklabels=category_id_df[k].values,
        # yticklabels=category_id_df[k].values,
        xticklabels=category_to_id,
        yticklabels=category_to_id,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("conf_log.pdf")
    plt.close()


def scikit_classify(
    df=None,
    key="term",
    value="title",
    CV=3,
    test_size=0.2,
    min_df=5,
    ngram_range=[1, 2],
    print_common=False,
    model_selection=False,
):
    """Classifcy data using scikit-learn library algos."""
    print(df)
    print(df[key].value_counts())
    df["category_id"] = df[key].factorize()[0]
    category_id_df = (
        df[[key, "category_id"]].drop_duplicates().sort_values("category_id")
    )

    category_to_id = dict(category_id_df.values)
    # id_to_category = dict(category_id_df[["category_id", key]].values)

    print(df.head())

    title_tfidf = TfidfVectorizer(
        sublinear_tf=True,
        # logarithmic form for frequency
        min_df=min_df,
        # minimum numbers of documents a word must be present in to be kept
        norm="l2",
        # norm is set to l2, to ensure all our feature
        # vectors have a euclidian norm of 1
        encoding="latin-1",  # utf-8 possible
        ngram_range=ngram_range,
        # (1, 2) to indicate that we want to consider both unigrams and bigrams
        stop_words="english",
        # remove common pronouns ("a", "the", ...)
        # to reduce the number of noisy features
    )

    # Title features:
    title_features = title_tfidf.fit_transform(
        df[value]
    )  # .toarray()  # title
    title_labels = df.category_id  # categories (cond-mat)
    print(
        "title_features.shape", title_features.shape
    )  # titles represented by features,
    # representing tf-idf score for different unigrams and bigrams

    if print_common:
        N = 2  # number of correlated items to display
        for title, category_id in sorted(category_to_id.items()):
            features_chi2 = chi2(title_features, title_labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(title_tfidf.get_feature_names_out())[
                indices
            ]
            unigrams = [v for v in feature_names if len(v.split(" ")) == 1]
            bigrams = [v for v in feature_names if len(v.split(" ")) == 2]
            print("# '{}':".format(title))
            print(
                "  . Most correlated unigrams:\n. {}".format(
                    "\n. ".join(unigrams[-N:])
                )
            )
            print(
                "  . Most correlated bigrams:\n. {}".format(
                    "\n. ".join(bigrams[-N:])
                )
            )
    if model_selection:
        models = [
            # GradientBoostingClassifier(),
            RandomForestClassifier(),
            # n_estimators=200, max_depth=3, random_state=0),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state=0),
            # try: multi_class='multinomial',
            # solver='lbfgs'...same accuracy actually
        ]

        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(
                model, title_features, title_labels, scoring="accuracy", cv=CV
            )  # features: ; labels:
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(
            entries, columns=["model_name", "fold_idx", "accuracy"]
        )

        print(cv_df)
        # Visualize model accuracy with box plot and strip plot:

        plt.figure(figsize=(10, 8))
        plt.rcParams.update({"font.size": 18})
        sns.boxplot(x="model_name", y="accuracy", data=cv_df)
        sns.stripplot(
            x="model_name",
            y="accuracy",
            data=cv_df,
            size=8,
            jitter=True,
            edgecolor="gray",
            linewidth=2,
        )

        plt.ylim(0, 1)

        # plt.title('Model Benchmark Comparison from Title Text Data')
        plt.ylabel("Accuracy (cross-validation score)")
        # plt.xlabel('Model')
        plt.xlabel("")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("cv.pdf")
        plt.close()
    print("Training logistic regression:")
    model = LogisticRegression(random_state=0)  # LinearSVC()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(
        title_features,
        title_labels,
        df.index,
        test_size=test_size,
        random_state=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_logistic", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("Logistic", accuracy_score(y_test, y_pred))
    pickle.dump(model, open("log.pk", "wb"))

    plt.rcParams.update({"font.size": 20})
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(
        conf_mat / conf_mat.sum(axis=1)[:, np.newaxis],
        annot=True,
        # fmt="d",
        fmt=".1%",
        cbar=False,
        square=True,
        cmap=sns.diverging_palette(20, 220, n=200),
        xticklabels=category_id_df[key].values,
        yticklabels=category_id_df[key].values,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("conf_log.pdf")
    plt.close()

    print("Training SVC:")
    model = LinearSVC()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(
        title_features,
        title_labels,
        df.index,
        test_size=test_size,
        random_state=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SVC", accuracy_score(y_test, y_pred))
    pickle.dump(model, open("svc.pk", "wb"))

    plt.rcParams.update({"font.size": 20})
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(
        conf_mat / conf_mat.sum(axis=1)[:, np.newaxis],
        annot=True,
        # fmt="d",
        fmt=".1%",
        cbar=False,
        square=True,
        cmap=sns.diverging_palette(20, 220, n=200),
        xticklabels=category_id_df[key].values,
        yticklabels=category_id_df[key].values,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("conf_svc.pdf")
    plt.close()
    f = open("accuracy_svc", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()


if __name__ == "__main__":
    # python generate_data.py
    # generate_data()
    # sk_class(csv_path='cond_mat.csv')
    args = parser.parse_args(sys.argv[1:])
    csv_path = args.csv_path
    test_ratio = float(args.test_ratio)
    key_column = args.key_column
    value_column = args.value_column
    min_df = int(args.min_df)
    shuffle = args.shuffle

    sk_class(
        csv_path=csv_path,
        test_size=test_ratio,
        key=key_column,
        value=value_column,
        min_df=min_df,
        shuffle=shuffle,
    )
    # df = pd.read_csv("cond_mat.csv")
    # scikit_classify(df=df, key="categories", value="title_abstract")
