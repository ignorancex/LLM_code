from operator import index
from regex import F
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
from datetime import date
import faiss
import json
import torch
import tqdm
import numpy as np
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from eval_utils import QADataset
import traceback


corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],

    "Textbooks_single": ["textbooks_1"],
    "PubMed_single":["pubmed_1"],
    "StatPearls_single":["statpearls_1"],
    "Wikipedia_single":["wikipedia_1"],
    "medcorp_singleRAG":["medcorp_1"],

    "Textbooks_5": [
        "textbooks_half",
        "textbooks_1",
        "textbooks_2",
        "textbooks_4",
        "textbooks_8",
    ],
    "Textbooks_5_graph": [
        "textbooks_half",
        "textbooks_con_1",
        "textbooks_graph_1",
        "textbooks_con_2",
        "textbooks_graph_2",
    ],
    "Statpearls_5": [
        "statpearls_half",
        "statpearls_1",
        "statpearls_2",
        "statpearls_4",
        "statpearls_8",
    ],
    "Statpearls_5_graph": [
        "statpearls_half",
        "statpearls_con_1",
        "statpearls_graph_1",
        "statpearls_con_2",
        "statpearls_graph_2",
    ],
    "Pubmed_5": [
        "pubmed_half",
        "pubmed_1",
        "pubmed_2",
        "pubmed_4",
        "pubmed_8",
    ],
    "Pubmed_5_graph": [
        "pubmed_half",
        "pubmed_con_1",
        "pubmed_graph_1",
        "pubmed_con_2",
        "pubmed_graph_2",
    ],
    "Wikipedia_5": [
        "wikipedia_half",
        "wikipedia_1",
        "wikipedia_2",
        "wikipedia_4",
        "wikipedia_8",
    ],
    "Wikipedia_5_graph": [
        "wikipedia_half",
        "wikipedia_con_1",
        "wikipedia_graph_1",
        "wikipedia_con_2",
        "wikipedia_graph_2",
    ],
    "Medcorp_5": [
        "medcorp_half",
        "medcorp_1",
        "medcorp_2",
        "medcorp_4",
        "medcorp_8",
    ],
    "Medcorp_5_graph": [
        "medcorp_half",
        "medcorp_con_1",
        "medcorp_graph_1",
        "medcorp_con_2",
        "medcorp_graph_2",
    ],
}

retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["facebook_contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
    "RRF-2": ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4": [
        "bm25",
        "facebook_contriever",
        "allenai/specter",
        "ncbi/MedCPT-Query-Encoder",
    ],
    "RRF-5": ["bm25", "bm25", "bm25", "bm25", "bm25"],
}


class CustomizeSentenceTransformer(
    SentenceTransformer
):  # change the default pooling "MEAN" to "CLS"

    def _load_auto_model(self, model_name_or_path):
        """
        Creates a simple Transformer + CLS Pooling model and returns the modules
        """
        print(
            "No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(
                model_name_or_path
            )
        )
        transformer_model = Transformer(model_name_or_path)
        print(f"Transformer model initialized. Model name: {model_name_or_path}")
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")
        print("Pooling model created with the previously mentioned transformer model.")
        return [transformer_model, pooling_model]


def embed(chunk_dir, index_dir, model_name, **kwarg):

    save_dir = os.path.join(index_dir, "embedding")

    if "contriever" in model_name:
        model = SentenceTransformer(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        model = CustomizeSentenceTransformer(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    model.eval()

    fnames = sorted(
        [fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")]
    )
    print(f"There are {len(fnames)} files inside this corpus to be processed.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue
            texts = [
                json.loads(item) for item in open(fpath).read().strip().split("\n")
            ]
            if "specter" in model_name.lower():
                texts = [
                    model.tokenizer.sep_token.join([item["title"], item["content"]])
                    for item in texts
                ]
            elif "contriever" in model_name.lower():
                texts = [
                    ". ".join([item["title"], item["content"]])
                    .replace("..", ".")
                    .replace("?.", "?")
                    for item in texts
                ]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts]
            embed_chunks = model.encode(texts, **kwarg)
            np.save(save_path, embed_chunks)
        embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]


def construct_index(index_dir, model_name, h_dim=768):

    with open(os.path.join(index_dir, "metadatas.jsonl"), "w") as f:
        f.write("")

    if "specter" in model_name.lower():
        index = faiss.IndexFlatL2(h_dim)
    else:
        index = faiss.IndexFlatIP(h_dim)

    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        index.add(curr_embed)
        with open(os.path.join(index_dir, "metadatas.jsonl"), "a+") as f:
            f.write(
                "\n".join(
                    [
                        json.dumps({"index": i, "source": fname.replace(".npy", "")})
                        for i in range(len(curr_embed))
                    ]
                )
                + "\n"
            )

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index


class Retriever:

    def __init__(
        self,
        retriever_name="ncbi/MedCPT-Query-Encoder",
        corpus_name="textbooks",
        db_dir="./corpus",
        retriever_cache_dir="pt_models/",
        threshold=None,
        **kwarg,
    ):
        # print("Building retriever...")

        self.retriever_name = retriever_cache_dir + retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.threshold = threshold

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        if not os.path.exists(self.chunk_dir):
            print(f"Didn't find the chunk folder at [{self.chunk_dir}]")
            print(
                "Cloning the {:s} corpus from Huggingface...".format(self.corpus_name)
            )
            os.system(
                "git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(
                    corpus_name, os.path.join(self.db_dir, self.corpus_name)
                )
            )
            if self.corpus_name == "statpearls":
                print("Downloading the statpearls corpus from NCBI bookshelf...")
                os.system(
                    "wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(
                        os.path.join(self.db_dir, self.corpus_name)
                    )
                )
                os.system(
                    "tar -xzvf {:s} -C {:s}".format(
                        os.path.join(
                            db_dir, self.corpus_name, "statpearls_NBK430685.tar.gz"
                        ),
                        os.path.join(self.db_dir, self.corpus_name),
                    )
                )
                print("Chunking the statpearls corpus...")
                os.system("python src/data/statpearls.py")
        # print("Corpus initialized.")

        self.index_dir = os.path.join(
            self.db_dir,
            self.corpus_name,
            "index",
            retriever_name.replace("Query-Encoder", "Article-Encoder"),
        )

        if "bm25" in self.retriever_name.lower():
            # print("Using BM25 retriver.")
            from pyserini.search.lucene import LuceneSearcher

            self.metadatas = None
            self.embedding_function = None
            if os.path.exists(self.index_dir):
                # print("Found the path of the index: {:s}".format(self.index_dir))
                self.index = LuceneSearcher(os.path.join(self.index_dir))
            else:
                print(
                    "[In progress] Didn't find the index file. Creating the index now."
                )

                os.system(
                    "python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 48".format(
                        self.chunk_dir, self.index_dir
                    )
                )
                self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            faiss_index_path = os.path.join(self.index_dir, "faiss.index")
            if os.path.exists(faiss_index_path):
                # print(f"Found the path of faiss index: {faiss_index_path}")
                self.index = faiss.read_index(
                    os.path.join(self.index_dir, "faiss.index")
                )
                self.metadatas = [
                    json.loads(line)
                    for line in open(os.path.join(self.index_dir, "metadatas.jsonl"))
                    .read()
                    .strip()
                    .split("\n")
                ]
            else:
                print("Didn't find any faiss index. Creating embeddings now.")
                # print(f"The cache path for the encoder is: {self.retriever_name}. Will try to use cached model first.")
                # print("[In progress] Encoding the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))

                h_dim = embed(
                    chunk_dir=self.chunk_dir,
                    index_dir=self.index_dir,
                    model_name=self.retriever_name.replace(
                        "Query-Encoder", "Article-Encoder"
                    ),
                    **kwarg,
                )

                # print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
                self.index = construct_index(
                    index_dir=self.index_dir,
                    model_name=self.retriever_name.replace(
                        "Query-Encoder", "Article-Encoder"
                    ),
                    h_dim=h_dim,
                )
                # print("[Done] Corpus indexing finished!")
                self.metadatas = [
                    json.loads(line)
                    for line in open(os.path.join(self.index_dir, "metadatas.jsonl"))
                    .read()
                    .strip()
                    .split("\n")
                ]
            if "contriever" in retriever_name.lower():
                self.embedding_function = SentenceTransformer(
                    self.retriever_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            else:
                self.embedding_function = CustomizeSentenceTransformer(
                    self.retriever_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            self.embedding_function.eval()
        # print("Retriever initialized.")

    def get_relevant_documents(
        self,
        question,
        k=32,
        threshold=0,
        **kwarg,
    ):
        assert type(question) == str
        question = [question]

        if "bm25" in self.retriever_name.lower():
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            # filter the retrieval results
            hits = [h for h in hits if h.score >= threshold]
            res_[0].append(np.array([h.score for h in hits]))

            try:
                indices = []
                for h in hits:
                    if "|" in h.docid:
                        try:
                            ind_ele_list = h.docid.split("|")
                            source_str = "_".join(
                                ind_ele_list[0].split("_")[:-1]
                            ).split("#")[-1]
                            line_num = "_".join(ind_ele_list[0].split("_")[:-1]).split(
                                "#"
                            )[0]
                            ind_list = [
                                ind_ele.split("_")[-1] for ind_ele in ind_ele_list
                            ]
                            one_index = {
                                "source": source_str,
                                "index": ind_list,
                                "line_num": line_num,
                                "docid": h.docid,
                            }
                            indices.append(one_index)
                        except Exception as e:
                            print(f"Error in get_relevant_documents")
                            print("h.docid")
                            print(h.docid)
                    else:
                        line_num = h.docid.split("#")[0]
                        # backward compartible
                        if len(h.docid.split("#")) == 2:
                            new_h_docid = h.docid.split("#")[1]
                        else:
                            new_h_docid = h.docid.split("#")[-1]
                        source_str = "_".join(new_h_docid.split("_")[:-1])
                        one_index = {
                            "source": source_str,
                            "index": new_h_docid.split("_")[-1],
                            "line_num": line_num,
                            "docid": h.docid,
                        }
                        indices.append(one_index)
            except Exception as e:
                indices = [
                    {
                        "source": "_".join(h.docid.split("_")[:-1]),
                        "index": eval(h.docid.split("_")[-1]),
                    }
                    for h in hits
                ]
        else:
            with torch.no_grad():
                query_embed = self.embedding_function.encode(question, **kwarg)
            res_ = self.index.search(query_embed, k=k)
            indices = [self.metadatas[i] for i in res_[1][0]]
        if len(indices) == 0:
            # there is no document retrieved
            texts = ["NO_TEXT_RETRIEVED"]
            scores = [0]
        else:
            texts = self.idx2txt(indices)
            scores = res_[0][0].tolist()
        return texts, scores

    def idx2txt(self, indices):  # return List of Dict of str
        """
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of str
        """
        i = indices[0]
        json_ele_list = []

        for i in indices:
            if "line_num" not in i:
                # backwards compatibility
                json_ele_list.append(
                    json.loads(
                        open(os.path.join(self.chunk_dir, i["source"] + ".jsonl"))
                        .read()
                        .strip()
                        .split("\n")[i["index"]]
                    )
                )
            else:
                try:
                    # new format
                    jsonl_file_path = os.path.join(
                        self.chunk_dir, i["source"] + ".jsonl"
                    )
                    json_object_list = open(jsonl_file_path).read().strip().split("\n")
                    line_num_int = int(i["line_num"])
                    json_object = json.loads(json_object_list[line_num_int - 1])
                    json_ele_list.append(json_object)
                except Exception as e:
                    # unknown error
                    print(f"Error in idx2txt")
                    error_json_object = {
                        "line_num": str(i["line_num"]),
                        "jsonl_file_path": jsonl_file_path,
                        "error_message": "TEXT_RETRIEVED_IDX2TXT_ERROR",
                        "docid": i["docid"],
                    }
                    print(error_json_object)
                    print("error")
                    print(e)
                    json_ele_list.append(error_json_object)
                    raise (e)
        return json_ele_list


class RetrievalSystem:

    def __init__(
        self,
        cache_dir,
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        db_dir="../corpus",
        pred_with_router=False,
        router_model=None,
        threshold=None,
    ):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        self.pred_with_router = pred_with_router
        self.router_model = router_model
        self.threshold = threshold

        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                # print(f"initializing corpus: {corpus}")
                self.retrievers[-1].append(
                    Retriever(
                        retriever,
                        corpus,
                        db_dir,
                        retriever_cache_dir=cache_dir,
                        threshold=threshold,
                    )
                )
        print("All retrievers initialized. Retrieval system ready.")

    def retrieve(self, question, k=32, rrf_k=100, threshold=0):
        """
        Given questions, return the relevant snippets from the corpus
        """
        assert type(question) == str

        texts = []
        scores = []

        if "RRF" in self.retriever_name:
            k_ = max(k * 2, rrf_k)
        else:
            k_ = k
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(
                    question, k=k_, threshold=threshold
                )
                texts[-1].append(t)
                scores[-1].append(s)

        return texts, scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        """
        Merge the texts and scores from different retrievers
        """
        if len(texts) == 0 or len(scores) == 0:
            # if there is no documents retrieved at all
            texts = ["NO_TEXT_RETRIEVED"]
            scores = [0]
            return texts, scores
        RRF_dict = {}
        for i in range(len(retriever_names[self.retriever_name])):
            texts_all, scores_all = None, None
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]
            if "specter" in retriever_names[self.retriever_name][i].lower():
                sorted_index = np.array(scores_all).argsort()
            else:
                sorted_index = np.array(scores_all).argsort()[::-1]
            texts[i] = [texts_all[i] for i in sorted_index]
            scores[i] = [scores_all[i] for i in sorted_index]
            for j, item in enumerate(texts[i]):
                if item["id"] in RRF_dict:
                    RRF_dict[item["id"]]["score"] += 1 / (rrf_k + j + 1)
                    RRF_dict[item["id"]]["count"] += 1
                else:
                    RRF_dict[item["id"]] = {
                        "id": item["id"],
                        "title": item["title"],
                        "content": item["content"],
                        "score": 1 / (rrf_k + j + 1),
                        "count": 1,
                    }
        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        if len(texts) == 1:
            texts = texts[0][:k]
            scores = scores[0][:k]
        else:
            texts = [
                dict((key, item[1][key]) for key in ("id", "title", "content"))
                for item in RRF_list[:k]
            ]
            scores = [item[1]["score"] for item in RRF_list[:k]]
        return texts, scores


def set_split(dataset_name):
    split = "dev" if dataset_name == "medmcqa" else "test"
    return split


def profile_benchmark_dataset(
    benchmark, profiling_datasets=False, benchmark_repo_dir="PATH_TO_MIRAGE"
):
    # Profiling the datasets
    if profiling_datasets:
        print("\n Profiling the datasets in the MIRAGE benchmark...")
        print("Possible sub datasets are:")
        for dataset_name in benchmark:
            print(dataset_name)
            dataset = QADataset(data=dataset_name, dir=benchmark_repo_dir)

            print("Length of dataset:")
            print(len(dataset))

            print("First example in dataset:")
            print(dataset[0])
            print(dataset.index[0])
            print("\n")
    return 0


def prepare_prediction_output_folders(
    prediction_folder,
    dataset_name,
    rag_k,
    llm,
    corpus_name,
    retriever_name,
    clear_previous_eval_results=False,
    pred_with_router=False,
    exp_option=None,
):
    # create folder paths
    prediction_res_folder_cot = os.path.join(
        prediction_folder, dataset_name, "cot", llm
    )

    if pred_with_router:
        prediction_res_folder_rag = os.path.join(
            prediction_folder,
            dataset_name,
            f"{exp_option}_router_rag_{rag_k}",
            llm,
            corpus_name,
            retriever_name,
        )
    else:
        prediction_res_folder_rag = os.path.join(
            prediction_folder,
            dataset_name,
            f"rag_{rag_k}",
            llm,
            corpus_name,
            retriever_name,
        )

    # clear the previous results if flagged
    if clear_previous_eval_results:
        print(
            f"[In progress] Clearing previous prediction results on datasets [{dataset_name}]... \n"
        )
        # if the folders exist
        if os.path.exists(prediction_res_folder_cot):
            os.system(f"rm -rf {prediction_res_folder_cot}")
        if os.path.exists(prediction_res_folder_rag):
            os.system(f"rm -rf {prediction_res_folder_rag}")
        print("\n[Done] Previous prediction results cleared. \n")

    # create the folders if not exist already
    if not os.path.exists(prediction_res_folder_cot):
        os.makedirs(prediction_res_folder_cot)
    if not os.path.exists(prediction_res_folder_rag):
        os.makedirs(prediction_res_folder_rag)

    print("\n---------------------------\nPrediction results' folders created")
    print(f"CoT: {prediction_res_folder_cot}")
    print(f"RAG: {prediction_res_folder_rag}")
    print("---------------------------\n")
    return prediction_res_folder_cot, prediction_res_folder_rag


def prepare_prediction_output_files(
    prediction_res_folder_cot, prediction_res_folder_rag, split, ind
):
    prediction_res_file_cot = os.path.join(
        prediction_res_folder_cot, split + "_" + str(ind) + ".json"
    )
    prediction_res_file_rag = os.path.join(
        prediction_res_folder_rag, split + "_" + str(ind) + ".json"
    )
    return prediction_res_file_cot, prediction_res_file_rag


def load_benchmark_dataset(
    dataset_name="SUB_DATASET_NAME_IN_MIRAGE",
    benchmark_repo_dir="PATH_TO_MIRAGE_REPO",
    test_limit=None,
):
    dataset = QADataset(data=dataset_name, dir=benchmark_repo_dir)
    index = dataset.index
    # if test_limit is set, only run on the first test_limit samples
    if test_limit is not None:
        dataset = dataset[:test_limit]
        index = index[:test_limit]
    return dataset, index


def run_cot_single(cot, ind, question, options, prediction_res_file_cot):
    if not os.path.exists(prediction_res_file_cot):
        answer, _, _ = cot.answer(question=question, options=options)
        with open(prediction_res_file_cot, "w") as f:
            json.dump([answer], f, indent=4)
    return 0


def run_cot(
    cot,
    dataset,
    index,
    prediction_res_folder_cot,
    prediction_res_folder_rag,
    split,
    pbar,
    max_threads=1,
):

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []

        for i in range(len(dataset)):
            ind, question, options = (
                index[i],
                dataset[i]["question"],
                dataset[i]["options"],
            )
            prediction_res_file_cot, _ = prepare_prediction_output_files(
                prediction_res_folder_cot, prediction_res_folder_rag, split, ind
            )

            future = executor.submit(
                run_cot_single, cot, ind, question, options, prediction_res_file_cot
            )
            futures.append(future)

        # Wait for the futures to complete and update the progress bar
        for future in futures:
            future.result()
            pbar.update(1)

    pbar.close()

    return 0


def run_rag_single(
    medrag,
    ind,
    question,
    options,
    prediction_res_file_rag,
    rag_k,
    save_dir=None,
    threshold=0,
):
    if not os.path.exists(prediction_res_file_rag):
        answer, snippets, scores = medrag.answer(
            question=question,
            options=options,
            k=rag_k,
            save_dir=save_dir,
            threshold=threshold,
        )
        with open(prediction_res_file_rag, "w") as f:
            json.dump([answer], f, indent=4)
    return 0


def run_rag(
    medrag,
    dataset,
    index,
    prediction_res_folder_cot,
    prediction_res_folder_rag,
    split,
    pbar,
    rag_k,
    threshold=0,
    max_threads=1,
    save_dir=None,
):

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []

        for i in range(len(dataset)):
            ind, question, options = (
                index[i],
                dataset[i]["question"],
                dataset[i]["options"],
            )
            _, prediction_res_file_rag = prepare_prediction_output_files(
                prediction_res_folder_cot, prediction_res_folder_rag, split, ind
            )

            prediction_res_file_name = prediction_res_file_rag.split("/")[-1]
            prediction_res_file_name = prediction_res_file_name.split(".")[0]
            save_dir = os.path.join(prediction_res_folder_rag, prediction_res_file_name)

            future = executor.submit(
                run_rag_single,
                medrag,
                ind,
                question,
                options,
                prediction_res_file_rag,
                rag_k,
                save_dir=save_dir,
                threshold=threshold,
            )
            futures.append(future)

        # Wait for the futures to complete and update the progress bar
        for future in futures:
            try:
                future.result()
            except Exception as e:
                traceback.print_exc()
            pbar.update(1)

    pbar.close()

    return 0


from torch.utils.data import Dataset


class SingleRouterDataset(Dataset):
    def __init__(self, data, labels, retrieval_snippets, scores, soft_label=None):
        self.data = data
        self.labels = labels
        self.retrieval_snippets = retrieval_snippets
        self.scores = scores
        self.soft_label = soft_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        retrieval_snippets = self.retrieval_snippets[index]
        scores = self.scores[index]
        soft_label = self.soft_label[index] if self.soft_label is not None else None
        if soft_label is not None:
            return sample, label, retrieval_snippets, scores, soft_label
        else:
            return sample, label, retrieval_snippets, scores


class RouterDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError("Index out of range")


def get_questions_from_raw_data(dataset_name, data_type="all"):
    assert data_type in [
        "train",
        "eval",
        "all",
    ], "must choose to load train data or eval data"
    train_data_path = "MoG/qa_datasets_rawdata/train_data.json"
    eval_data_path = "MoG/qa_datasets_rawdata/eval_data.json"
    train_questions = []
    eval_questions = []
    with open(train_data_path, "r") as f:
        datas = json.load(f)
    for key in datas[dataset_name].keys():
        question = datas[dataset_name][key]["question"]
        train_questions.append(question)
    with open(eval_data_path, "r") as f:
        datas = json.load(f)
    for key in datas[dataset_name].keys():
        question = datas[dataset_name][key]["question"]
        eval_questions.append(question)
    all_questions = train_questions + eval_questions
    return eval(f"{data_type}_questions")


def question_to_index(retrieval_result_path, dataset_name):
    file_path = os.path.join(
        retrieval_result_path, f"{dataset_name}_retrieval_result.jsonl"
    )
    question2idx = {}
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            json_obj = json.loads(line)
            question = json_obj["question"]
            question2idx[question] = idx

    return question2idx


def load_rawdata(
    medmcqa_path=None,
    bioasq_path=None,
    pubmedqa_path=None,
    medqa_path=None,
    mmlu_path=None,
    retrieval_result_path=None,
    sim_option="",
):

    # medmcqa
    def load_medmcqa_training_data(medmcqa_path, retrieval_result_path):
        data = []
        labels = []
        medmcqa_path = os.path.join(medmcqa_path, "data", "dev.json")
        eval_data_questions = get_questions_from_raw_data("medmcqa", "eval")
        # medmcqa is stored in fact as jsonl format
        for line in open(medmcqa_path):
            line = json.loads(line)
            if line["question"] in eval_data_questions:
                continue
            data.append(line["question"])

            exp = line["exp"] if line["exp"] is not None else ""
            if line["cop"] == 1:
                labels.append(exp + line["opa"])
            elif line["cop"] == 2:
                labels.append(exp + line["opb"])
            elif line["cop"] == 3:
                labels.append(exp + line["opc"])
            elif line["cop"] == 4:
                labels.append(exp + line["opd"])

        retrieval_results = []
        scores = []
        question2idx = question_to_index(retrieval_result_path, "medmcqa")
        with open(
            os.path.join(retrieval_result_path, "medmcqa_retrieval_result.jsonl"), "r"
        ) as file:
            lines = file.readlines()
        for d in data:
            idx = question2idx[d]
            json_obj = json.loads(lines[idx])
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])
        print("medmcqa training data loaded. ")
        print("medmcqa training data num: ", len(data))
        assert len(data) == len(retrieval_results), "data and snippets not match"
        return data, labels, retrieval_results, scores

    # bioasq
    def load_bioasq_training_data(bioasq_path, retrieval_result_path):
        data = []
        labels = []
        # bioasq_path = os.path.join(bioasq_path, "Task7BGoldenEnriched")
        eval_data_questions = get_questions_from_raw_data("bioasq", "eval")
        # get all the json file names under bioasq_path
        # file_names = [f for f in os.listdir(bioasq_path) if f.endswith(".json")]
        file_names = ["all_dataset.json"]
        for file_name in file_names:
            file_path = os.path.join(bioasq_path, file_name)
            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)
                json_data_questions = json_data["questions"]
                for question in json_data_questions:
                    if question["body"] in eval_data_questions:
                        continue
                    data.append(question["body"])
                    context_label = []
                    # context_label should be a list of the strings under snippets.text
                    for snippet in question["snippets"]:
                        context_label.append(snippet["text"])
                    context_label = " ".join(context_label)
                    labels.append(context_label)

        retrieval_results = []
        scores = []
        question2idx = question_to_index(retrieval_result_path, "bioasq")
        with open(
            os.path.join(retrieval_result_path, "bioasq_retrieval_result.jsonl"), "r"
        ) as file:
            lines = file.readlines()
        for d in data:
            idx = question2idx[d]
            json_obj = json.loads(lines[idx])
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])
        print("bioasq training data loaded. ")
        print("bioasq training data num: ", len(data))
        assert len(data) == len(retrieval_results), "data and snippets not match"
        return data, labels, retrieval_results, scores

    # pubmedqa
    def load_pubmedqa_training_data(pubmedqa_path, retrieval_result_path):
        data = []
        labels = []
        pubmedqa_path = os.path.join(pubmedqa_path, "data", "test_set.json")
        eval_data_questions = get_questions_from_raw_data("pubmedqa", "eval")
        with open(pubmedqa_path, "r") as json_file:
            json_data = json.load(json_file)
            for question_id, question_data in json_data.items():
                if question_data["QUESTION"] in eval_data_questions:
                    continue
                data.append(question_data["QUESTION"])
                # labels.append(question_data["CONTEXTS"][0])
                labels.append(".".join(question_data["CONTEXTS"]))

        retrieval_results = []
        scores = []
        question2idx = question_to_index(retrieval_result_path, "pubmedqa")
        with open(
            os.path.join(retrieval_result_path, "pubmedqa_retrieval_result.jsonl"), "r"
        ) as file:
            lines = file.readlines()
        for d in data:
            idx = question2idx[d]
            json_obj = json.loads(lines[idx])
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])
        print("pubmedqa training data loaded. ")
        print("pubmedqa training data num: ", len(data))
        assert len(data) == len(retrieval_results), "data and snippets not match"
        return data, labels, retrieval_results, scores

    # medqa
    def load_medqa_training_data(medqa_path, retrieval_result_path):
        data = []
        labels = []
        medqa_path = os.path.join(
            medqa_path,
            "data_clean",
            "questions",
            "US",
            "4_options",
            "phrases_no_exclude_test.jsonl",
        )
        eval_data_questions = get_questions_from_raw_data("medqa", "eval")
        # medqa is stored in fact as jsonl format
        for line in open(medqa_path):
            line = json.loads(line)
            if line["question"] in eval_data_questions:
                continue
            data.append(line["question"])
            labels.append("Q:" + line["question"] + "; A:" + line["answer"])

        retrieval_results = []
        scores = []
        question2idx = question_to_index(retrieval_result_path, "medqa")
        with open(
            os.path.join(retrieval_result_path, "medqa_retrieval_result.jsonl"), "r"
        ) as file:
            lines = file.readlines()
        for d in data:
            idx = question2idx[d]
            json_obj = json.loads(lines[idx])
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])
        print("medqa training data loaded. ")
        print("medqa training data num: ", len(data))
        assert len(data) == len(retrieval_results), "data and snippets not match"
        return data, labels, retrieval_results, scores

    # mmlu
    def load_mmlu_training_data(mmlu_path, retrieval_result_path):
        data = []
        labels = []
        mmlu_path = os.path.join(mmlu_path, "data", "dev.json")
        eval_data_questions = get_questions_from_raw_data("mmlu", "eval")
        # mmlu is stored in fact as jsonl format
        with open(mmlu_path, "r") as json_file:
            json_data = json.load(json_file)
            for question_id, question_data in json_data.items():
                if question_data["question"] in eval_data_questions:
                    continue
                data.append(question_data["question"])
                options = question_data["options"]
                answer = question_data["answer"]
                labels.append("Q:" + question_data["question"] + "; A:" + options[answer])

        retrieval_results = []
        scores = []
        question2idx = question_to_index(retrieval_result_path, "mmlu")
        with open(
            os.path.join(retrieval_result_path, "mmlu_retrieval_result.jsonl"), "r"
        ) as file:
            lines = file.readlines()
        for d in data:
            idx = question2idx[d]
            json_obj = json.loads(lines[idx])
            retrieval_results.append(json_obj["retrieved_snippets"])
            scores.append(json_obj["scores"])
        print("mmlu training data loaded. ")
        print("mmlu training data num: ", len(data))
        assert len(data) == len(retrieval_results), "data and snippets not match"
        return data, labels, retrieval_results, scores

    def load_from_cache(cache_file_path):
        data, labels, retrieval_results, scores, soft_labels = [], [], [], [], []
        with open(cache_file_path, "r") as file:
            cache_data = json.load(file)
        for d in cache_data:
            data.append(d["question"])
            labels.append(d["labels"])
            retrieval_results.append(d["retrieved_snippets"])
            scores.append(d["scores"])
            soft_labels.append(d["soft_label"])

        return data, labels, retrieval_results, scores, soft_labels

    print("[In progress] Loading training data... ")
    dataset_totrain = []
    if medmcqa_path is not None:
        dataset_totrain.append("medmcqa")
    if bioasq_path is not None:
        dataset_totrain.append("bioasq")
    if pubmedqa_path is not None:
        dataset_totrain.append("pubmedqa")
    if medqa_path is not None:
        dataset_totrain.append("medqa")
    if mmlu_path is not None:
        dataset_totrain.append("mmlu")
    cache_file_path = os.path.join(
        retrieval_result_path, f"cache_soft_labels_{sim_option}_{dataset_totrain}.json"
    )

    if os.path.exists(cache_file_path):
        print(f"Loading training data from cache at {cache_file_path}...")
        dataset_list = []
        data, labels, retrieval_results, scores, soft_labels = load_from_cache(
            cache_file_path
        )
        dataset_list.append(
            SingleRouterDataset(data, labels, retrieval_results, scores, soft_labels)
        )
        router_dataset = RouterDataset(dataset_list)
        return router_dataset

    else:
        print("No cache file found. Loading training data from scratch.")
        datasets_to_load = []
        if medmcqa_path is not None:
            datasets_to_load.append("medmcqa")
        if bioasq_path is not None:
            datasets_to_load.append("bioasq")
        if pubmedqa_path is not None:
            datasets_to_load.append("pubmedqa")
        if medqa_path is not None:
            datasets_to_load.append("medqa")
        if mmlu_path is not None:
            datasets_to_load.append("mmlu")

        print("Datasets to load: ", datasets_to_load)

        print("[Done] Retriever initialized.")
        dataset_list = []
        if medmcqa_path is not None:
            medmcqa_data, medmcqa_labels, medmcqa_retrieval_snippets, medmcqa_scores = (
                load_medmcqa_training_data(medmcqa_path, retrieval_result_path)
            )
            dataset_list.append(
                SingleRouterDataset(
                    medmcqa_data,
                    medmcqa_labels,
                    medmcqa_retrieval_snippets,
                    medmcqa_scores,
                )
            )

        if bioasq_path is not None:
            bioasq_data, bioasq_labels, bioasq_retrieval_snippets, bioasq_scores = (
                load_bioasq_training_data(bioasq_path, retrieval_result_path)
            )
            dataset_list.append(
                SingleRouterDataset(
                    bioasq_data, bioasq_labels, bioasq_retrieval_snippets, bioasq_scores
                )
            )

        if pubmedqa_path is not None:
            (
                pubmedqa_data,
                pubmedqa_labels,
                pubmedqa_retrieval_snippets,
                pubmedqa_scores,
            ) = load_pubmedqa_training_data(pubmedqa_path, retrieval_result_path)
            dataset_list.append(
                SingleRouterDataset(
                    pubmedqa_data,
                    pubmedqa_labels,
                    pubmedqa_retrieval_snippets,
                    pubmedqa_scores,
                )
            )

        if medqa_path is not None:
            medqa_data, medqa_labels, medqa_retrieval_snippets, medqa_scores = (
                load_medqa_training_data(medqa_path, retrieval_result_path)
            )
            dataset_list.append(
                SingleRouterDataset(
                    medqa_data, medqa_labels, medqa_retrieval_snippets, medqa_scores
                )
            )

        if mmlu_path is not None:
            mmlu_data, mmlu_labels, mmlu_retrieval_snippets, mmlu_scores = (
                load_mmlu_training_data(mmlu_path, retrieval_result_path)
            )
            dataset_list.append(
                SingleRouterDataset(
                    mmlu_data, mmlu_labels, mmlu_retrieval_snippets, mmlu_scores
                )
            )

        router_dataset = RouterDataset(dataset_list)
        print("[Done] All training data loaded and merged as one dataset. \n")

    return router_dataset


def determine_checkpoint_folder(
    exp_counter_file, router_checkpoint_path, exp_option=None
):
    # determin the checkpoint_id
    # read the current experiment counter, there is only a number in this txt file

    checkpoint_id = int(exp_option.split("exp")[1])

    # format current date
    current_date = date.today().strftime("%Y%m%d")

    # Create the checkpoint folder name
    checkpoint_folder_name = f"{checkpoint_id}_{current_date}"

    # Create the checkpoint folder if not exist
    checkpoint_folder_path = os.path.join(
        router_checkpoint_path, checkpoint_folder_name
    )
    if not os.path.exists(checkpoint_folder_path):
        os.makedirs(checkpoint_folder_path)

    return checkpoint_folder_path, checkpoint_folder_name


def determine_checkpoint_path(checkpoint_folder_path, loss_value, epoch_num):
    # Format the final loss
    loss_value_formatted = "{:.4f}".format(loss_value)

    # Create the checkpoint file name
    checkpoint_file_name = f"Epoch_{epoch_num}_Loss_{loss_value_formatted}.pt"

    # Create the checkpoint file path
    checkpoint_file_path = os.path.join(checkpoint_folder_path, checkpoint_file_name)

    return checkpoint_file_path


def build_snippet_score_dict(retrieved_snippets, scores, weights, device):
    snippet_score_list_dict = {}
    for retriever_idx in range(len(retrieved_snippets)):
        retriever_result = retrieved_snippets[retriever_idx]
        for copora_idx in range(len(retriever_result)):
            copora_result = retriever_result[copora_idx]
            for snippet_idx in range(len(copora_result)):
                snippet = copora_result[snippet_idx]
                if snippet != "NO_TEXT_RETRIEVED":
                    snippet_id_list = snippet["id"].split("#")[1].split("|")
                    for snippet_id in snippet_id_list:
                        if snippet_id not in snippet_score_list_dict:
                            score_list = [0] * len(retriever_result)
                        else:
                            # already exist, modify the score_list
                            score_list = snippet_score_list_dict[snippet_id]

                        score_list[copora_idx] = scores[retriever_idx][copora_idx][
                            snippet_idx
                        ]
                        snippet_score_list_dict[snippet_id] = score_list

    snippet_score_dict = {}
    for snippet in snippet_score_list_dict:
        weighted_score = torch.dot(
            weights,
            torch.tensor(snippet_score_list_dict[snippet], dtype=torch.float32).to(
                device
            ),
        )

        weighted_score = weighted_score
        snippet_score_dict[snippet] = weighted_score

    return snippet_score_list_dict, snippet_score_dict


def weighted_merge_snippets(
    snippet_input,
    score,
    snippet_score_list_dict,
    snippet_score_dict,
    router_merge_k,
    weights,
    return_separate_list=False,
):

    top_k_ids = sorted(snippet_score_dict, key=snippet_score_dict.get, reverse=True)[
        :router_merge_k
    ]

    # get the optimal granularity level
    weights_sorted = torch.argsort(weights, descending=True)
    sorted_weights_index = weights_sorted.tolist()
    optimal_gra_lvl_list = [0] * len(top_k_ids)
    for i in range(len(top_k_ids)):
        top_id = top_k_ids[i]
        score_list = snippet_score_list_dict[top_id]

        def find_first_nonzero_index(score_list, sorted_indices_list):
            for index in sorted_indices_list:
                if score_list[index] != 0:
                    return index
            return None  # Return None if no non-zero value is found

        optimal_gra_lvl_list[i] = find_first_nonzero_index(
            score_list, sorted_weights_index
        )

    snippet_merged = []
    snippet_merged_list = []
    for j in range(len(optimal_gra_lvl_list)):
        gra_lvl = optimal_gra_lvl_list[j]
        snippet_id = top_k_ids[j]

        retriever_result = snippet_input[0]
        # in the setting of router_moe there is only 1 retriever

        copora_result = retriever_result[gra_lvl]
        for snippet_idx in range(len(copora_result)):
            snippet = copora_result[snippet_idx]
            if snippet_id in snippet["id"] and str(snippet) not in snippet_merged:
                snippet_merged.append(str(snippet))
            if snippet_id in snippet["id"] and snippet not in snippet_merged_list:
                snippet_merged_list.append(snippet)

    if return_separate_list:
        return snippet_merged_list
    else:
        return "".join(snippet_merged)


def collate_fn(batch):
    soft_label_flag = True
    try:
        samples, labels, snippets, scores, soft_labels = zip(*batch)
    except Exception as e:
        samples, labels, snippets, scores = zip(*batch)
        soft_label_flag = False

    # Determine the maximum length among samples in the batch
    max_sample_length = max(len(sample) for sample in samples)

    # Pad shorter samples within the batch to match the maximum length
    padded_samples = [sample.ljust(max_sample_length) for sample in samples]

    if soft_label_flag:
        return (
            list(padded_samples),
            list(labels),
            list(snippets),
            list(scores),
            list(soft_labels),
        )
    else:
        return list(padded_samples), list(labels), list(snippets), list(scores)


# Functions for multi-threading in retrieval
def run_single_retriever(
    retriever,
    question_idx,
    question,
    rag_k,
    result_lock,
    result_path,
    threshold=0,
):
    retrieved_snippets, scores = retriever.answer(
        question=question, k=rag_k, threshold=threshold
    )
    result_lock.acquire()
    try:
        with open(result_path, "a") as f:
            json.dump(
                {
                    "question_idx": question_idx,
                    "question": question,
                    "retrieved_snippets": retrieved_snippets,
                    "scores": scores,
                },
                f,
            )
            f.write("\n")
    finally:
        result_lock.release()


def retrieve_with_thread(
    retriever, questions, rag_k, max_threads=32, result_path=None, threshold=0
):
    result_lock = threading.Lock()
    # explicitly initialize the result_path file if not exists
    if not os.path.exists(result_path):
        with open(result_path, "w+") as f:
            pass

    # Collect the question_idx that exists already in the result_path file, to avoid re-retrieving
    question_idx_set = set()
    with open(result_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            question_idx_set.add(json_obj["question_idx"])

    if threshold == 0:
        raise ValueError("threshold cannot be zero")
    else:
        print(f"[In progress] running with threshold [{threshold}] for retrieve...")

    # Create a progress bar
    pbar = tqdm.tqdm(total=len(questions), desc="Retrieving snippets")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []

        # Retrieve with threads
        for question_idx, question in enumerate(questions):
            if question_idx in question_idx_set:
                pbar.update(1)
                continue
            # print(question)
            future = executor.submit(
                run_single_retriever,
                retriever,
                question_idx,
                question,
                rag_k,
                result_lock,
                result_path,
                threshold,
            )
            futures.append(future)

        # Wait for the futures to complete and update the progress bar
        for future in futures:
            future.result()
            pbar.update(1)

    pbar.close()

    return 0


def run_single_weight_merge_calculate_loss(
    weights,
    snippet,
    score,
    question_idx,
    result_lock,
    results,
    device,
    router_merge_k,
    loss_function,
    labels,
    pbar_router_training_sample,
    cache_train=False,
    cache_dict=None,
):

    snippet_score_list_dict, snippet_score_dict = build_snippet_score_dict(
        snippet, score, weights, device
    )

    if cache_train:
        assert cache_dict is not None, "cache_dict not set, can't use cache_train"

        snippet_merged_list = weighted_merge_snippets(
            snippet,
            score,
            snippet_score_list_dict,
            snippet_score_dict,
            router_merge_k,
            weights,
            return_separate_list=True,
        )

        # get a set of the snippet_id in snippet_merged
        snip_id_list = []
        for snip in snippet_merged_list:
            snip_id_list.append(snip["id"])
        snip_id_list.sort()
        id_string = "".join(snip_id_list)

        result_lock.acquire()
        try:
            # retrieve the loss value from cache if exists
            if id_string in cache_dict:
                loss = cache_dict[id_string]
            else:
                # if not exists, compute the loss
                snippet_merged = [str(s) for s in snippet_merged_list]
                snippet_merged = "".join(snippet_merged)
                loss = loss_function(snippet_merged, labels[question_idx])
                cache_dict[id_string] = loss
            pbar_router_training_sample.update(1)
        finally:
            result_lock.release()
    else:
        snippet_merged = weighted_merge_snippets(
            snippet,
            score,
            snippet_score_list_dict,
            snippet_score_dict,
            router_merge_k,
            weights,
        )
        # no cache, calculate each time
        loss = loss_function(snippet_merged, labels[question_idx])
        pbar_router_training_sample.update(1)

    result_lock.acquire()
    try:
        results[question_idx] = loss
    finally:
        result_lock.release()


def weight_merge_calculate_loss_with_thread(
    weights,
    snippets,
    scores,
    labels,
    pbar_router_training_sample,
    device,
    router_merge_k,
    loss_function,
    cache_train=None,
    cache_dict=None,
):
    threads = []
    results = {}
    result_lock = threading.Lock()

    # retrieve with threads
    for question_idx, snippet in enumerate(snippets):
        thread = threading.Thread(
            target=run_single_weight_merge_calculate_loss,
            args=(
                weights[question_idx],
                snippet,
                scores[question_idx],
                question_idx,
                result_lock,
                results,
                device,
                router_merge_k,
                loss_function,
                labels,
                pbar_router_training_sample,
                cache_train,
                cache_dict,
            ),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        # wait for the thread to finish

    # order the items in results according to its original order in 'questions'
    ordered_loss = [
        results[question_idx] for question_idx, snippet in enumerate(snippets)
    ]

    return ordered_loss

def retrieve_and_cache_with_thread(
    medmcqa_path=None,
    bioasq_path=None,
    pubmedqa_path=None,
    medqa_path=None,
    mmlu_path=None,
    retrieval_result_path=None,
    retriever=None,
    rag_k=None,
    thresholds_dict={},
):
    assert retriever is not None, "no retriever provided, cant retrieve, cant cache"
    assert rag_k is not None, "rag_k not set"

    # medmcqa
    def cache_medmcqa_retrieval_results(
        medmcqa_path, retriever, retrieval_result_path, threshold=0
    ):
        result_path = os.path.join(
            retrieval_result_path, "medmcqa_retrieval_result.jsonl"
        )
        medmcqa_path = os.path.join(medmcqa_path, "data", "dev.json")
        # check if the cache files already exist
        if os.path.exists(result_path):
            # check the number of lines in the cache file
            # check if it is the same as the lines in the original file
            # if yes, skip the caching
            # if not, re-cache
            original_line_count = 0
            for line in open(medmcqa_path):
                original_line_count += 1

            cache_line_count = 0
            with open(
                os.path.join(retrieval_result_path, "medmcqa_retrieval_result.jsonl")
            ) as file:
                for line in file:
                    cache_line_count += 1

            if original_line_count == cache_line_count:
                return 0

        # cache the retrieval results
        questions = []
        for line in open(medmcqa_path):
            line = json.loads(line)
            questions.append(line["question"])

        _ = retrieve_with_thread(
            retriever,
            questions,
            rag_k=rag_k,
            result_path=result_path,
            threshold=threshold,
        )

        # Read the list of json in result_path
        json_list = []
        with open(result_path, "r") as file:
            for obj in file:
                json_list.append(json.loads(obj))

        # Reorder the json_list according to the question_idx field, ASC
        json_list = sorted(json_list, key=lambda x: x["question_idx"])

        # Write the ordered json_list back to the result_path
        with open(result_path, "w") as file:
            for obj in json_list:
                json.dump(obj, file)
                file.write("\n")
        return 0

    # bioasq [need to modify]
    def cache_bioasq_retrieval_results(
        bioasq_path, retriever, retrieval_result_path, threshold=0
    ):
        result_path = os.path.join(
            retrieval_result_path, "bioasq_retrieval_result.jsonl"
        )
        bioasq_path = os.path.join(bioasq_path, "Task7BGoldenEnriched")
        questions = get_questions_from_raw_data("bioasq")
        # check if the cache files already exist
        if os.path.exists(result_path):
            # check the number of lines in the cache file
            # check if it is the same as the lines in the original file
            # if yes, skip the caching
            # if not, re-cache
            original_line_count = len(questions)

            cache_line_count = 0
            with open(
                os.path.join(retrieval_result_path, "bioasq_retrieval_result.jsonl")
            ) as file:
                for line in file:
                    cache_line_count += 1

            if original_line_count == cache_line_count:
                return 0

        # cache the retrieval results

        _ = retrieve_with_thread(
            retriever,
            questions,
            rag_k=rag_k,
            result_path=result_path,
            threshold=threshold,
        )

        # Read the list of json in result_path
        json_list = []
        with open(result_path, "r") as file:
            for obj in file:
                json_list.append(json.loads(obj))

        # Reorder the json_list according to the question_idx field, ASC
        json_list = sorted(json_list, key=lambda x: x["question_idx"])

        # Write the ordered json_list back to the result_path
        with open(result_path, "w") as file:
            for obj in json_list:
                json.dump(obj, file)
                file.write("\n")
        return 0

    # pubmedqa
    def cache_pubmedqa_retrieval_results(
        pubmedqa_path, retriever, retrieval_result_path, threshold=0
    ):
        result_path = os.path.join(
            retrieval_result_path, "pubmedqa_retrieval_result.jsonl"
        )
        pubmedqa_path = os.path.join(pubmedqa_path, "data", "test_set.json")
        questions = get_questions_from_raw_data("pubmedqa")
        # check if the cache files already exist
        if os.path.exists(result_path):
            # check the number of lines in the cache file
            # check if it is the same as the lines in the original file
            # if yes, skip the caching
            # if not, re-cache

            original_line_count = len(questions)

            cache_line_count = 0
            with open(
                os.path.join(retrieval_result_path, "pubmedqa_retrieval_result.jsonl")
            ) as file:
                for line in file:
                    cache_line_count += 1

            if original_line_count == cache_line_count:
                return 0

        # cache the retrieval results

        _ = retrieve_with_thread(
            retriever,
            questions,
            rag_k=rag_k,
            result_path=result_path,
            threshold=threshold,
        )

        # Read the list of json in result_path
        json_list = []
        with open(result_path, "r") as file:
            for obj in file:
                json_list.append(json.loads(obj))

        # Reorder the json_list according to the question_idx field, ASC
        json_list = sorted(json_list, key=lambda x: x["question_idx"])

        # Write the ordered json_list back to the result_path
        with open(result_path, "w") as file:
            for obj in json_list:
                json.dump(obj, file)
                file.write("\n")
        return 0

    # medqa
    def cache_medqa_retrieval_results(
        medqa_path, retriever, retrieval_result_path, threshold=0
    ):
        result_path = os.path.join(
            retrieval_result_path, "medqa_retrieval_result.jsonl"
        )
        medqa_path = os.path.join(
            medqa_path,
            "data_clean",
            "questions",
            "US",
            "4_options",
            "phrases_no_exclude_test.jsonl",
        )
        # check if the cache files already exist
        if os.path.exists(result_path):
            # check the number of lines in the cache file
            # check if it is the same as the lines in the original file
            # if yes, skip the caching
            # if not, re-cache
            original_line_count = 0
            for line in open(medqa_path):
                original_line_count += 1

            cache_line_count = 0
            with open(
                os.path.join(retrieval_result_path, "medqa_retrieval_result.jsonl")
            ) as file:
                for line in file:
                    cache_line_count += 1

            if original_line_count == cache_line_count:
                return 0

        # cache the retrieval results
        questions = []
        for line in open(medqa_path):
            line = json.loads(line)
            questions.append(line["question"])

        _ = retrieve_with_thread(
            retriever,
            questions,
            rag_k=rag_k,
            result_path=result_path,
            threshold=threshold,
        )

        # Read the list of json in result_path
        json_list = []
        with open(result_path, "r") as file:
            for obj in file:
                json_list.append(json.loads(obj))

        # Reorder the json_list according to the question_idx field, ASC
        json_list = sorted(json_list, key=lambda x: x["question_idx"])

        # Write the ordered json_list back to the result_path
        with open(result_path, "w") as file:
            for obj in json_list:
                json.dump(obj, file)
                file.write("\n")
        return 0

    # mmlu
    def cache_mmlu_retrieval_results(
        mmlu_path, retriever, retrieval_result_path, threshold=0
    ):
        result_path = os.path.join(retrieval_result_path, "mmlu_retrieval_result.jsonl")
        mmlu_path = os.path.join(mmlu_path, "data", "dev.json")
        # check if the cache files already exist
        questions = get_questions_from_raw_data("mmlu")
        if os.path.exists(result_path):
            # check the number of lines in the cache file
            # check if it is the same as the lines in the original file
            # if yes, skip the caching
            # if not, re-cache
            original_line_count = len(questions)

            cache_line_count = 0
            with open(
                os.path.join(retrieval_result_path, "mmlu_retrieval_result.jsonl")
            ) as file:
                for line in file:
                    cache_line_count += 1

            if original_line_count == cache_line_count:
                return 0

        # cache the retrieval results

        _ = retrieve_with_thread(
            retriever,
            questions,
            rag_k=rag_k,
            result_path=result_path,
            threshold=threshold,
        )

        # Read the list of json in result_path
        json_list = []
        with open(result_path, "r") as file:
            for obj in file:
                json_list.append(json.loads(obj))

        # Reorder the json_list according to the question_idx field, ASC
        json_list = sorted(json_list, key=lambda x: x["question_idx"])

        # Write the ordered json_list back to the result_path
        with open(result_path, "w") as file:
            for obj in json_list:
                json.dump(obj, file)
                file.write("\n")
        return 0

    print("[In progress] Caching retrieval results for training data... ")
    datasets_to_cache = []
    if medmcqa_path is not None:
        datasets_to_cache.append("medmcqa")
    if bioasq_path is not None:
        datasets_to_cache.append("bioasq")
    if pubmedqa_path is not None:
        datasets_to_cache.append("pubmedqa")
    if medqa_path is not None:
        datasets_to_cache.append("medqa")
    if mmlu_path is not None:
        datasets_to_cache.append("mmlu")

    print("Datasets to cache: ", datasets_to_cache)
    print(f"Retrieval results will be cached in: {retrieval_result_path}")

    if medmcqa_path is not None:
        _ = cache_medmcqa_retrieval_results(
            medmcqa_path,
            retriever,
            retrieval_result_path,
            threshold=thresholds_dict["medmcqa"],
        )
    if bioasq_path is not None:
        _ = cache_bioasq_retrieval_results(
            bioasq_path,
            retriever,
            retrieval_result_path,
            threshold=thresholds_dict["bioasq"],
        )
    if pubmedqa_path is not None:
        _ = cache_pubmedqa_retrieval_results(
            pubmedqa_path,
            retriever,
            retrieval_result_path,
            threshold=thresholds_dict["pubmedqa"],
        )
    if medqa_path is not None:
        _ = cache_medqa_retrieval_results(
            medqa_path,
            retriever,
            retrieval_result_path,
            threshold=thresholds_dict["medqa"],
        )
    if mmlu_path is not None:
        _ = cache_mmlu_retrieval_results(
            mmlu_path,
            retriever,
            retrieval_result_path,
            threshold=thresholds_dict["mmlu"],
        )

    print("[Done] Retrieval results cached.")
