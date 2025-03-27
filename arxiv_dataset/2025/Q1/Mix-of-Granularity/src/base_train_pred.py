"""
Predict with CoT and RAG using InternLM / ChatGLM3 / OpenAI API to serve as the baseline for further researches.
Merge from the previous separated scripts.

Zijie 3 April 2024

------------------------------------------------------------------
Simplified version, removing all the features proven to be less efficient or unnecessary

Zijie 10 May 2024

"""

# Import libraries
import json
import os
import sys
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import BCELoss

from config import config
from tqdm import tqdm
import json

# Use mirror site of huggingface
os.environ["HF_ENDPOINT"] = config["hf_endpoint"]
os.environ["HF_HUB_URL"] = config["hf_hub_url"]

# -----------------------------------------------
# DEFINE THE ARGUMENTS
# -----------------------------------------------
parser = argparse.ArgumentParser(description="Prediction_job_parser")
parser.add_argument("--llm", type=str, default="internlm", choices=["gpt", "internlm", "glm", "qwen", "llama3"])
parser.add_argument("--retriever_name", type=str, default="BM25")
parser.add_argument("--corpus_name", type=str, default="Textbooks")
parser.add_argument(
    "--train_qa_dataset_name",
    type=str,
    default="medmcqa_bioasq_pubmedqa_medqa_mmlu",
    help="the name of the dataset to train the router",
)
parser.add_argument(
    "--retrieve_qa_dataset_name",
    type=str,
    default="medmcqa_bioasq_pubmedqa_medqa_mmlu",
    help="the name of the dataset to retrieve",
)
parser.add_argument(
    "--pred_qa_dataset_name",
    type=str,
    default="medmcqa_bioasq_pubmedqa_medqa_mmlu",
    help="the name of the dataset to predict",
)
parser.add_argument(
    "--corpus_folder",
    type=str,
    default="../corpus/corpus_mog",
    help="path to the folder of corpus, also known as db_dir",
)
parser.add_argument(
    "--checkpoint_load_path",
    type=str,
    default=None,
    help="the path to the checkpoint file of router",
)
parser.add_argument("--retrieval_result_path", type=str, default=None)

parser.add_argument(
    "--sim_option",
    type=str,
    default="bce_roberta",
    help="set the similarity option to build soft-labels",
)

parser.add_argument("--rag_k", type=int, default=3)
parser.add_argument("--rrf_k", type=int, default=100)
parser.add_argument("--granularity_n", type=int, default=5)
parser.add_argument("--exp_option", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--router_epoch", type=int, default=400)
parser.add_argument(
    "--router_checkpoint_steps",
    type=int,
    default=50,
    help="save the router model every n steps",
)
parser.add_argument("--router_lr", type=float, default=0.001)
#parser.add_argument("--threshold", type=float, default=50)
parser.add_argument(
    "--router_merge_k",
    type=int,
    default=2,
    help="number of documents to keep when merging with the router",
)

parser.add_argument(
    "--do_train_router",
    action="store_true",
    default=False,
    help="set to enable training the router",
)
parser.add_argument(
    "--do_cot", action="store_true", default=False, help="set to inference with cot"
)
parser.add_argument(
    "--do_rag", action="store_true", default=False, help="set to inference with rag"
)
parser.add_argument(
    "--do_single_rag", action="store_true", default=False, help="set to inference with single rag"
)
parser.add_argument(
    "--pred_with_router",
    action="store_true",
    default=False,
    help="set to enable router when predicting with RAG",
)
parser.add_argument(
    "--clear_previous_eval_results",
    action="store_true",
    default=False,
    help="set to clear the previous prediction results before storing the new prediction results",
)


args = parser.parse_args()

# Read the parameters (arguments) defined in the parser
(
    llm,
    retriever_name,
    corpus_name,
    rag_k,
    clear_previous_eval_results,
    do_cot,
    do_rag,
    do_single_rag,
    rrf_k,
    pred_with_router,
    do_train_router,
    batch_size,
    granularity_n,
    router_lr,
    router_epoch,
    router_checkpoint_steps,
    router_merge_k,
    retrieve_qa_dataset_name,
    train_qa_dataset_name,
    pred_qa_dataset_name,
    checkpoint_load_path,
    exp_option,
#   threshold,
    sim_option,
) = (
    args.llm,
    args.retriever_name,
    args.corpus_name,
    args.rag_k,
    args.clear_previous_eval_results,
    args.do_cot,
    args.do_rag,
    args.do_single_rag,
    args.rrf_k,
    args.pred_with_router,
    args.do_train_router,
    args.batch_size,
    args.granularity_n,
    args.router_lr,
    args.router_epoch,
    args.router_checkpoint_steps,
    args.router_merge_k,
    args.retrieve_qa_dataset_name,
    args.train_qa_dataset_name,
    args.pred_qa_dataset_name,
    args.checkpoint_load_path,
    args.exp_option,
#   args.threshold,
    args.sim_option,
)

# Read the paths defined in the config file
(
    db_dir,
    prediction_folder,
    cache_dir,
    benchmark_repo_dir,
    benchmark_dataset_json,
    medmcqa_path,
    bioasq_path,
    pubmedqa_path,
    medqa_path,
    mmlu_path,
    tensorboard_log_dir,
    router_checkpoint_path,
    retrieval_result_path,
    exp_counter_file,
    medrag_path,
) = (
    config["db_dir"],
    config["prediction_folder"],
    config["cache_dir"],
    config["benchmark_repo_dir"],
    config["benchmark_dataset_json"],
    config["medmcqa_path"],
    config["bioasq_path"],
    config["pubmedqa_path"],
    config["medqa_path"],
    config["mmlu_path"],
    config["tensorboard_log_dir"],
    config["router_checkpoint_path"],
    config["retrieval_result_path"],
    config["exp_counter_file"],
    config["medrag_path"],
)

# import from 'src' and 'utils' requires medrag_path in sys.path
sys.path.insert(0, medrag_path)
from src.medrag import MedRAG
from src.moe import Router, simLoss, softLabel
from utils import (
    set_split,
    prepare_prediction_output_folders,
    load_benchmark_dataset,
    run_cot,
    run_rag,
    load_rawdata,
    determine_checkpoint_folder,
    determine_checkpoint_path,
    collate_fn,
    retrieve_and_cache_with_thread,
)
# set the threshold for benchmarks
thresholds_dict = {
    "medmcqa":5,
    "bioasq":6,
    "pubmedqa":8,
    "medqa":50,
    "mmlu":10
}

retrieve_medmcqa_path = None if "medmcqa" not in retrieve_qa_dataset_name.lower() else medmcqa_path
retrieve_bioasq_path = None if "bioasq" not in retrieve_qa_dataset_name.lower() else bioasq_path
retrieve_pubmedqa_path = (
    None if "pubmed" not in retrieve_qa_dataset_name.lower() else pubmedqa_path
)
retrieve_medqa_path = None if "medqa" not in retrieve_qa_dataset_name.lower() else medqa_path
retrieve_mmlu_path = None if "mmlu" not in retrieve_qa_dataset_name.lower() else mmlu_path



# set the llm_name according to llm
llm2llm_name = {
    "internlm": "internlm",
    "gpt": "OpenAI/gpt-3.5-turbo-0125",
    "glm": "glm",
    "qwen": "qwen_moe",
    "llama3": "llama3",
}
if llm in llm2llm_name:
    llm_name = llm2llm_name[llm]
else:
    print("llm argument is not valid, please check the llm argument")

# Overwrite some arguments when provided
if args.corpus_folder is not None:
    db_dir = args.corpus_folder
if exp_option is not None:
    prediction_folder = os.path.join(prediction_folder, exp_option)
else:
    prediction_folder = os.path.join(prediction_folder, "exp_option_not_specified")
if args.retrieval_result_path is not None:
    retrieval_result_path = args.retrieval_result_path
retrieval_result_path = (
    retrieval_result_path + f"rag_{rag_k}_{db_dir.split('/')[-1]}_{retriever_name}/"
)
if not os.path.exists(retrieval_result_path):
    os.makedirs(retrieval_result_path)

# If a dataset's name is not mentioned, set its corresponding path to None
medmcqa_path = None if "medmcqa" not in train_qa_dataset_name.lower() else medmcqa_path
bioasq_path = None if "bioasq" not in train_qa_dataset_name.lower() else bioasq_path
pubmedqa_path = (
    None if "pubmed" not in train_qa_dataset_name.lower() else pubmedqa_path
)
medqa_path = None if "medqa" not in train_qa_dataset_name.lower() else medqa_path
mmlu_path = None if "mmlu" not in train_qa_dataset_name.lower() else mmlu_path

# define the cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm_local = None

# -----------------------------------------------
# TRAIN THE ROUTER
# -----------------------------------------------
checkpoint_path_use = None
# Instantiate the Router model
router = Router(
    query="sample query",
    output_dim=granularity_n,
    device=device,
    exp_option=args.exp_option,
)
if do_train_router:
    # Set checkpoint folder, grouping all the checkpoints of this training experiment
    checkpoint_folder_path, checkpoint_folder_name = determine_checkpoint_folder(
        exp_counter_file, router_checkpoint_path, exp_option
    )
    print(
        f"------------------------------\nThe checkpoints of the router will be saved in\n{checkpoint_folder_path}\n------------------------------\n"
    )

    # Initialize the retriever
    medrag_only_retrieve = MedRAG(
        llm_name=llm_name,
        rag=True,
        retriever_name=retriever_name,
        corpus_name=corpus_name,
        cache_dir=cache_dir,
        db_dir=db_dir,
        llm_local=llm_local,
        rrf_k=rrf_k,
        only_retrieve=True,
        router_merge_k=router_merge_k,
#       threshold=threshold,
    )

    retriever = medrag_only_retrieve

    # Cache the retrieval results to speed up the training if flagged
    _ = retrieve_and_cache_with_thread(
        medmcqa_path=retrieve_medmcqa_path,
        bioasq_path=retrieve_bioasq_path,
        pubmedqa_path=retrieve_pubmedqa_path,
        medqa_path=retrieve_medqa_path,
        mmlu_path=retrieve_mmlu_path,
        retrieval_result_path=retrieval_result_path,
        retriever=retriever,
        rag_k=rag_k,
        thresholds_dict=thresholds_dict
)

    # Build the soft-labels for training
    slb = softLabel(retrieval_result_path=retrieval_result_path, sim_option=sim_option)
    slb.build(
        medmcqa_path=medmcqa_path,
        bioasq_path=bioasq_path,
        pubmedqa_path=pubmedqa_path,
        medqa_path=medqa_path,
        mmlu_path=mmlu_path,
    )

    # Load the data for training the Router


    router_dataset = load_rawdata(
        medmcqa_path=medmcqa_path,
        bioasq_path=bioasq_path,
        pubmedqa_path=pubmedqa_path,
        medqa_path=medqa_path,
        mmlu_path=mmlu_path,
        retrieval_result_path=retrieval_result_path,
        sim_option=sim_option,
    )

    # Define the loss function and optimizer
    loss_function = BCELoss()
    optimizer = Adam(router.mlp.parameters(), lr=router_lr)
    # Set the Router model to train model
    router.mlp.train()
    router.mlp.to(device)
    print("[Done] Router model initialized.")

    # Create a writer for TensorBoard
    writer = SummaryWriter(
        log_dir=os.path.join(tensorboard_log_dir, checkpoint_folder_name)
    )

    print("\n[In progress] Training the Router model...")
    train_loss_cache_dict = {}
    for epoch in tqdm(range(router_epoch), desc="Router training [Epochs]"):
        # Create a data loader
        data_loader = DataLoader(
            router_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        pbar_router_training_sample = tqdm(
            total=len(router_dataset), desc="Router training [Samples]", leave=False
        )
        epoch_loss, loss_counter = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            questions, labels, snippets, scores, *soft_labels = (
                batch if len(batch) == 5 else (*batch, None)
            )

            # Forward pass
            weights = router.run(questions)  # [batch_size, granularity_n]

            # Calculate the loss
            optimizer.zero_grad()
            soft_labels = (
                torch.tensor(soft_labels, dtype=torch.float32).to(device).squeeze(0)
            )
            loss = loss_function(weights, soft_labels)
            loss.backward()
            optimizer.step()

            # Record the loss
            epoch_loss += loss.item()
            loss_counter += 1
            pbar_router_training_sample.update(len(questions))

        # Write the loss to TensorBoard at the end of each epoch
        writer.add_scalar("Loss/Training", epoch_loss / loss_counter, epoch)

        # Save a checkpoint of the model every n steps
        if (epoch + 1) % router_checkpoint_steps == 0 or epoch + 1 == router_epoch:
            checkpoint_path = determine_checkpoint_path(
                checkpoint_folder_path=checkpoint_folder_path,
                loss_value=epoch_loss / loss_counter,
                epoch_num=epoch,
            )
            torch.save(router.state_dict(), checkpoint_path)
            checkpoint_path_use = checkpoint_path
    print(
        f"\n[Done] Router model trained.\nCheckpoints save in: \n{checkpoint_folder_path}\nCheckpoint file use to load the model for prediction: \n{checkpoint_path_use}"
    )

else:
    assert checkpoint_load_path is not None, "checkpoint_load_path is not specified"
    checkpoint_path_use = checkpoint_load_path

# -----------------------------------------------
# EVALUATION
# -----------------------------------------------

# load the model from checkpoint
if not do_single_rag:
    router.load_state_dict(torch.load(checkpoint_path_use))
    router.eval()
    print(f"Loaded router model from: {checkpoint_path_use}\n")

# Load the benchmark datasets
benchmark = json.load(open(benchmark_dataset_json))
if pred_qa_dataset_name != "medmcqa_bioasq_pubmedqa_medqa_mmlu":
    benchmark = {k: v for k, v in benchmark.items() if k in args.pred_qa_dataset_name}
print(
    f"Benchmark dataset loaded, the subdataset used for this test: {[k for k, v in benchmark.items()]}"
)

# ------------------------------------------------
# COT
# ------------------------------------------------
if do_cot:
    cot = MedRAG(llm_name=llm_name, rag=False, cache_dir=cache_dir, llm_local=llm_local)

    # Prediction
    print(f"[In progress] Inferencing with model [{llm}] with CoT...")
    for dataset_name in benchmark:
        split = set_split(dataset_name)

        prediction_res_folder_cot, prediction_res_folder_rag = (
            prepare_prediction_output_folders(
                prediction_folder,
                dataset_name,
                rag_k,
                llm,
                clear_previous_eval_results=clear_previous_eval_results,
                corpus_name=corpus_name.lower(),
                retriever_name=retriever_name.lower(),
                pred_with_router=pred_with_router,
                exp_option=args.exp_option,
            )
        )

        dataset, index = load_benchmark_dataset(
            dataset_name=dataset_name,
            benchmark_repo_dir=benchmark_repo_dir,
        )

        # Create the tqdm progress bar separately so track the tasks "finished"
        pbar_cot = tqdm(total=len(dataset), desc=f"{dataset_name}_cot")

        run_cot(
            cot,
            dataset,
            index,
            prediction_res_folder_cot,
            prediction_res_folder_rag,
            split,
            pbar_cot,
            max_threads=32,
        )

        pbar_cot.close()

# ------------------------------------------------
# RAG
# ------------------------------------------------
if do_rag:
    medrag = MedRAG(
        llm_name=llm_name,
        rag=True,
        retriever_name=retriever_name,
        corpus_name=corpus_name,
        cache_dir=cache_dir,
        db_dir=db_dir,
        llm_local=llm_local,
        rrf_k=rrf_k,
        pred_with_router=pred_with_router,
        router_model=router,
        router_merge_k=router_merge_k,
#       threshold=threshold,
    )

    # Prediction
    print(f"[In progress] Inferencing with model [{llm}] with MedRAG...")

    for dataset_name in benchmark:
        split = set_split(dataset_name)

        prediction_res_folder_cot, prediction_res_folder_rag = (
            prepare_prediction_output_folders(
                prediction_folder,
                dataset_name,
                rag_k,
                llm,
                clear_previous_eval_results=clear_previous_eval_results,
                corpus_name=corpus_name.lower(),
                retriever_name=retriever_name.lower(),
                pred_with_router=pred_with_router,
                exp_option=args.exp_option,
            )
        )

        dataset, index = load_benchmark_dataset(
            dataset_name=dataset_name,
            benchmark_repo_dir=benchmark_repo_dir,
        )
        try:
            threshold = thresholds_dict[dataset_name]
            print(f"[In progress] running with threshold [{threshold}] for [{dataset_name}]...")
        except KeyError as e:
            print(f"Error: {e} does not exist in thresholds_dict.")

        # Create the tqdm progress bar separately so track the tasks "finished"
        pbar_rag = tqdm(total=len(dataset), desc=f"{dataset_name}_rag")

        run_rag(
            medrag,
            dataset,
            index,
            prediction_res_folder_cot,
            prediction_res_folder_rag,
            split,
            pbar_rag,
            rag_k,
            threshold=threshold,
            max_threads=1,
        )

        pbar_rag.close()

if do_cot or do_rag:
    print("\n[Done] Inference on MIRAGE benchmark datasets finished.")

    print("The results of prediction are stored in:")
    print(f"{prediction_folder}")
