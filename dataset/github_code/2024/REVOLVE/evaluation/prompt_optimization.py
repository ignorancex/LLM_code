import os
import argparse
import concurrent
from dotenv import load_dotenv
load_dotenv(override=True)
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random
import json


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    parser.add_argument("--backbone_engine", type=str, default="azure-gpt4o", help="The backbone engine for textgrad.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="The model on which the prompt is optimized.")
    parser.add_argument("--optimizer_version", type=str, default="v1", help="The optimizer to use.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=1, help="The maximum number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--num_threads", type=int, default=3, help="The number of threads to use for evaluation.")
    return parser.parse_args()


def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):

            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    print("Running the current prompt on the validation set...")
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("Val acc: ", val_performance)
    print("Previous acc: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if val_performance < previous_performance:
        print(f"Rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


def get_eval_output(x, y, model, eval_fn):
    """
    Helper function to be used in the ThreadPoolExecutor to evaluate the model on a batch of data.
    """
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    except:
        eval_output_variable = eval_fn([x, y, response])
    return eval_output_variable


args = config()
print(vars(args))
set_seed(args.seed)
if 'llama' in args.backbone_engine:
    llm_api = tg.get_engine(engine_name=args.backbone_engine, batch_size=args.num_threads)
else:
    llm_api = tg.get_engine(engine_name=args.backbone_engine)
tg.set_backward_engine(llm_api, override=True)

# Load the data and the evaluation function
train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()

train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
print(f'Starting system prompt: {STARTING_SYSTEM_PROMPT}')

system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")

if 'llama' in args.model:
    model_api = tg.get_engine(engine_name=args.model, batch_size=args.num_threads)
else:
    model_api = tg.get_engine(engine_name=args.model)
model = tg.BlackboxLLM(model_api, system_prompt)

if args.optimizer_version == "v1":
    optimizer = tg.TextualGradientDescent(engine=llm_api, parameters=[system_prompt])
elif args.optimizer_version == "v1_momentum":
    optimizer = tg.TextualGradientDescentwithMomentum(engine=llm_api, parameters=[system_prompt], momentum_window=12)
elif args.optimizer_version == "v2":
    optimizer = tg.TextualGradientDescent_v2(engine=llm_api, parameters=[system_prompt])
else:
    raise ValueError(f"Invalid optimizer version: {args.optimizer_version}")

results = {"test_acc": [], "prompt": [], "validation_acc": []}
print("Evaluating the initial prompt on the test set...")
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
print("Evaluating the initial prompt on the validation set...")
results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())

for epoch in range(args.max_epochs):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        optimizer.zero_grad()
        losses = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = []
            for (x, y) in zip(batch_x, batch_y):
                future = executor.submit(get_eval_output, x, y, model, eval_fn)
                futures.append(future)

            tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
            for future in tqdm_loader:
                eval_output_variable = future.result()
                losses.append(eval_output_variable)

        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()
        if args.run_validation:
            run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        print("Current sys prompt: ", system_prompt)
        print("Evaluating the current prompt on the test set...")
        test_acc = eval_dataset(test_set, eval_fn, model)
        results["test_acc"].append(test_acc)
        results["prompt"].append(system_prompt.get_value())
        if steps == 11:
            break

os.makedirs("./results", exist_ok=True)
model_name = args.backbone_engine.split("/")[-1]
with open(f"./results/results_{args.task}_{model_name}_{args.optimizer_version}.json", "w") as f:
    json.dump(results, f)
