#!/usr/bin/env python3
"""
Evaluator Script for CRAG-MM dataset

This script evaluates an agent (using a user-provided agent `UserAgent` as configured in `agents/user_config.py`) 
on the CRAG-MM dataset. It generates responses, evaluates them (using an optional semantic evaluation model via OpenAI API),
computes multi-turn conversation metrics, and optionally saves the results.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

# Set tokenizers parallelism before importing any HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from agents.base_agent import BaseAgent
from agents.user_config import UserAgent
from crag_batch_iterator import CRAGTurnBatchIterator
from cragmm_search.search import UnifiedSearchPipeline
from utils import display_results, ensure_crag_cache_dir_is_configured
from tokenizers import Tokenizer

# Load environment variables
load_dotenv()
ensure_crag_cache_dir_is_configured()

console = Console()

# Constants for configuration
DEFAULT_EVAL_MODEL = "gpt-4o-mini"
MAX_API_RETRIES = 3
DEFAULT_NUM_WORKERS = 8

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 16


MAX_RESPONSE_LENGTH_IN_TOKENS = 75


class CRAGTurnEvaluationResult(BaseModel):
    """Structured output model for CRAG turn evaluation results."""
    accuracy: bool


class CRAGEvaluator:
    """
    A class to evaluate an agent on the CRAG-MM dataset.

    This evaluator generates responses, evaluates them (optionally using a semantic evaluation model),
    computes multi-turn conversation metrics, and (optionally) saves the results.
    """
    def __init__(
        self,
        dataset: Dataset,
        agent: BaseAgent,
        eval_model_name: str | None = None,
        num_conversations: int | None = None,
        show_progress: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        self.dataset = dataset
        self.agent = agent
        self.eval_model_name = eval_model_name
        self.num_conversations = num_conversations
        self.show_progress = show_progress
        self.num_workers = num_workers

        # Internal state for evaluation; these are set during initialization
        self.batch_iterator: CRAGTurnBatchIterator | None = None
        self.conversations_count: int = 0
        self.agent_response_map: dict[str, str] = {}
        self.all_turn_data: list[dict[str, any]] = []
        self.session_ids_evaluated: set[str] = set()
        
        self.tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.tokenizer.enable_truncation(max_length=MAX_RESPONSE_LENGTH_IN_TOKENS)



    @staticmethod
    def get_system_message() -> str:
        """
        Returns the system message for the evaluator.
        """
        return (
            "You are an expert evaluator for question answering systems. "
            "Your task is to determine if a prediction correctly answers a question based on the ground truth.\n\n"
            "Rules:\n"
            "1. The prediction is correct if it captures all the key information from the ground truth.\n"
            "2. The prediction is correct even if phrased differently as long as the meaning is the same.\n"
            "3. The prediction is incorrect if it contains incorrect information or is missing essential details.\n"
            "Output a JSON object with a single field 'accuracy' whose value is true or false."
        )

    def attempt_api_call(
        self,
        client: OpenAI,
        model_name: str,
        messages: list,
        max_retries: int = MAX_API_RETRIES,
    ) -> CRAGTurnEvaluationResult | None:
        """
        Attempt a structured output call to the OpenAI API with retries.

        Args:
            client: The OpenAI client instance to use for the API call.
            model_name: The model to query (e.g., "gpt-4o-mini").
            messages: List of message objects for the conversation.
            max_retries: Maximum number of retry attempts before giving up.

        Returns:
            CRAGTurnEvaluationResult object if successful, None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=CRAGTurnEvaluationResult,
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                error_message = f"API call failed on attempt {attempt + 1}/{max_retries}: {str(e)}"
                if attempt == max_retries - 1:
                    console.print(f"[red]Failed after {MAX_API_RETRIES} attempts: {str(e)}[/red]")
                else:
                    console.print(f"[yellow]{error_message}, retrying...[/yellow]")
        return None

    def evaluate_response(self, crag_turn_data: dict[str, any]) -> dict[str, any]:
        """
        Evaluate a single response and return evaluation results.

        Args:
            crag_turn_data: A dictionary containing query, ground truth, and agent response.

        Returns:
            A dictionary with evaluation results added to crag_turn_data.
        """
        agent_response = crag_turn_data["agent_response"]
        ground_truth = crag_turn_data["ground_truth"]
        query = crag_turn_data["query"]

        is_idk = "i don't know" in agent_response.lower()
        is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
        is_semantically_correct = False
        api_response = None

        # Begin by assuming exact match correctness
        is_correct = is_exact_match

        # Use semantic evaluation if not an exact match and an evaluation model is provided.
        if not is_idk and not is_exact_match and self.eval_model_name:
            local_openai_client = OpenAI()
            messages = [
                {"role": "system", "content": self.get_system_message()},
                {"role": "user", "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n"},
            ]
            api_response = self.attempt_api_call(local_openai_client, self.eval_model_name, messages)
            if api_response:
                is_semantically_correct = api_response.accuracy
                is_correct = is_semantically_correct
        if is_exact_match:
            is_semantically_correct = True

        return {
            **crag_turn_data,
            "is_exact_match": is_exact_match,
            "is_correct": is_correct,
            "is_miss": is_idk,
            "is_semantically_correct": is_semantically_correct,
            "api_response": api_response.model_dump() if api_response else None,
        }

    def initialize_evaluation(self) -> None:
        """
        Initialize variables needed for agent evaluation.

        This method sets internal state including the batch iterator, conversation count, 
        agent response map, and turn data list.
        """
        console.print(f"[blue]Starting evaluation with {self.num_workers} workers[/blue]")
        if self.eval_model_name:
            console.print(f"[blue]Using semantic evaluation with model: {self.eval_model_name}[/blue]")

        self.conversations_count = len(self.dataset) if self.num_conversations is None else min(self.num_conversations, len(self.dataset))
        batch_size = int(np.clip(self.agent.get_batch_size(), MIN_BATCH_SIZE, MAX_BATCH_SIZE))
        self.agent_response_map = {}
        self.all_turn_data = []
        self.session_ids_evaluated = set()

        # Instantiate the CRAG turn based batch iterator 
        self.batch_iterator = CRAGTurnBatchIterator(dataset=self.dataset, batch_size=batch_size, shuffle=False)

    def generate_agent_responses(self, progress_callback: Callable[[int, int], None] = None) -> None:
        """
        Phase 1: Generate agent responses for each turn in the dataset.
        Phase 1: Generate agent responses for each turn in the dataset.

        This method iterates over the dataset batches using the internal batch iterator and updates the evaluator's state
        with agent responses and turn data.
        """
        if self.batch_iterator is None:
            raise ValueError("Batch iterator is not initialized. Please call initialize_evaluation() first.")

        for batch_idx, batch in enumerate(tqdm.tqdm(self.batch_iterator, desc="Generating responses", disable=not self.show_progress)):
            interaction_ids = batch["interaction_ids"]
            queries = batch["queries"]
            images = batch["images"]
            conversation_histories = batch["conversation_histories"]

            message_histories = []
            interaction_id_histories = []
            # Build message histories for multi-turn conversations
            for conversation_history in conversation_histories:
                message_history = []
                interaction_id_history = []
                for turn in conversation_history:
                    turn_interaction_id = turn["interaction_id"]
                    turn_agent_response = self.agent_response_map.get(turn_interaction_id)
                    if not turn_agent_response:
                        raise AssertionError(
                            f"Agent response not found for turn {turn_interaction_id}. "
                            "Did you shuffle the multi-turn conversations by mistake?"
                        )
                    message_history.append({"role": "user", "content": turn["query"]})
                    message_history.append({"role": "assistant", "content": turn_agent_response})
                    interaction_id_history.append(turn_interaction_id)
                message_histories.append(message_history)
                interaction_id_histories.append(interaction_id_history)

            # Generate responses for the current batch
            agent_responses = self.agent.batch_generate_response(queries, images, message_histories)
            agent_responses = self.truncate_agent_responses(agent_responses) # Truncase each response to the maximum allowed length (75 tokens)
            
            # Collect responses and add evaluation data
            for idx, interaction_id in enumerate(interaction_ids):
                agent_response = agent_responses[idx]
                self.agent_response_map[interaction_id] = agent_response
                self.all_turn_data.append({
                    "session_id": batch["session_ids"][idx],
                    "interaction_id": interaction_id,
                    "turn_idx": batch["turn_idxs"][idx],
                    "is_ego": batch["image_urls"][idx] is None,
                    "image_quality": batch["image_qualities"][idx],
                    "query_category": batch["query_categories"][idx],
                    "domain": batch["domains"][idx],
                    "dynamism": batch["dynamisms"][idx],
                    "query": queries[idx],
                    "ground_truth": batch["answers"][idx],
                    "agent_response": agent_response,
                    "total_turn_count": batch["total_turn_counts"][idx],
                    "interaction_id_history": interaction_id_histories[idx]
                })
                self.session_ids_evaluated.add(batch["session_ids"][idx])

            if progress_callback:
                conversations_evaluated = len(self.session_ids_evaluated)
                progress_callback(conversations_evaluated, self.conversations_count)

            if len(self.session_ids_evaluated) > self.conversations_count:
                console.print(f"[yellow]Already evaluated {len(self.session_ids_evaluated)} conversations. Abruptly stopping evaluation.[/yellow]")
                break

    def evaluate_agent_responses(
        self,
        turn_data: list[dict[str, any]],
        progress_callback: Callable[[int, int], None] = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
        """
        Phase 2: Evaluate agent responses and calculate scores.

        This method uses a thread-based parallel executor to avoid pickling issues.
        Args:
            turn_data: List of turn data including agent responses.
        Returns:
            A tuple containing turn evaluation results and score dictionaries.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.evaluate_response, data) for data in turn_data]
            for future_idx, future in tqdm.tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Evaluating responses", disable=not self.show_progress):
                results.append(future.result())
                if progress_callback is not None:
                    progress_callback(future_idx, len(turn_data))

        # Convert the interim evaluation results to a pandas dataframe
        turn_evaluation_results_df = pd.DataFrame(results)
        turn_evaluation_results_df = turn_evaluation_results_df.sort_values(by=["session_id", "turn_idx"])

        ego_turn_evaluation_results_df = turn_evaluation_results_df[turn_evaluation_results_df["is_ego"] == True]

        all_scores_dictionary = self.calculate_scores(turn_evaluation_results_df)
        ego_scores_dictionary = self.calculate_scores(ego_turn_evaluation_results_df)

        turn_evaluation_results = {"all": turn_evaluation_results_df, "ego": ego_turn_evaluation_results_df}
        score_dictionaries = {"all": all_scores_dictionary, "ego": ego_scores_dictionary}

        return turn_evaluation_results, score_dictionaries

    def calculate_scores(self, turn_evaluation_results_df: pd.DataFrame) -> dict[str, float]:
        """
        Calculate scores for both single-turn and multi-turn conversations.

        Args:
            turn_evaluation_results_df: DataFrame with evaluation results for turns.
        Returns:
            Dictionary of calculated metrics.
        """
        multi_turn_conversation_score_map: dict[str, float] = {}

        def _set_is_correct_false_after_consecutive(group: pd.DataFrame) -> pd.DataFrame:
            """
            Mark as is_miss after consecutive incorrect responses
            and calculate multi-turn conversation score for each conversation.
            """
            group_copy = group.copy().reset_index(drop=True)
            for i in range(1, len(group_copy)):
                if not group_copy.loc[i - 1, 'is_correct'] and not group_copy.loc[i, 'is_correct']:
                    group_copy.loc[i + 1:, 'is_correct'] = False
                    group_copy.loc[i + 1:, 'is_exact_match'] = False
                    group_copy.loc[i + 1:, 'is_miss'] = True
                    group_copy.loc[i + 1:, 'is_semantically_correct'] = False
                    break

            group_copy["is_hallucination"] = ~group_copy["is_correct"] & ~group_copy["is_miss"]
            multi_turn_conversation_score = group_copy["is_correct"].mean() - group_copy["is_hallucination"].mean()
            group_copy["multi_turn_conversation_score"] = multi_turn_conversation_score
            session_id = group_copy.iloc[0]["session_id"]
            multi_turn_conversation_score_map[session_id] = multi_turn_conversation_score
            return group_copy

        turn_evaluation_results_df = turn_evaluation_results_df.groupby("session_id", group_keys=False)[turn_evaluation_results_df.columns].apply(_set_is_correct_false_after_consecutive)

        total = len(turn_evaluation_results_df)
        correct_exact = turn_evaluation_results_df["is_exact_match"].sum()
        correct = turn_evaluation_results_df["is_correct"].sum()
        miss = turn_evaluation_results_df["is_miss"].sum()
        hallucination = total - (correct + miss)

        exact_match = correct_exact / total
        accuracy = correct / total
        missing = miss / total
        hallucination_rate = hallucination / total
        truthfulness_score = ((2 * correct + miss) / total) - 1 if total > 1 else 0.0
        mean_multi_turn_conversation_score = np.mean(list(multi_turn_conversation_score_map.values()))

        scores_dictionary = {
            "total": float(total),
            "correct_exact": float(correct_exact),
            "correct": float(correct),
            "miss": float(miss),
            "hallucination": float(hallucination),
            "exact_match": float(exact_match),
            "accuracy": float(accuracy),
            "missing": float(missing),
            "hallucination_rate": float(hallucination_rate),
            "truthfulness_score": float(truthfulness_score),
            "mean_multi_turn_conversation_score": float(mean_multi_turn_conversation_score)
        }

        return scores_dictionary

    def save_results(self, turn_evaluation_results: dict[str, any], scores_dictionary: dict[str, any], output_dir: str) -> None:
        """
        Save evaluation results to the specified directory.

        Args:
            turn_evaluation_results: The evaluation results to save.
            scores_dictionary: The scores dictionary to save.
            output_dir: Path where to save the results.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        turn_evaluation_results["all"].to_csv(os.path.join(output_dir, "turn_evaluation_results_all.csv"), index=False)
        turn_evaluation_results["ego"].to_csv(os.path.join(output_dir, "turn_evaluation_results_ego.csv"), index=False)
        with open(os.path.join(output_dir, "scores_dictionary.json"), "w") as f:
            json.dump(scores_dictionary, f, indent=2)

    def evaluate_agent(self) -> tuple[dict[str, any], dict[str, any]]:
        """
        Evaluate an agent on a dataset and return performance metrics.

        Returns:
            A tuple containing a dictionary of turn evaluation results and a dictionary of scores.
        """
        # Phase 0: Initialize evaluation state
        self.initialize_evaluation()
        
        # Phase 1: Generate agent responses (updates internal state)
        def _generation_progress_callback(conversations_evaluated: int, total_conversations: int) -> None:
            # Can be useful to track progress of the evaluation
            # console.log(f"[blue]Generated responses for {conversations_evaluated}/{total_conversations} conversations[/blue]")
            pass
            
        self.generate_agent_responses(_generation_progress_callback)
        
        # Phase 2: Evaluate responses using stored turn data
        
        def _evaluation_progress_callback(turn_evaluated: int, total_turns: int) -> None:
            # Can be useful to track progress of the evaluation
            # console.log(f"[blue]Evaluated {turn_evaluated}/{total_turns} turns[/blue]")
            pass
            
        turn_evaluation_results, score_dictionaries = self.evaluate_agent_responses(self.all_turn_data, _evaluation_progress_callback)
        return turn_evaluation_results, score_dictionaries
    
    def truncate_agent_responses(self, agent_responses: list[str]) -> list[str]:
        """
        Truncate each agent response to the maximum allowed length.
        """
        encodings = self.tokenizer.encode_batch(agent_responses)
        trimmed_agent_responses = [self.tokenizer.decode(enc.ids) for enc in encodings]
        return trimmed_agent_responses    


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent on the CRAG-MM dataset"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="single-turn",
        choices=["single-turn", "multi-turn"],
        help="Dataset type to load",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use ('validation', 'public_test')",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=-1,
        help="Number of conversations to evaluate (default: -1). -1 evaluates all conversations, while a positive number evaluates that many conversations.",
    )
    parser.add_argument(
        "--suppress-web-search-api",
        action="store_true",
        help="Suppress web search API when calling the agent"
    )
    parser.add_argument(
        "--display-conversations",
        type=int,
        default=10,
        help="Number of evaluation examples to show",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help="OpenAI model for semantic evaluation. Pass 'None' to disable semantic evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to save turn evaluation results and scores dictionary",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="v0.1.2",
        help="Dataset revision/version to use when loading from HuggingFace",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of worker processes for parallel evaluation (default: {DEFAULT_NUM_WORKERS})",
    )
    args = parser.parse_args()

    console.print(f"[bold blue]Loading {args.dataset_type} dataset...[/bold blue]")
    repo_name = f"crag-mm-2025/crag-mm-{args.dataset_type}-public"
    console.print(
        f"[bold green]Loading from HuggingFace:[/bold green] {repo_name} (revision: {args.revision})"
    )
    dataset = load_dataset(repo_name, revision=args.revision)
    available_splits = list(dataset.keys())
    split_to_use = args.split if args.split in available_splits else available_splits[0]
    console.print(
        f"[bold green]Using split:[/bold green] '{split_to_use}' with {len(dataset[split_to_use])} examples"
    )

    if args.eval_model.lower() == "none":
        args.eval_model = None
        console.print(
            Panel(
                "[bold red]WARNING: SEMANTIC EVALUATION IS DISABLED[/bold red]\n\n"
                "No calls to LLM-as-a-Judge will be made!\n"
                "Results will rely only on exact string matching.",
                title="[bold red]ATTENTION[/bold red]",
                border_style="red",
                width=100,
                padding=(2, 5),
                expand=False,
            )
        )

    if args.num_conversations == -1:
        args.num_conversations = len(dataset[split_to_use])

    # Suppress web search API if the flag is set - useful for Task 1 (Single-source Augmentation)
    search_api_text_model_name = "BAAI/bge-large-en-v1.5"
    search_api_image_model_name = "openai/clip-vit-large-patch14-336"
    search_api_web_hf_dataset_id = "crag-mm-2025/web-search-index-validation"
    search_api_image_hf_dataset_id = "crag-mm-2025/image-search-index-validation"
        
    if args.suppress_web_search_api:
        # Suppress web search API - useful for Task 1 (Single-source Augmentation)
        search_api_web_hf_dataset_id = None
    
    search_pipeline = UnifiedSearchPipeline(
        text_model_name=search_api_text_model_name,
        image_model_name=search_api_image_model_name,
        web_hf_dataset_id=search_api_web_hf_dataset_id,
        image_hf_dataset_id=search_api_image_hf_dataset_id,
    )

    evaluator = CRAGEvaluator(
        dataset=dataset[split_to_use],
        agent=UserAgent(search_pipeline=search_pipeline),
        eval_model_name=args.eval_model,
        num_conversations=args.num_conversations,
        show_progress=not args.no_progress,
        num_workers=args.num_workers,
    )

    turn_evaluation_results, score_dictionaries = evaluator.evaluate_agent()

    display_results(
        console,
        turn_evaluation_results["all"],
        score_dictionaries["all"],
        display_conversations=args.display_conversations,
        is_ego=False,
        is_multi_turn=(args.dataset_type == "multi-turn"),
    )
    if len(turn_evaluation_results["ego"]) > 0:
        display_results(
            console,
            turn_evaluation_results["ego"],
            score_dictionaries["ego"],
            display_conversations=args.display_conversations,
            is_ego=True,
            is_multi_turn=(args.dataset_type == "multi-turn"),
        )

    if args.output_dir:
        evaluator.save_results(turn_evaluation_results, score_dictionaries, args.output_dir)


if __name__ == "__main__":
    main()
