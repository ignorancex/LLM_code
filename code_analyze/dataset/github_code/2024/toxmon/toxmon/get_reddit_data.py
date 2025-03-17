import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import praw
import prawcore
from praw.models import Comment, Submission
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("log.log", mode="a"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


@dataclass
class RedditDataRetriever:
    data_dir: str
    reddit: praw.Reddit = field(init=False)
    threads_dir: str = field(init=False)
    comments_dir: str = field(init=False)

    def __post_init__(self):
        self.reddit = praw.Reddit("bot1", config_interpolation="basic")
        self.threads_dir = os.path.join(self.data_dir, "threads")
        self.comments_dir = os.path.join(self.data_dir, "comments")
        os.makedirs(self.threads_dir, exist_ok=True)
        os.makedirs(self.comments_dir, exist_ok=True)

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=60, max=180),
        retry=retry_if_exception_type(
            (prawcore.exceptions.TooManyRequests, prawcore.exceptions.RequestException)
        ),
        reraise=True,  # Reraise the exception after all retries have failed
        before_sleep=before_sleep_log(logger, logging.INFO),  # Logging before sleep
    )
    def _get_submissions(
        self,
        subreddit_name: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 1000,
    ) -> List[praw.models.Submission]:
        """
        API call to get submissions wrapped with tenacity for retries.
        Allows sorting by 'top', 'hot', 'new', etc. and time filtering for 'top' and 'controversial'.
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        if sort in ["top", "controversial"]:
            return list(getattr(subreddit, sort)(time_filter=time_filter, limit=limit))
        else:
            return list(getattr(subreddit, sort)(limit=limit))

    @staticmethod
    def _serialize_praw_object(obj: Any) -> Any:
        if hasattr(obj, "json_dict"):
            return obj.json_dict()
        elif hasattr(obj, "__dict__"):
            data = obj.__dict__
            return {
                key: RedditDataRetriever._serialize_praw_object(value)
                for key, value in data.items()
                if not key.startswith("_")
            }
        else:
            return str(obj)

    def _process_comment(
        self,
        comment: Comment,
    ) -> Dict[str, Union[str, int, float, List[Any], None]]:
        """Process a single comment to extract relevant information."""
        return {
            "id": comment.id,
            "body": comment.body,
            "author": str(comment.author) if comment.author else None,
            "created_utc": comment.created_utc,
            "score": comment.score,
            "is_submitter": comment.is_submitter,
            "parent_id": comment.parent_id,
            "link_id": comment.link_id,
            "permalink": comment.permalink,
            "controversiality": comment.controversiality,
            "gilded": comment.gilded,
            "likes": comment.likes,
            "num_reports": comment.num_reports
            if hasattr(comment, "num_reports")
            else None,  # num_reports might not be available
            "replies": [self._process_comment(reply) for reply in comment.replies]
            if comment.replies
            else [],
            "saved": comment.saved,
            "score_hidden": comment.score_hidden,
            "stickied": comment.stickied,
            "subreddit_id": comment.subreddit_id,
            "total_awards_received": comment.total_awards_received,
            "upvote_ratio": getattr(
                comment, "upvote_ratio", None
            ),  # upvote_ratio may not always be present
            "depth": comment.depth,
        }

    def _extract_comments(self, submission: Submission) -> List[Dict[str, Any]]:
        """Extract comments from a submission."""
        submission.comments.replace_more(limit=0)  # Ensures no "MoreComments" objects
        return [
            self._process_comment(comment) for comment in submission.comments.list()
        ]

    def _serialize_submission(self, submission: Submission) -> Dict[str, Any]:
        submission_dict = vars(submission)
        serialized_data = {
            key: self._serialize_praw_object(value)
            for key, value in submission_dict.items()
            if not key.startswith("_")
        }
        return serialized_data

    def _save_data(
        self, data: Dict[str, Any], filename: str, is_comment: bool = False
    ) -> None:
        """Save data to a JSON file in the specified directory."""
        directory = self.comments_dir if is_comment else self.threads_dir
        file_path = os.path.join(directory, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _save_checkpoint(
        self, index: int, subreddit_name: str, sort: str, time_filter: str
    ) -> None:
        """Saves the last processed index to a single file with sort type and time filter as keys."""
        checkpoint_filename = "checkpoint.json"
        checkpoint_key = f"{subreddit_name}_{sort}_{time_filter}"

        try:
            with open(os.path.join(self.data_dir, checkpoint_filename), "r") as f:
                checkpoints = json.load(f)
        except FileNotFoundError:
            checkpoints = {}

        checkpoints[checkpoint_key] = {"last_processed_index": index}

        with open(os.path.join(self.data_dir, checkpoint_filename), "w") as f:
            json.dump(checkpoints, f)

    def _load_checkpoint(self, subreddit_name: str, sort: str, time_filter: str) -> int:
        """Loads the last processed index from a single file using a specific key with time filter."""
        checkpoint_filename = "checkpoint.json"
        checkpoint_key = f"{subreddit_name}_{sort}_{time_filter}"

        try:
            with open(os.path.join(self.data_dir, checkpoint_filename), "r") as f:
                checkpoints = json.load(f)
            return checkpoints.get(checkpoint_key, {}).get("last_processed_index", 0)
        except FileNotFoundError:
            return 0  # If the file doesn't exist, start from the beginning

    def retrieve_threads(
        self,
        subreddit_name: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = "1000",
        log_progress: bool = True,
    ) -> None:
        """High-level method to retrieve top threads and handle API interaction."""
        last_processed_index = self._load_checkpoint(subreddit_name, sort, time_filter)
        try:
            top_submissions = self._get_submissions(
                subreddit_name=subreddit_name,
                limit=limit,
                sort=sort,
                time_filter=time_filter,
            )
            print(limit, last_processed_index, len(top_submissions))
            for i, submission in enumerate(
                top_submissions[last_processed_index:], start=last_processed_index
            ):
                if log_progress:
                    logger.info(
                        f"Processing thread {i + 1} of {limit} from r/{subreddit_name} from {sort} of {time_filter}..."
                    )

                submission_data = self._serialize_submission(submission)
                submission_filename = f"{subreddit_name}_{submission.id}.json"
                self._save_data(submission_data, submission_filename)
                time.sleep(1)

                comments_data = self._extract_comments(submission)
                comments_filename = f"{subreddit_name}_{submission.id}_comments.json"
                self._save_data(comments_data, comments_filename, is_comment=True)
                self._save_checkpoint(i, subreddit_name, sort, time_filter)
                time.sleep(1)

        except Exception as e:
            logger.error(f"An error occurred while retrieving top threads: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit Data Retrieval Script")
    parser.add_argument("subreddit", help="Name of the subreddit")
    parser.add_argument("n_threads", type=int, help="Number of threads to retrieve")
    parser.add_argument("sort", help="Sort type (e.g., top, hot, new)")
    parser.add_argument(
        "--time_filter",
        default="all",
        help="Time filter for sorting, if applicable (day, week, month, year, all)",
    )

    args = parser.parse_args()

    retriever = RedditDataRetriever("../data")
    retriever.retrieve_threads(
        subreddit_name=args.subreddit,
        sort=args.sort,
        time_filter=args.time_filter,
        limit=args.n_threads,
    )
