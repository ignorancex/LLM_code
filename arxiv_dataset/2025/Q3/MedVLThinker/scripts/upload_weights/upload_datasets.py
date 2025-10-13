#!/usr/bin/env python3
"""
Script to upload datasets from med-vlrm organization to UCSC-VLAA organization on Hugging Face.

This script downloads datasets from the source organization and uploads them to the target organization
with renamed repositories.

Requirements:
- huggingface_hub
- datasets
- git-lfs (for large files)

Usage:
    python upload_datasets.py [--dry-run] [--token YOUR_HF_TOKEN]
"""

import argparse
import logging
import tempfile
from typing import Dict

from huggingface_hub import HfApi, Repository, create_repo, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Dataset mappings: source -> target
DATASET_MAPPINGS = {
    "med-vlrm/med-vlm-pmc_vqa-gpt_4o_reasoning-tokenized": "UCSC-VLAA/MedVLThinker-pmc_vqa-gpt_4o_reasoning-tokenized",
    "med-vlrm/med-vlm-eval-v2": "UCSC-VLAA/MedVLThinker-Eval",
    "med-vlrm/med-vlm-pmc_vqa": "UCSC-VLAA/MedVLThinker-pmc_vqa",
    "med-vlrm/med-vlm-m23k-tokenized": "UCSC-VLAA/MedVLThinker-m23k-tokenized",
}


class DatasetUploader:
    def __init__(self, token: str = None, dry_run: bool = False):
        """
        Initialize the dataset uploader.

        Args:
            token: Hugging Face API token. If None, will try to use cached token.
            dry_run: If True, only show what would be done without actually uploading.
        """
        self.api = HfApi(token=token)
        self.dry_run = dry_run
        self.token = token

        if not dry_run:
            # Verify authentication
            try:
                user_info = self.api.whoami()
                logger.info(f"Authenticated as: {user_info['name']}")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                raise

    def check_source_dataset_exists(self, repo_id: str) -> bool:
        """Check if source dataset exists."""
        try:
            self.api.dataset_info(repo_id)
            return True
        except RepositoryNotFoundError:
            return False
        except Exception as e:
            logger.warning(f"Error checking dataset {repo_id}: {e}")
            return False

    def create_target_repo(self, repo_id: str, source_repo_id: str) -> bool:
        """
        Create target repository if it doesn't exist.

        Args:
            repo_id: Target repository ID
            source_repo_id: Source repository ID (for description)

        Returns:
            True if repo was created or already exists, False otherwise
        """
        try:
            # Check if repo already exists
            self.api.dataset_info(repo_id)
            logger.info(f"Target repository {repo_id} already exists")
            return True
        except RepositoryNotFoundError:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would create repository: {repo_id}")
                return True

            try:
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=False,
                    token=self.token,
                )
                logger.info(f"Created repository: {repo_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create repository {repo_id}: {e}")
                return False
        except Exception as e:
            logger.error(f"Error checking repository {repo_id}: {e}")
            return False

    def upload_dataset(self, source_repo_id: str, target_repo_id: str) -> bool:
        """
        Upload dataset from source to target repository.

        Args:
            source_repo_id: Source repository ID
            target_repo_id: Target repository ID

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Processing: {source_repo_id} -> {target_repo_id}")

        # Check if source exists
        if not self.check_source_dataset_exists(source_repo_id):
            logger.error(f"Source dataset {source_repo_id} not found")
            return False

        # Create target repo
        if not self.create_target_repo(target_repo_id, source_repo_id):
            return False

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would download and upload: {source_repo_id} -> {target_repo_id}"
            )
            return True

        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                logger.info(f"Downloading {source_repo_id}...")

                # Download source dataset
                download_path = snapshot_download(
                    repo_id=source_repo_id,
                    repo_type="dataset",
                    local_dir=temp_dir,
                    token=self.token,
                )

                logger.info(f"Downloaded to {download_path}")

                # Initialize target repository
                logger.info(f"Uploading to {target_repo_id}...")

                # Use Repository class for upload
                repo = Repository(
                    local_dir=temp_dir,
                    clone_from=target_repo_id,
                    repo_type="dataset",
                    token=self.token,
                )

                # Commit and push all files
                repo.git_add()
                repo.git_commit(f"Upload dataset from {source_repo_id}")
                repo.git_push()

                logger.info(
                    f"Successfully uploaded {source_repo_id} -> {target_repo_id}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to upload {source_repo_id} -> {target_repo_id}: {e}"
                )
                return False

    def upload_all_datasets(self) -> Dict[str, bool]:
        """
        Upload all datasets defined in DATASET_MAPPINGS.

        Returns:
            Dictionary mapping source repo to success status
        """
        results = {}

        logger.info(f"Starting upload of {len(DATASET_MAPPINGS)} datasets...")

        for source_repo, target_repo in DATASET_MAPPINGS.items():
            success = self.upload_dataset(source_repo, target_repo)
            results[source_repo] = success

            if success:
                logger.info(f"✅ Successfully processed: {source_repo}")
            else:
                logger.error(f"❌ Failed to process: {source_repo}")

        return results

    def print_summary(self, results: Dict[str, bool]):
        """Print summary of upload results."""
        successful = sum(1 for success in results.values() if success)
        total = len(results)

        logger.info(f"\n{'='*50}")
        logger.info("UPLOAD SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total datasets: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {total - successful}")

        if successful < total:
            logger.info("\nFailed datasets:")
            for source_repo, success in results.items():
                if not success:
                    target_repo = DATASET_MAPPINGS[source_repo]
                    logger.info(f"  - {source_repo} -> {target_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload datasets from med-vlrm to UCSC-VLAA organization"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token (if not provided, will use cached token)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually uploading",
    )
    parser.add_argument(
        "--dataset", type=str, help="Upload only a specific dataset (source repo ID)"
    )

    args = parser.parse_args()

    # Initialize uploader
    try:
        uploader = DatasetUploader(token=args.token, dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"Failed to initialize uploader: {e}")
        return 1

    # Upload datasets
    if args.dataset:
        if args.dataset not in DATASET_MAPPINGS:
            logger.error(f"Dataset {args.dataset} not found in mappings")
            logger.info(f"Available datasets: {list(DATASET_MAPPINGS.keys())}")
            return 1

        target_repo = DATASET_MAPPINGS[args.dataset]
        success = uploader.upload_dataset(args.dataset, target_repo)
        results = {args.dataset: success}
    else:
        results = uploader.upload_all_datasets()

    # Print summary
    uploader.print_summary(results)

    # Return appropriate exit code
    failed_count = sum(1 for success in results.values() if not success)
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    exit(main())
