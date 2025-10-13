from dotenv import load_dotenv
from pathlib import Path
import json
import datetime
from itertools import cycle
from src.core.config import Config
from src.core.models import DebateTopic, DebateType, DebateTotal
from src.core.utils import sanitize_model_name

def check_debate_completion(file_path: Path) -> bool:
    """Check if a debate file is complete (no -1 in speeches)"""
    try:
        debate = DebateTotal.load_from_json(file_path)

        # Check all speech types for both sides
        for side_output in [debate.proposition_output, debate.opposition_output]:
            for speech_type, speech_content in side_output.speeches.items():
                if speech_content == -1:
                    return False
        return True
    except Exception:
        return False

def count_complete_debates(output_dir: Path, model_name: str, debate_type: DebateType) -> int:
    """Count how many complete debates exist for a given model"""
    if not output_dir.exists():
        return 0

    model_name_for_path = sanitize_model_name(model_name)
    pattern = f"{model_name_for_path}_vs_self_{debate_type.value}_*.json"

    complete_count = 0
    for file_path in output_dir.glob(pattern):
        if check_debate_completion(file_path):
            complete_count += 1

    return complete_count

def remove_incomplete_debates(output_dir: Path, model_name: str, debate_type: DebateType):
    """Remove incomplete debate files for a given model"""
    if not output_dir.exists():
        return

    model_name_for_path = sanitize_model_name(model_name)
    pattern = f"{model_name_for_path}_vs_self_{debate_type.value}_*.json"

    for file_path in output_dir.glob(pattern):
        if not check_debate_completion(file_path):
            file_path.unlink()
            print(f"Removed incomplete debate: {file_path.name}")

def run_single_model_debate(
    model_name: str,
    topic: DebateTopic,
    config: Config,
    output_dir: Path,
    debate_type: DebateType
) -> None:
    """Run a debate where a model debates against itself with the specified debate type"""
    model_name_for_path = sanitize_model_name(model_name)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_to_store = output_dir / f"{model_name_for_path}_vs_self_{debate_type.value}_{timestamp_str}.json"

    logger = config.logger.get_logger()
    logger.info(f"Running debate: {model_name} vs self under {debate_type.value}")
    logger.info(f"Topic: {topic.topic_description}")

    try:
        config.debate_service.run_debate(
            proposition_model=model_name,
            opposition_model=model_name,
            motion=topic,
            path_to_store=path_to_store,
            debate_type=debate_type
        )
        logger.info(f"Successfully completed debate and saved to {path_to_store}")
    except Exception as e:
        logger.error(f"Error running debate: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()

    # Initialize config
    config = Config()
    logger = config.logger.get_logger()
    logger.info("Starting redteam v2 self-debate experiment script")

    debate_type = DebateType.REDTEAM_V2

    # Create output directory
    output_dir = Path("experiments/redteam_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load topics
    with open(config.topic_list_path, 'r') as f:
        topic_list_raw = json.load(f)
        topic_list = [DebateTopic(**topic) for topic in topic_list_raw]
        logger.info(f"Loaded {len(topic_list)} topics")

    # Load models
    with open(config.debate_models_list_path, 'r') as f:
        models = list(json.load(f).keys())
        logger.info(f"Loaded {len(models)} models")

    # Number of debates per model (should match topic list length)
    num_debates_per_model = 6
    assert len(topic_list) == num_debates_per_model, f"Topic list length ({len(topic_list)}) should match debates per model ({num_debates_per_model})"

    # Count total completed debates to advance the infinite topic iterator
    total_completed_debates = 0

    for model_idx, model in enumerate(models):
        logger.info(f"Processing model {model_idx+1}/{len(models)}: {model}")

        # Check existing complete debates for this model
        complete_count = count_complete_debates(output_dir, model, debate_type)
        logger.info(f"Found {complete_count} complete debates for {model}")

        # Remove any incomplete debates for this model
        remove_incomplete_debates(output_dir, model, debate_type)

        # Add to total count for topic cycling
        total_completed_debates += complete_count

        if complete_count < num_debates_per_model:
            logger.info(f"Running {num_debates_per_model - complete_count} additional debates for {model}")

            # Create infinite topic list and advance it to current position
            infinite_topic_list = cycle(topic_list)
            for _ in range(total_completed_debates):
                next(infinite_topic_list)

            # Run remaining debates
            for debate_count in range(complete_count, num_debates_per_model):
                topic = next(infinite_topic_list)
                logger.info(f"Starting debate {debate_count+1}/{num_debates_per_model}")

                run_single_model_debate(
                    model_name=model,
                    topic=topic,
                    config=config,
                    output_dir=output_dir,
                    debate_type=debate_type
                )

                total_completed_debates += 1
        else:
            logger.info(f"All debates already complete for {model}")

    logger.info("All redteam v2 self-debates completed")

if __name__ == "__main__":
    main()
