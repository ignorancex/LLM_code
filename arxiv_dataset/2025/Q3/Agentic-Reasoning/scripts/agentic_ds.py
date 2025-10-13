import os
import sys

# Add the project root and scripts directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([project_root, scripts_dir])

from agentic_research import STORMARRunnerArguments, STORMARRunner, STORMARLMConfigs
from agentic_research.rm import YouRM, DuckDuckGoSearchRM, BingSearch
import dspy
import litellm
import json
from utils.remote_llm import setup_dspy_model
def agentic_ds(topic: str, model_name: str, search_engine: str, tools: list = []):
    
    # Set LiteLLM to debug mode using the new recommended approach
    os.environ["LITELLM_LOG"] = "DEBUG"

    # Initialize configurations
    lm_configs = STORMARLMConfigs()

    # Set up models
    llm = setup_dspy_model(model_name=model_name)

    # Configure LM components
    lm_configs.set_conv_simulator_lm(llm)
    lm_configs.set_question_asker_lm(llm)
    lm_configs.set_outline_gen_lm(llm)
    lm_configs.set_article_gen_lm(llm)
    lm_configs.set_article_polish_lm(llm)

    # Set up engine arguments with default values
    engine_args = STORMARRunnerArguments(
        output_dir='./research_output'
    )

    # Initialize retrieval model and runner
    if search_engine == 'ydc':
        rm = YouRM(
            ydc_api_key=os.getenv('YDC_API_KEY'), 
            k=engine_args.search_top_k
        )
    elif search_engine == 'ddg':
        rm = DuckDuckGoSearchRM(
            k=engine_args.search_top_k
        )
    elif search_engine == 'bing':
        rm = BingSearch(
            bing_search_api_key=os.getenv('BING_SUBSCRIPTION_KEY'),
            k=engine_args.search_top_k
        )

    runner = STORMARRunner(engine_args, lm_configs, rm, tools = tools)

    # Define topic directly instead of using input()
    topic = topic
    print(f"Starting research on topic: {topic}")

    try:
        # Run the research process with progress updates
        print("Starting research process...")
        runner.run(
            topic=topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
        print("Research process completed")

        print("Running post-processing...")
        runner.post_run()
        print("Post-processing completed")

        print("Generating summary...")
        runner.summary()
        print("Summary completed")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
