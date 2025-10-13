import os
import argparse
from agent import Agent
from environment import Environment
import parameters
from azure_openai_client import AzureOpenAIClient
from openrouter_client import OpenRouterClient
from kluster_ai_client import KlusterAIClient
from openai_client import OpenAIClient

def main():
    parser = argparse.ArgumentParser(
        description="Run the public goods game simulation with configurable API clients."
    )
    # --- Existing API Client Arguments ---
    api_group = parser.add_argument_group('Main API Client Configuration')
    api_group.add_argument('--api-provider', type=str, default=os.getenv('API_PROVIDER', 'azure'),
                           help='API provider for main agent LLM calls (azure, openrouter, kluster, openai).')
    api_group.add_argument('--model-name', type=str, default=None,
                           help='Model name for the main API client.')
    api_group.add_argument('--deployment-name', type=str, default=None,
                           help='Azure Deployment name (for Azure API).')
    api_group.add_argument('--azure-endpoint', type=str, default=None,
                           help='Endpoint for Azure OpenAI. Overrides AZURE_ENDPOINT environment variable if provided.')
    api_group.add_argument('--main-api-key', type=str, default=None,
                           help='API key for the main API client. Overrides environment variables.')

    summary_api_group = parser.add_argument_group('Summary API Client Configuration')
    summary_api_group.add_argument('--summary-api-provider', type=str, default='openai',
                                   help='API provider for summary LLM calls (azure, openrouter, kluster, openai).')
    summary_api_group.add_argument('--summary-model-name', type=str, default='gpt-3.5-turbo',
                                   help='Model name for the summary API client.')
    summary_api_group.add_argument('--summary-deployment-name', type=str, default=None,
                                   help='Azure Deployment name for summary API (if using Azure).')
    summary_api_group.add_argument('--summary-azure-endpoint', type=str, default=None,
                                   help='Endpoint for Azure OpenAI (summary client).')
    summary_api_group.add_argument('--summary-api-key', type=str, default=None,
                                   help='API key for the summary API client. Overrides environment variables.')

    # --- Results Filename Argument ---
    parser.add_argument('--results-filename', type=str, default=None,
                        help='Filename to save simulation results to. Dynamic filename if not provided.')

    # --- New Simulation Parameters ---
    parser.add_argument('--num-rounds', type=int, default=parameters.NUM_ROUNDS,
                        help=f"Number of rounds for the simulation (default: {parameters.NUM_ROUNDS})")
    parser.add_argument('--num-agents', type=int, default=parameters.NUM_AGENTS,
                        help=f"Number of agents in the simulation (default: {parameters.NUM_AGENTS})")

    args = parser.parse_args()

    # Override parameters with command-line values (if provided)
    parameters.NUM_ROUNDS = args.num_rounds
    parameters.NUM_AGENTS = args.num_agents

    # --- Initialize Main API Client ---
    api_provider = args.api_provider.lower()
    model_name = args.model_name
    deployment_name = args.deployment_name
    azure_endpoint = args.azure_endpoint or os.getenv('AZURE_ENDPOINT')
    main_api_key = args.main_api_key

    if api_provider == 'azure':
        api_key = main_api_key or os.getenv('AZURE_API_KEY')
        deployment_name = deployment_name or os.getenv('AZURE_DEPLOYMENT_NAME')
        if not all([api_key, azure_endpoint, deployment_name]):
            raise Exception("Missing Azure OpenAI credentials for main client.")
        api_client = AzureOpenAIClient(api_key=api_key, endpoint=azure_endpoint,
                                       deployment_name=deployment_name)
    elif api_provider == 'openrouter':
        api_key = main_api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise Exception("Missing OpenRouter API key for main client.")
        api_client = OpenRouterClient(api_key=api_key, model_name=model_name or 'deepseek/deepseek-chat')
    elif api_provider == 'kluster':
        api_key = main_api_key or os.getenv('KLUSTER_API_KEY')
        if not api_key:
            raise Exception("Missing KluSter API key for main client.")
        api_client = KlusterAIClient(api_key=api_key, model_name=model_name or 'deepseek-ai/DeepSeek-R1')
    elif api_provider == 'openai':
        api_key = main_api_key or os.getenv('OPENAI_API_KEY')
        model_name = model_name or os.getenv('OPENAI_MODEL_NAME')
        if not all([api_key, model_name]):
            raise Exception("Missing OpenAI credentials for main client.")
        api_client = OpenAIClient(api_key=api_key, model_name=model_name, reasoning_effort='low')
    else:
        raise Exception(f"Unsupported API provider '{api_provider}' for main client.")

    # --- Initialize Summary API Client ---
    summary_api_provider = args.summary_api_provider.lower()
    summary_model_name = args.summary_model_name
    summary_deployment_name = args.summary_deployment_name
    summary_azure_endpoint = args.summary_azure_endpoint or os.getenv('AZURE_ENDPOINT') or azure_endpoint
    summary_api_key = args.summary_api_key

    if summary_api_provider == 'openai':
        summary_api_key = summary_api_key or os.getenv('OPENAI_API_KEY')
        summary_api_client = OpenAIClient(api_key=summary_api_key, model_name=summary_model_name,
                                          reasoning_effort='low')
    elif summary_api_provider == 'azure':
        summary_api_key = summary_api_key or os.getenv('AZURE_API_KEY')
        summary_deployment_name = summary_deployment_name or os.getenv('AZURE_DEPLOYMENT_NAME') or deployment_name
        if not all([summary_api_key, summary_azure_endpoint, summary_deployment_name]):
            raise Exception("Missing Azure OpenAI credentials for summary client.")
        summary_api_client = AzureOpenAIClient(api_key=summary_api_key, endpoint=summary_azure_endpoint,
                                               deployment_name=summary_deployment_name)
    elif summary_api_provider == 'openrouter':
        summary_api_key = summary_api_key or os.getenv('OPENROUTER_API_KEY')
        if not summary_api_key:
            raise Exception("Missing OpenRouter API key for summary client.")
        summary_api_client = OpenRouterClient(api_key=summary_api_key, model_name=summary_model_name or 'deepseek/deepseek-chat')
    elif summary_api_provider == 'kluster':
        summary_api_key = summary_api_key or os.getenv('KLUSTER_API_KEY')
        if not summary_api_key:
            raise Exception("Missing KluSter API key for summary client.")
        summary_api_client = KlusterAIClient(api_key=summary_api_key, model_name=summary_model_name or 'deepseek-ai/DeepSeek-R1')
    else:
        raise Exception(f"Unsupported API provider '{summary_api_provider}' for summary client.")

    results_filename = args.results_filename

    # --- Initialize agents using the (possibly overridden) parameters ---
    agents = [
        Agent(agent_id=i, api_client=api_client, summary_api_client=summary_api_client)
        for i in range(parameters.NUM_AGENTS)
    ]

    # --- Initialize and run the environment ---
    env = Environment(agents)
    env.run_simulation()

    # --- Save results ---
    if parameters.SAVE_RESULTS:
        env.save_results_to_filename(
            results_filename,
            model_name,  # Main model name for filename
            parameters.NUM_AGENTS,
            parameters.NUM_ROUNDS
        )

    if hasattr(api_client, 'get_total_cost'):
        total_cost = api_client.get_total_cost()
        print(f"\nTotal API Cost (Main Client - {api_provider} - {model_name or 'default'}): ${total_cost:.6f}")
    if hasattr(summary_api_client, 'get_total_cost'):
        summary_total_cost = summary_api_client.get_total_cost()
        print(f"Total API Cost (Summary Client - {summary_api_provider} - {summary_model_name or 'default'}): ${summary_total_cost:.6f}")

if __name__ == "__main__":
    main()
