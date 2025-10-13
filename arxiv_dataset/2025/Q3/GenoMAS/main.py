import asyncio
import os

from agents import PIAgent, GEOAgent, TCGAAgent, StatisticianAgent, CodeReviewerAgent, DomainExpertAgent
from core.au_generator import generate_and_save_all
from core.context import ActionUnit
from core.prompt_loader import load_prompts
from environment import Environment
from utils.config import setup_arg_parser
from utils.llm import get_llm_client, get_role_specific_args
from utils.logger import Logger
from utils.utils import extract_function_code, get_question_pairs, check_slow_inference


async def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    model = args.model
    scaler = 1.0
    if check_slow_inference(model, args.thinking):
        scaler = 6.0 if ('deepseek' in model.lower() and '671b' in model.lower()) else 3.0
    elif ('deepseek' in model.lower() and 'v3' in model.lower()):
        scaler = 3.0
    args.max_time = args.max_time * scaler
    task_info_file = './metadata/task_info.json'
    all_pairs = get_question_pairs(task_info_file)
    # Set the path containing input data on different devices. Change to yours.
    in_data_root = '/media/techt/DATA' if os.path.exists('/media/techt/DATA') else '../DATA'
    tcga_root = os.path.join(in_data_root, 'TCGA')
    output_root = './output/'
    version = args.version
    log_file = os.path.join(output_root, f"log_{version}.txt")
    
    logger = Logger(log_file=log_file, max_msg_length=getattr(args, 'max_log_msg_chars', 10000))
    
    # Role-specific clients
    pi_client = get_llm_client(get_role_specific_args(args, 'pi'), logger)
    statistician_client = get_llm_client(get_role_specific_args(args, 'statistician'), logger)
    data_engineer_client = get_llm_client(get_role_specific_args(args, 'data-engineer'), logger)
    code_reviewer_client = get_llm_client(get_role_specific_args(args, 'code-reviewer'), logger)
    domain_expert_client = get_llm_client(get_role_specific_args(args, 'domain-expert'), logger)
    planning_client = get_llm_client(get_role_specific_args(args, 'planning'), logger)
    
    # Handle auto-generation of action units if requested
    use_autogen = False
    if args.auto_generate_action_units:
        logger.info("Auto-generating Action Unit prompts from guidelines...")
        
        # Generate and save prompts
        generated_files = await generate_and_save_all(
            planning_client, 
            logger,
        )
        
        if generated_files:
            logger.info("\nGenerated Action Unit prompt files:")
            for agent_id, filepath in generated_files.items():
                logger.info(f"  {agent_id}: {filepath}")
            
            # Human-in-the-loop refinement
            if not args.non_interactive:
                print("\n" + "="*60)
                print("Action Unit prompts have been generated.")
                print("You can now edit the files listed above to refine them.")
                print("Press Enter when you're done editing...")
                print("="*60)
                input()

            # Confirmation
            if args.non_interactive:
                # In non-interactive mode, skip confirmation and use generated prompts by default
                use_autogen = True
                logger.info("Non-interactive mode: auto-confirmed use of auto-generated Action Unit prompts")
            else:
                print("\nDo you want to use the generated prompts?")
                confirm = input("[Y/n]: ").strip().lower() or 'y'
                
                if confirm == 'y':
                    use_autogen = True
                    logger.info("Using auto-generated Action Unit prompts")
                else:
                    logger.info("Using base Action Unit prompts (ignoring generated files)")
        else:
            logger.warning("Failed to generate any prompts, using base prompts")
    
    # Load prompts (either base or with autogen overlay)
    prompts = load_prompts(use_autogen=use_autogen)

    prep_tool_file = "./tools/preprocess.py"
    with open(prep_tool_file, 'r') as file:
        prep_tools_code_full = file.read()
    geo_selected_code = extract_function_code(prep_tool_file,
                                              ["validate_and_save_cohort_info", "geo_select_clinical_features",
                                               "preview_df"])
    geo_tools = {"full": prompts.PREPROCESS_TOOLS.format(tools_code=prep_tools_code_full),
                 "domain_focus": prompts.PREPROCESS_TOOLS.format(tools_code=geo_selected_code)}
    # Create agents once
    geo_action_units = [
        ActionUnit("Initial Data Loading", prompts.GEO_DATA_LOADING_PROMPT),
        ActionUnit("Dataset Analysis and Clinical Feature Extraction", prompts.GEO_FEATURE_ANALYSIS_EXTRACTION_PROMPT),
        ActionUnit("Gene Data Extraction", prompts.GEO_GENE_DATA_EXTRACTION_PROMPT),
        ActionUnit("Gene Identifier Review", prompts.GEO_GENE_IDENTIFIER_REVIEW_PROMPT),
        ActionUnit("Gene Annotation", prompts.GEO_GENE_ANNOTATION_PROMPT),
        ActionUnit("Gene Identifier Mapping", prompts.GEO_GENE_IDENTIFIER_MAPPING_PROMPT),
        ActionUnit("Data Normalization and Linking", prompts.GEO_DATA_NORMALIZATION_LINKING_PROMPT),
        ActionUnit("TASK COMPLETED", prompts.TASK_COMPLETED_PROMPT)
    ]

    geo_agent = GEOAgent(
        client=data_engineer_client,
        logger=logger,
        role_prompt=prompts.GEO_ROLE_PROMPT,
        guidelines=prompts.GEO_GUIDELINES,
        tools=geo_tools,
        setups='',
        action_units=geo_action_units,
        args=args,
        planning_client=planning_client
    )

    tcga_selected_code = extract_function_code(prep_tool_file, ["tcga_get_relevant_filepaths",
                                                                "tcga_convert_trait",
                                                                "tcga_convert_age",
                                                                "tcga_convert_gender",
                                                                "tcga_select_clinical_features",
                                                                "preview_df"])
    tcga_tools = {"full": prompts.PREPROCESS_TOOLS.format(tools_code=prep_tools_code_full),
                  "domain_focus": prompts.PREPROCESS_TOOLS.format(tools_code=tcga_selected_code)}
    # Inline subdirectory listing for TCGA initial data loading
    subdirs_prompt = f"\nAvailable subdirectories under `tcga_root_dir`:\n{os.listdir(tcga_root)}\n"

    tcga_action_units = [
        ActionUnit(
            "Initial Data Loading",
            prompts.TCGA_DATA_LOADING_PROMPT + subdirs_prompt
        ),
        ActionUnit("Find Candidate Demographic Features", prompts.TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT),
        ActionUnit("Select Demographic Features", prompts.TCGA_SELECT_DEMOGRAPHIC_PROMPT),
        ActionUnit("Feature Engineering and Validation", prompts.TCGA_FEATURE_ENGINEERING_PROMPT),
        ActionUnit("TASK COMPLETED", prompts.TASK_COMPLETED_PROMPT)
    ]

    tcga_agent = TCGAAgent(
        client=data_engineer_client,
        logger=logger,
        role_prompt=prompts.TCGA_ROLE_PROMPT,
        guidelines=prompts.TCGA_GUIDELINES,
        tools=tcga_tools,
        setups='',
        action_units=tcga_action_units,
        args=args,
        planning_client=planning_client
    )

    stat_tool_file = "./tools/statistics.py"
    with open(stat_tool_file, 'r') as file:
        stat_tools_code_full = file.read()
    stat_selected_code = stat_tools_code_full
    stat_tools = {"full": prompts.STATISTICIAN_TOOLS.format(tools_code=stat_tools_code_full),
                  "domain_focus": prompts.STATISTICIAN_TOOLS.format(tools_code=stat_selected_code)}

    stat_action_units = [
        ActionUnit("Unconditional One-step Regression",
                   prompts.UNCONDITIONAL_ONE_STEP_PROMPT),
        ActionUnit("Conditional One-step Regression",
                   prompts.CONDITIONAL_ONE_STEP_PROMPT),
        ActionUnit("Two-step Regression", prompts.TWO_STEP_PROMPT),
        ActionUnit("TASK COMPLETED", prompts.TASK_COMPLETED_PROMPT)
    ]

    statistician = StatisticianAgent(client=statistician_client,
                                     logger=logger,
                                     role_prompt=prompts.STATISTICIAN_ROLE_PROMPT,
                                     guidelines=prompts.STATISTICIAN_GUIDELINES,
                                     tools=stat_tools,
                                     setups='',
                                     action_units=stat_action_units,
                                     args=args,
                                     planning_client=planning_client
                                     )

    agents = [
        PIAgent(client=pi_client, logger=logger, args=args),
        geo_agent,
        tcga_agent,
        statistician,
        CodeReviewerAgent(client=code_reviewer_client, logger=logger, args=args),
        DomainExpertAgent(client=domain_expert_client, logger=logger, args=args)
    ]

    env = Environment(logger=logger, agents=agents, args=args)

    await env.run(all_pairs, in_data_root, output_root, version, task_info_file)


if __name__ == "__main__":
    asyncio.run(main())
