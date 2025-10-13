"""
Action Unit generator that creates instruction prompts from guidelines.
"""
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from utils.llm import LLMClient
from utils.logger import Logger
from core.prompt_loader import load_prompts


@dataclass
class ActionUnitSpec:
    """Specification for an action unit to generate."""
    var_name: str  # e.g., 'GEO_DATA_LOADING_PROMPT'
    au_name: str   # e.g., 'Initial Data Loading'

async def generate_for_agent(
    agent_id: str,
    guidelines: str, 
    specs: List[ActionUnitSpec],
    client: LLMClient,
    logger: Optional[Logger] = None
) -> Dict[str, str]:
    """
    Generate action unit instruction prompts for a specific agent.
    
    Args:
        agent_id: Agent identifier ('GEO', 'TCGA', or 'STAT')
        guidelines: The agent's high-level guidelines
        specs: List of action unit specifications
        client: LLM client for generation
        logger: Optional logger
        
    Returns:
        Dictionary mapping au_name to generated instruction text
    """
    # Prepare the generation prompt
    system_prompt = """You are drafting Action Unit instructions for a programming agent in a gene expression analysis system.
Your instructions must be practical, specific, and directly implementable in Python code.
Output strict JSON only, no additional text."""
    
# Build the user prompt
    step_descriptions = "\n".join([
        f"- {spec.au_name}"
        for spec in specs
    ])
    
    au_names = [spec.au_name for spec in specs]
    json_schema = "{\n" + ",\n".join([f'  "{name}": "instruction text"' for name in au_names]) + "\n}"
    
    user_prompt = f"""Given these high-level guidelines for the {agent_id} agent:

{guidelines}

Generate detailed instruction strings for the following action units:
{step_descriptions}

Output a JSON object with exactly these keys (the Action Unit names):
{json.dumps(au_names)}

Use this JSON structure:
{json_schema}

Requirements:
1. Each instruction should be a clear, step-by-step guide for what the agent needs to do
2. Include specific technical details from the guidelines
3. Reference actual variables and file paths used in the codebase (e.g., out_data_file, clinical_data)
4. Number the steps within each instruction for clarity
5. Keep the language direct and action-oriented
6. For data processing steps, specify exact transformations needed
7. Ensure instructions are consistent with the overall workflow described in the guidelines

Generate the JSON now:"""
    
    # Call the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = await client.generate_completion(messages)
        content = response.get("content", "")
        
        # Parse JSON response
        # First try to extract JSON if wrapped in markdown
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        generated = json.loads(content)
        
# Validate that all required keys are present
        missing_keys = set(au_names) - set(generated.keys())
        if missing_keys:
            if logger:
                logger.warning(f"Missing keys in generated response: {missing_keys}")
            # Will fall back to base prompts for missing keys
        
        return generated
        
    except (json.JSONDecodeError, KeyError) as e:
        if logger:
            logger.error(f"Failed to parse LLM response for {agent_id}: {e}")
        return {}


def write_autogen_file(agent_id: str, prompts: Dict[str, str], output_dir: str = "prompts"):
    """
    Write auto-generated prompts to a Python module file.
    
    Args:
        agent_id: Agent identifier ('GEO', 'TCGA', or 'STAT')
        prompts: Dictionary mapping var_name to prompt text
        output_dir: Directory to write the file to
    """
    if agent_id == 'STAT':
        filename = 'statistics_autogen.py'
    else:
        filename = f'{agent_id}_autogen.py'
    
    filepath = os.path.join(output_dir, filename)
    
    # Generate file content
    lines = [
        '"""',
        f'Auto-generated Action Unit prompts for {agent_id} agent',
        f'Generated: {datetime.now().isoformat()}',
        '"""',
        '',
    ]
    
    # Add each prompt as a module-level constant
    for var_name, prompt_text in prompts.items():
        # Escape the prompt text properly
        escaped = prompt_text.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        lines.append(f'{var_name}: str = \\')
        lines.append('"""')
        lines.append(escaped)
        lines.append('"""')
        lines.append('')
    
    content = '\n'.join(lines)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(content)
    
    return filepath




async def generate_and_save_all(
    planning_client: LLMClient,
    logger: Optional[Logger] = None,
) -> Dict[str, str]:
    """
    Generate and save all action unit prompts for all agents.
    
    Args:
        planning_client: LLM client to use for generation
        logger: Optional logger
        
    Returns:
        Dictionary mapping agent_id to filepath of generated module
    """
    # Load base prompts for reference and fallback
    base_prompts = load_prompts(use_autogen=False)
    
    generated_files = {}
    
# GEO Agent
    geo_specs = [
        ActionUnitSpec('GEO_DATA_LOADING_PROMPT', 'Initial Data Loading'),
        ActionUnitSpec('GEO_FEATURE_ANALYSIS_EXTRACTION_PROMPT', 'Dataset Analysis and Clinical Feature Extraction'),
        ActionUnitSpec('GEO_GENE_DATA_EXTRACTION_PROMPT', 'Gene Data Extraction'),
        ActionUnitSpec('GEO_GENE_IDENTIFIER_REVIEW_PROMPT', 'Gene Identifier Review'),
        ActionUnitSpec('GEO_GENE_ANNOTATION_PROMPT', 'Gene Annotation'),
        ActionUnitSpec('GEO_GENE_IDENTIFIER_MAPPING_PROMPT', 'Gene Identifier Mapping'),
        ActionUnitSpec('GEO_DATA_NORMALIZATION_LINKING_PROMPT', 'Data Normalization and Linking'),
    ]
    
    geo_generated = await generate_for_agent(
        'GEO', base_prompts.GEO_GUIDELINES, geo_specs, planning_client, logger
    )
    
    # Only include generated keys; fallback is handled in prompt_loader
    geo_final = {}
    for spec in geo_specs:
        if spec.au_name in geo_generated:
            geo_final[spec.var_name] = geo_generated[spec.au_name]
    
    if geo_final:
        filepath = write_autogen_file('GEO', geo_final)
        generated_files['GEO'] = filepath
        if logger:
            logger.info(f"Generated GEO prompts: {filepath}")
    
# TCGA Agent
    tcga_specs = [
        ActionUnitSpec('TCGA_DATA_LOADING_PROMPT', 'Initial Data Loading'),
        ActionUnitSpec('TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT', 'Find Candidate Demographic Features'),
        ActionUnitSpec('TCGA_SELECT_DEMOGRAPHIC_PROMPT', 'Select Demographic Features'),
        ActionUnitSpec('TCGA_FEATURE_ENGINEERING_PROMPT', 'Feature Engineering and Validation'),
    ]
    
    tcga_generated = await generate_for_agent(
        'TCGA', base_prompts.TCGA_GUIDELINES, tcga_specs, planning_client, logger
    )
    
    # Only include generated keys; fallback is handled in prompt_loader
    tcga_final = {}
    for spec in tcga_specs:
        if spec.au_name in tcga_generated:
            tcga_final[spec.var_name] = tcga_generated[spec.au_name]
    
    if tcga_final:
        filepath = write_autogen_file('TCGA', tcga_final)
        generated_files['TCGA'] = filepath
        if logger:
            logger.info(f"Generated TCGA prompts: {filepath}")
    
# Statistician Agent
    stat_specs = [
        ActionUnitSpec('UNCONDITIONAL_ONE_STEP_PROMPT', 'Unconditional One-step Regression'),
        ActionUnitSpec('CONDITIONAL_ONE_STEP_PROMPT', 'Conditional One-step Regression'),
        ActionUnitSpec('TWO_STEP_PROMPT', 'Two-step Regression'),
    ]
    
    stat_generated = await generate_for_agent(
        'STAT', base_prompts.STATISTICIAN_GUIDELINES, stat_specs, planning_client, logger
    )
    
    # Only include generated keys; fallback is handled in prompt_loader
    stat_final = {}
    for spec in stat_specs:
        if spec.au_name in stat_generated:
            stat_final[spec.var_name] = stat_generated[spec.au_name]
    
    if stat_final:
        filepath = write_autogen_file('STAT', stat_final)
        generated_files['STAT'] = filepath
        if logger:
            logger.info(f"Generated Statistician prompts: {filepath}")
    
    return generated_files
