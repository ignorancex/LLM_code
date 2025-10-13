"""
Dynamic prompt loader that supports both base and auto-generated prompts.
"""
import importlib
import importlib.util


class PromptContainer:
    """Container for all prompts used in the system."""
    
    def __init__(self):
        # Shared prompts
        self.PREPROCESS_TOOLS = None
        self.STATISTICIAN_TOOLS = None
        self.MULTI_STEP_SETUPS = None
        self.TASK_COMPLETED_PROMPT = None
        self.CODE_INDUCER = None
        
        # GEO prompts
        self.GEO_ROLE_PROMPT = None
        self.GEO_GUIDELINES = None
        self.GEO_DATA_LOADING_PROMPT = None
        self.GEO_FEATURE_ANALYSIS_EXTRACTION_PROMPT = None
        self.GEO_GENE_DATA_EXTRACTION_PROMPT = None
        self.GEO_GENE_IDENTIFIER_REVIEW_PROMPT = None
        self.GEO_GENE_ANNOTATION_PROMPT = None
        self.GEO_GENE_IDENTIFIER_MAPPING_PROMPT = None
        self.GEO_DATA_NORMALIZATION_LINKING_PROMPT = None
        
        # TCGA prompts
        self.TCGA_ROLE_PROMPT = None
        self.TCGA_GUIDELINES = None
        self.TCGA_DATA_LOADING_PROMPT = None
        self.TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT = None
        self.TCGA_SELECT_DEMOGRAPHIC_PROMPT = None
        self.TCGA_FEATURE_ENGINEERING_PROMPT = None
        
        # Statistician prompts
        self.STATISTICIAN_ROLE_PROMPT = None
        self.STATISTICIAN_GUIDELINES = None
        self.UNCONDITIONAL_ONE_STEP_PROMPT = None
        self.CONDITIONAL_ONE_STEP_PROMPT = None
        self.TWO_STEP_PROMPT = None


def load_prompts(use_autogen: bool = False) -> PromptContainer:
    """
    Load prompts from the prompts package, optionally overlaying auto-generated versions.
    
    Args:
        use_autogen: If True, overlay auto-generated prompts if they exist
        
    Returns:
        PromptContainer with all loaded prompts
    """
    container = PromptContainer()
    
    # Load base prompts
    shared = importlib.import_module('prompts.shared')
    geo = importlib.import_module('prompts.GEO')
    tcga = importlib.import_module('prompts.TCGA')
    statistics = importlib.import_module('prompts.statistics')
    
    # Load shared prompts (never auto-generated)
    container.PREPROCESS_TOOLS = shared.PREPROCESS_TOOLS
    container.STATISTICIAN_TOOLS = shared.STATISTICIAN_TOOLS
    container.MULTI_STEP_SETUPS = shared.MULTI_STEP_SETUPS
    container.TASK_COMPLETED_PROMPT = shared.TASK_COMPLETED_PROMPT
    container.CODE_INDUCER = shared.CODE_INDUCER
    
    # Load GEO prompts
    container.GEO_ROLE_PROMPT = geo.GEO_ROLE_PROMPT
    container.GEO_GUIDELINES = geo.GEO_GUIDELINES
    container.GEO_DATA_LOADING_PROMPT = geo.GEO_DATA_LOADING_PROMPT
    container.GEO_FEATURE_ANALYSIS_EXTRACTION_PROMPT = geo.GEO_FEATURE_ANALYSIS_EXTRACTION_PROMPT
    container.GEO_GENE_DATA_EXTRACTION_PROMPT = geo.GEO_GENE_DATA_EXTRACTION_PROMPT
    container.GEO_GENE_IDENTIFIER_REVIEW_PROMPT = geo.GEO_GENE_IDENTIFIER_REVIEW_PROMPT
    container.GEO_GENE_ANNOTATION_PROMPT = geo.GEO_GENE_ANNOTATION_PROMPT
    container.GEO_GENE_IDENTIFIER_MAPPING_PROMPT = geo.GEO_GENE_IDENTIFIER_MAPPING_PROMPT
    container.GEO_DATA_NORMALIZATION_LINKING_PROMPT = geo.GEO_DATA_NORMALIZATION_LINKING_PROMPT
    
    # Load TCGA prompts
    container.TCGA_ROLE_PROMPT = tcga.TCGA_ROLE_PROMPT
    container.TCGA_GUIDELINES = tcga.TCGA_GUIDELINES
    container.TCGA_DATA_LOADING_PROMPT = tcga.TCGA_DATA_LOADING_PROMPT
    container.TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT = tcga.TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT
    container.TCGA_SELECT_DEMOGRAPHIC_PROMPT = tcga.TCGA_SELECT_DEMOGRAPHIC_PROMPT
    container.TCGA_FEATURE_ENGINEERING_PROMPT = tcga.TCGA_FEATURE_ENGINEERING_PROMPT
    
    # Load Statistician prompts
    container.STATISTICIAN_ROLE_PROMPT = statistics.STATISTICIAN_ROLE_PROMPT
    container.STATISTICIAN_GUIDELINES = statistics.STATISTICIAN_GUIDELINES
    container.UNCONDITIONAL_ONE_STEP_PROMPT = statistics.UNCONDITIONAL_ONE_STEP_PROMPT
    container.CONDITIONAL_ONE_STEP_PROMPT = statistics.CONDITIONAL_ONE_STEP_PROMPT
    container.TWO_STEP_PROMPT = statistics.TWO_STEP_PROMPT
    
    # Overlay auto-generated prompts if requested and available
    if use_autogen:
        # Try to load GEO autogen
        try:
            geo_autogen = importlib.import_module('prompts.GEO_autogen')
            # Only overlay the action unit prompts, not role/guidelines
            for attr in ['GEO_DATA_LOADING_PROMPT', 'GEO_FEATURE_ANALYSIS_EXTRACTION_PROMPT',
                        'GEO_GENE_DATA_EXTRACTION_PROMPT', 'GEO_GENE_IDENTIFIER_REVIEW_PROMPT',
                        'GEO_GENE_ANNOTATION_PROMPT', 'GEO_GENE_IDENTIFIER_MAPPING_PROMPT',
                        'GEO_DATA_NORMALIZATION_LINKING_PROMPT']:
                if hasattr(geo_autogen, attr):
                    setattr(container, attr, getattr(geo_autogen, attr))
        except (ImportError, ModuleNotFoundError):
            pass
        
        # Try to load TCGA autogen
        try:
            tcga_autogen = importlib.import_module('prompts.TCGA_autogen')
            for attr in ['TCGA_DATA_LOADING_PROMPT', 'TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT',
                        'TCGA_SELECT_DEMOGRAPHIC_PROMPT', 'TCGA_FEATURE_ENGINEERING_PROMPT']:
                if hasattr(tcga_autogen, attr):
                    setattr(container, attr, getattr(tcga_autogen, attr))
        except (ImportError, ModuleNotFoundError):
            pass
        
        # Try to load statistics autogen
        try:
            stats_autogen = importlib.import_module('prompts.statistics_autogen')
            for attr in ['UNCONDITIONAL_ONE_STEP_PROMPT', 'CONDITIONAL_ONE_STEP_PROMPT',
                        'TWO_STEP_PROMPT']:
                if hasattr(stats_autogen, attr):
                    setattr(container, attr, getattr(stats_autogen, attr))
        except (ImportError, ModuleNotFoundError):
            pass
    
    return container