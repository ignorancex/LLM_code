
"""Specific configuration for the FIRE framewrok."""

from common import shared_config

################################################################################
#                              SEARCH SETTINGS
# search_type: str = Google Search API used. Choose from ['serper'].
# num_searches: int = Number of results to show per search.
################################################################################
search_type = 'serper'
num_searches = 3

################################################################################
#                               FIRE SETTINGS
# max_steps: int = maximum number of break-down steps for factuality check.
# max_retries: int = maximum number of retries when fact checking fails.
# max_tolerance: int = maximum number of repetitive searches when fact checking.
# diverse_prompt: bool = whether to use diverse prompts for fact checking.
################################################################################
max_steps = 5
max_retries = 10
max_tolerance = 2
diverse_prompt = False