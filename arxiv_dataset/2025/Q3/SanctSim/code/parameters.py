
# parameters.py

"""
This file contains all the constants and configuration settings for the simulation.
These parameters are based on the experimental design described in:
"The Competitive Advantage of Sanctioning Institutions" by Gürerk, Irlenbusch, and Rockenbach.
"""

# Simulation Settings
NUM_AGENTS = 7        # Total number of participants in the experiment
NUM_ROUNDS = 15        # Total number of periods (rounds) in the experiment
SEED = 42               # Random seed for reproducibility (optional)

# Initial Endowments
INITIAL_TOKENS = 1000   # Initial tokens assigned to each agent at the beginning of the experiment
ENDOWMENT_STAGE_1 = 20  # Tokens endowed to each agent at the beginning of Stage 1 in each round
ENDOWMENT_STAGE_2 = 20  # Tokens endowed to each agent at the beginning of Stage 2 in each round

# Public Goods Game Parameters
PUBLIC_GOOD_MULTIPLIER = 1.6  # The multiplication factor for the sum of contributions
MIN_CONTRIBUTION = 0          # Minimum amount an agent can contribute to the public good
MAX_CONTRIBUTION = ENDOWMENT_STAGE_1  # Maximum amount an agent can contribute (entire endowment)

# Punishment and Reward Settings (Applicable in Sanctioning Institution)
MAX_PUNISHMENT_TOKENS = 20        # Maximum number of tokens an agent can assign in Stage 2
PUNISHMENT_EFFECT = 3            # Each negative token reduces the punished agent's payoff by 3 tokens
PUNISHMENT_COST = 1               # Each negative token costs the punishing agent 1 token
REWARD_EFFECT = 1                 # Each positive token increases the rewarded agent's payoff by 1 token
REWARD_COST = 1                   # Each positive token costs the rewarding agent 1 token

# Institution Types
INSTITUTION_TYPES = ['SFI', 'SI']  # 'SFI' = Sanction-Free Institution, 'SI' = Sanctioning Institution

# Decision Parameters (These can be adjusted to model different agent behaviors)
# For example, probability thresholds for choosing institutions or strategies
DEFAULT_STRATEGY = 'conditional_cooperator'  # Default behavior strategy for agents
STRATEGY_PARAMS = {
    'threshold_cooperation': 0.5,  # Threshold for cooperating based on others' cooperation
    'punishment_tolerance': 0.1,   # Tolerance level for others' defection before punishing
}

# Group Size Effects (Optional)
# MPCR (Marginal Per Capita Return) can vary with group size if implemented
VARIABLE_MPCR = False             # Set to True if MPCR depends on group size
MPCR_BASE = PUBLIC_GOOD_MULTIPLIER  # Base MPCR value when VARIABLE_MPCR is False

# Information Settings
ANONYMITY = True                  # Agents remain anonymous; no tracking of identities across rounds
DISPLAY_PAST_ACTIONS = 5          # Number of past rounds shown to agents (if implementing history display)

# Miscellaneous Settings
SAVE_RESULTS = True               # Whether to save the results to a file after the simulation
RESULTS_FILENAME = 'simulation_results.json'  # Default filename for saving results

# Debugging and Logging Settings
VERBOSE = True                   # If True, print detailed logs during the simulation
LOG_FILENAME = 'simulation_log.txt'  # Log file to record simulation details (if needed)

# Note:
# These parameters are intended to closely match the experimental conditions described in the paper.
# Adjustments can be made to explore different scenarios or to test the robustness of the simulation.