import itertools
import configparser
from magif.agent.agent import Agent
from magif.environment.agent_pool import AgentPool
from magif.environment.environment import Environment
from magif.utils.utils import read_file, Mode, generate_agent_name, normalize_path
from magif.utils.data_object import DataObject
from lms.gpt4 import GPT4
from lms.claude import Claude
import logging
import os

'''
In this experiment, we demonstrate the autoformalization of strategies, using a game solver 
and the tit-for-tat strategy as examples. 
'''


def main():
	logging.debug('Experiment 4')
	config = configparser.ConfigParser()

	# Step 1: Read configuration
	config.read(normalize_path("DATA/CONFIG/experiment_4.ini"))

	# Step 2: Extract configuration parameters
	OUT_DIR = config.get("Paths", "OUT_DIR")
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	# Extract file paths
	strategies_path = normalize_path(config.get("Paths", "STRATEGIES_PATH"))
	agent_json = normalize_path(config.get("Paths", "AGENT_PATH"))
	feedback_template_path = normalize_path(config.get("Paths", "FEEDBACK_TEMPLATE_PATH"))
	strategy_template_path = normalize_path(config.get("Paths", "STRATEGY_PROMPT"))

	# Extract experiment parameters
	num_rounds = config.getint("Params", "num_rounds")
	num_agents = config.getint("Params", "num_agents")
	target_payoffs = config.get("Params", "target_payoffs")
	target_payoffs = [int(payoff) for payoff in target_payoffs.split(";")]
	max_attempts = config.getint("Params", "max_attempts")

	# Step 3: Run the tournament providing the path to strategies descriptions
	for llm_name, llm in zip(["gpt4", "claude35"],[GPT4, Claude]):
		experiment_name = "experiment_4_"+llm_name
		for i in range(5):
			agent_pool = AgentPool()
			for i in range(num_agents):
				for strategy_path in sorted(os.listdir(strategies_path)):
					strategy_name = strategy_path.split(os.path.sep)[-1].replace(".txt", "")
					strategy_desc = read_file(os.path.join(strategies_path,strategy_path))
					prompt = read_file(strategy_template_path).format(strategy_description=strategy_desc)

					agent = Agent(agent_json=agent_json, max_attempts=max_attempts)
					strategy_data = DataObject(nl_description=strategy_desc, instruction_prompt=prompt,
											   feedback_prompt=read_file(feedback_template_path),
											   mode=Mode.AUTOFORMALIZATION, name=strategy_name)
					agent.set_strategy(strategy_data)
					agent.name = generate_agent_name(3)
					agent_pool.add_agent(agent)

			# Record original number of agents
			valid_agents_num = len(agent_pool.valid_agents)

			# Add copies of an agent with tat-for-tit strategy
			for i in range(valid_agents_num):
				agent = agent_pool.valid_agents[i]
				clone_game_data = DataObject(rules_string=agent.game.game_rules, mode=Mode.RULES_STRING)
				clone_strategy_data = DataObject(rules_path="../DATA/STRATEGIES/anti-tit-for-tat.pl", mode=Mode.RULES_PATH)
				clone = Agent(game_data=clone_game_data, strategy_data=clone_strategy_data, autoformalization_on=False)
				agent_pool.add_agent(clone)

			# Create matching: (original_agent, clone)
			match_maker = lambda agents: [(agents[i], agents[i + valid_agents_num]) for i in range(valid_agents_num)]

			tournament = Environment(
				agent_pool=agent_pool,
				num_rounds=num_rounds,
				match_maker=match_maker,
				target_payoffs=target_payoffs
			)

			# Run the tournament
			tournament.play_tournament()

			# Remove clones for the evaluation
			agent_pool.truncate_pool(valid_agents_num)

			# Validate results
			winners = tournament.get_winners()

			# Log the results
			tournament_dir = os.path.join(OUT_DIR, experiment_name)
			tournament.log_tournament(tournament_dir, "strategies")

			# Print winners
			print("Winners are:")
			for winner in winners:
				print(f"Agent {winner.name} with strategy {winner.strategy_name} and payoff {winner.mind.get_total_payoff()}")

			agent_pool.clean_agents()


if __name__ == "__main__":
	main()
