import itertools
import configparser
from magif.agent.agent import Agent
from magif.environment.agent_pool import AgentPool
from magif.environment.environment import Environment
from magif.utils.utils import Mode, generate_agent_name, normalize_path
from magif.utils.data_object import DataObject
import logging
import os

'''
In the experiment modeled after "Axelrod's tournament," five agents, each autoformalized with rules for a distinct game,
are loaded. Copies of these agents then compete against each other using various predefined strategies. The tournament 
results show which strategy proves most effective on average for each game.
'''


def main():
	logging.debug('Experiment 3')
	config = configparser.ConfigParser()

	# Step 1: Read configuration
	config.read(normalize_path("DATA/CONFIG/experiment_3.ini"))

	# Step 2: Extract configuration parameters
	OUT_DIR = config.get("Paths", "OUT_DIR")
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	strategies_path = normalize_path(config.get("Paths", "STRATEGIES_PATH"))
	agents_path = normalize_path(config.get("Paths", "AGENTS_PATH"))
	num_rounds = config.getint("Params", "num_rounds")

	# Step 3: Read agents and strategies
	strategies = [os.path.join(strategies_path, strat_name) for strat_name in os.listdir(strategies_path)]
	agent_jsons_dir = [os.path.join(agents_path, agent) for agent in os.listdir(agents_path)]

	match_maker = lambda agents: list(itertools.combinations_with_replacement(agents, 2))

	# Step 4: Run the tournament for each agent (game definition)
	experiment_name = "experiment_3"
	for agent_json_dir in agent_jsons_dir:

		agent_pool = AgentPool()
		for strategy_path in strategies:
			for agent_json in os.listdir(agent_json_dir):
				agent = Agent(agent_json=os.path.join(agent_json_dir,agent_json), autoformalization_on=False)
				strategy_data = DataObject(rules_path=strategy_path, mode=Mode.RULES_PATH)
				agent.set_strategy(strategy_data)
				agent.name = generate_agent_name(3)
				agent_pool.add_agent(agent)

		tournament = Environment(
			agent_pool=agent_pool,
			num_rounds=num_rounds,
			match_maker=match_maker
		)

		# Run the tournament
		tournament.play_tournament()
		winners = tournament.get_winners()

		exp_dir = os.path.join(OUT_DIR, experiment_name)
		agent_name = agent_json_dir
		tournament.log_tournament(experiment_dir=exp_dir, tournament_name=agent_name)
		# Print winners
		print("Winners are:")
		for winner in winners:
			print(f"Agent {winner.name} with strategy {winner.strategy_name} and payoff {winner.mind.get_total_payoff()}")

		agent_pool.clean_agents()


if __name__ == "__main__":
	main()
