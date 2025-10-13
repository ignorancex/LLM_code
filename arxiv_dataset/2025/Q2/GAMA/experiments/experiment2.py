import configparser
from magif.agent.agent import Agent
from magif.environment.environment import Environment
from magif.environment.agent_pool import AgentPool
from magif.utils.utils import read_file, Mode, normalize_path
from magif.utils.data_object import DataObject
from lms.gpt4 import GPT4
from lms.claude import Claude
import logging
import os

'''
In this experiment, a dataset of 55 natural-language game-theoretic scenarios with non-numerical payoffs is autoformalized 
into formal logic specifications. To ensure syntactic correctness, a Prolog solver is employed for validation. For 
semantic validation,  a tournament is conducted in which each agent using a tit-for-tat strategy competes against its 
clone using an anti-tit-for-tat strategy. 
'''


def main():
	logging.debug('Experiment 2')
	config = configparser.ConfigParser()

	# Step 1: Read configuration
	config.read(normalize_path("DATA/CONFIG/experiment_2.ini"))

	# Step 2: Extract input and output paths
	GAME_DIR = normalize_path(config.get("Paths", "GAME_DIR"))
	OUT_DIR = config.get("Paths", "OUT_DIR")
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	# Step 3: Extract file paths
	template_path = normalize_path(config.get("Paths", "TEMPLATE_PATH"))
	feedback_template_path = normalize_path(config.get("Paths", "FEEDBACK_TEMPLATE_PATH"))
	strategy_path = normalize_path(config.get("Paths", "STRATEGY"))

	# Step 4: Extract experiment parameters
	num_agents = config.getint("Params", "num_agents")
	num_rounds = config.getint("Params", "num_rounds")
	max_attempts = config.getint("Params", "max_attempts")

	# Step 5: Load game descriptions
	games_descriptions = [(game_file[:-4], read_file(os.path.join(GAME_DIR,game_file))) for game_file in os.listdir(GAME_DIR)]

	# Step 6: Run the tournament for each LLM and each game description
	for llm_name, llm in zip(["gpt4", "claude35"],[GPT4, Claude]):
		experiment_name = "experiment_2_"+llm_name
		for game_name, game_desc in games_descriptions:
			agent_pool = AgentPool()
			for i in range(num_agents):
				prompt = read_file(template_path).format(game_description=game_desc)
				game_data = DataObject(nl_description=game_desc, instruction_prompt=prompt,
									   feedback_prompt=read_file(feedback_template_path),
									   mode=Mode.AUTOFORMALIZATION)
				strategy_data = DataObject(rules_path=strategy_path, mode=Mode.RULES_PATH)
				agent = Agent(game_data, strategy_data, llm=llm, max_attempts=max_attempts)
				agent_pool.add_agent(agent)

			# Record original number of agents
			valid_agents_num = len(agent_pool.valid_agents)

			# Add copies of an agent with tat-for-tit strategy
			for i in range(valid_agents_num):
				agent = agent_pool.valid_agents[i]
				clone_game_data = DataObject(rules_string=agent.game.game_rules, mode=Mode.RULES_STRING)
				clone_strategy_data = DataObject(rules_path=normalize_path("DATA/STRATEGIES/anti-tit-for-tat.pl"), mode=Mode.RULES_PATH)
				clone = Agent(game_data=clone_game_data, strategy_data=clone_strategy_data, autoformalization_on=False)
				agent_pool.add_agent(clone)

			# Create matching: (original_agent, clone)
			match_maker = lambda agents: [(agents[i], agents[i + valid_agents_num]) for i in range(valid_agents_num)]

			tournament = Environment(
				agent_pool=agent_pool,
				num_rounds=num_rounds,
				match_maker=match_maker
			)

			# Run the tournament
			tournament.play_tournament()

			# Remove clones for the evaluation
			agent_pool.truncate_pool(valid_agents_num)

			# Validate results
			winners = tournament.get_winners()

			# Log the results
			tournament_dir = os.path.join(OUT_DIR, experiment_name)
			tournament.log_tournament(tournament_dir, game_name)

			# Print winners
			print("Winners are:")
			for winner in winners:
				print(f"Agent {winner.name} with strategy {winner.strategy_name} and payoff {winner.mind.get_total_payoff()}")

			agent_pool.clean_agents()


if __name__ == "__main__":
	main()
