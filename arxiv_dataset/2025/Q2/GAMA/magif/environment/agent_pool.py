from magif.agent.agent import Agent
from magif.utils.utils import AgentStatus


class AgentPool:
	"""
    A class to manage a pool of agents participating in the tournament.

    Attributes:
        valid_agents (list[Agent]): A list to store agents with a status of CORRECT.
        invalid_agents (list[Agent]): A list to store agents with any status other than CORRECT.
    """

	def __init__(self):
		"""
        Initializes the AgentPool instance with empty valid and invalid agent lists.
        """
		self.valid_agents: list[Agent] = []
		self.invalid_agents: list[Agent] = []

	def add_agent(self, agent: Agent):
		"""
        Adds an agent to the appropriate pool based on its status.

        Args:
            agent (Agent): The agent to be added to the pool.
        """
		if agent.status == AgentStatus.CORRECT:
			self.valid_agents.append(agent)
		else:
			self.invalid_agents.append(agent)

	def move_agent(self, agent: Agent):
		"""
        Moves an agent between the valid and invalid pools based on its updated status.

        If the agent's status is CORRECT and it is currently in the invalid pool,
        it is moved to the valid pool. Similarly, if the agent's status is not
        CORRECT and it is in the valid pool, it is moved to the invalid pool.

        Args:
            agent (Agent): The agent whose status has changed.
        """
		if agent.status == AgentStatus.CORRECT:
			if agent in self.invalid_agents:
				self.invalid_agents.remove(agent)
				self.valid_agents.append(agent)
		else:
			if agent in self.valid_agents:
				self.valid_agents.remove(agent)
				self.invalid_agents.append(agent)

	def clean_agents(self):
		for agent in self.valid_agents+self.invalid_agents:
			agent.release_solver()
		self.valid_agents = None
		self.invalid_agents = None

	def truncate_pool(self, num):
		self._truncate_list(self.valid_agents, num)
		self._truncate_list(self.invalid_agents, num)

	def _truncate_list(self, agent_list, num):
		valid_len = len(agent_list)
		if valid_len > num:
			for i in range(num, valid_len):
				agent_list[-1].reload_solver()
				agent_list.pop()

	def __str__(self) -> str:
		"""
        Returns a string representation of the agent pool, listing valid
        and invalid agents by name.

        Returns:
            str: A formatted string showing the names of valid and invalid agents.
        """
		valid_names = ", ".join([agent.name for agent in self.valid_agents])
		invalid_names = ", ".join([agent.name for agent in self.invalid_agents])
		return f"Valid Agents: {valid_names} | Invalid Agents: {invalid_names}"
