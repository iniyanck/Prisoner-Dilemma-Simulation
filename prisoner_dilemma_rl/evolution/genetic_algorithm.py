"""
GeneticAlgorithm class for evolving agent populations.
Improvements:
- Added type hints
- Added/expanded docstrings
- Ensured at least one agent is selected
- Optionally avoid same parent for crossover
- Error handling for edge cases
"""
import random
from typing import List
from core.agent import Agent
from core.neural_network import NeuralNetwork
from config.simulation_config import SimulationConfig
from config.neural_net_config import NeuralNetConfig

class GeneticAlgorithm:
    def __init__(self, input_dim: int):
        """
        Initialize the GeneticAlgorithm.
        Args:
            input_dim (int): Input dimension for agents' neural networks.
        """
        self.input_dim = input_dim

    def create_initial_population(self, num_agents: int) -> List[Agent]:
        """Create an initial population of agents."""
        return [Agent(self.input_dim) for _ in range(num_agents)]

    def select_top_agents(self, agents: List[Agent], selection_rate: float) -> List[Agent]:
        """
        Select the top-performing agents based on score.
        Args:
            agents (List[Agent]): List of agents (should be sorted by score).
            selection_rate (float): Fraction of agents to select.
        Returns:
            List[Agent]: Selected top agents.
        """
        num_selected = max(1, int(len(agents) * selection_rate))
        return agents[:num_selected]

    def reproduce(self, selected_agents: List[Agent], num_agents_to_create: int) -> List[Agent]:
        """
        Create new agents by crossover of selected agents' neural networks.
        Args:
            selected_agents (List[Agent]): Agents to use as parents.
            num_agents_to_create (int): Number of new agents to create.
        Returns:
            List[Agent]: New agents.
        """
        if not selected_agents:
            raise ValueError("No agents available for reproduction.")
        new_population = []
        while len(new_population) < num_agents_to_create:
            parent1, parent2 = random.sample(selected_agents, 2) if len(selected_agents) > 1 else (selected_agents[0], selected_agents[0])
            new_agent = Agent(self.input_dim)
            new_agent.nn = NeuralNetwork.crossover(parent1.nn, parent2.nn)
            new_population.append(new_agent)
        return new_population

    def mutate_population(self, population: List[Agent], mutation_rate: float) -> None:
        """
        Mutate the neural networks of all agents in the population.
        Args:
            population (List[Agent]): Agents to mutate.
            mutation_rate (float): Mutation rate.
        """
        for agent in population:
            agent.nn.mutate(mutation_rate)