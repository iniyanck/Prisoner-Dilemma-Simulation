"""
DataManager class for saving/loading agents and simulation results.
Improvements:
- Added type hints
- Added/expanded docstrings
- Used logging instead of print
- Sketched out agent serialization/deserialization using neural network methods
- Basic file I/O for agent data (JSON)
"""

import logging
import json
from typing import Any, Optional
from core.agent import Agent

class DataManager:
    def __init__(self) -> None:
        """Initialize the DataManager."""
        pass

    def save_agent(self, agent: Agent, filename: str) -> None:
        """
        Save an agent's neural network weights and metadata to a file.
        Args:
            agent (Agent): The agent to save.
            filename (str): Path to the file.
        """
        try:
            data = {
                'id': agent.id,
                'score': agent.score,
                'competition_wins': agent.competition_wins,
                'nn': agent.nn.serialize()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
            logging.info(f"Agent {agent.id} saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save agent: {e}")

    def load_agent(self, filename: str) -> Optional[Agent]:
        """
        Load an agent from a file (restores neural network weights and metadata).
        Args:
            filename (str): Path to the file.
        Returns:
            Optional[Agent]: The loaded agent, or None if failed.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            from core.neural_network import NeuralNetwork
            from config.neural_net_config import NeuralNetConfig
            input_dim = NeuralNetConfig.INPUT_DIM
            agent = Agent(input_dim, agent_id=data['id'])
            weights, biases = NeuralNetwork.deserialize(data['nn'])
            agent.nn.set_weights_biases(weights, biases)
            agent.score = data.get('score', 0)
            agent.competition_wins = data.get('competition_wins', 0)
            logging.info(f"Agent {agent.id} loaded from {filename}")
            return agent
        except Exception as e:
            logging.error(f"Failed to load agent: {e}")
            return None