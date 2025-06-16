"""
Agent class for Prisoner's Dilemma RL Simulation.
Improvements:
- Added type hints
- Added/expanded docstrings
- Used Enum for choices
- Added input validation for choices
- Allowed optional agent_id for reproducibility/testing
- Used dataclass for history entries
"""

import uuid
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from core.neural_network import NeuralNetwork
import torch
from config.neural_net_config import NeuralNetConfig
from config.simulation_config import SimulationConfig

class Choice(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"

@dataclass
class Interaction:
    my_choice: Choice
    enemy_choice: Choice

class Agent:
    def __init__(self, input_dim: int, agent_id: Optional[str] = None):
        """
        Initialize an Agent with a neural network and unique ID.
        Args:
            input_dim (int): Input dimension for the neural network.
            agent_id (Optional[str]): Optional agent ID for reproducibility.
        """
        self.id: str = agent_id if agent_id else str(uuid.uuid4())
        self.nn: NeuralNetwork = NeuralNetwork(
            input_dim=input_dim,
            hidden_layers=NeuralNetConfig.HIDDEN_LAYERS,
            output_dim=NeuralNetConfig.OUTPUT_DIM,
            activation_function=NeuralNetConfig.ACTIVATION_FUNCTION
        )
        self.score: int = 0
        self.competition_wins: int = 0
        self.history: List[Interaction] = []

    def make_decision(self, current_state_inputs: List[float]) -> str:
        """
        Predicts whether to cooperate or defect based on the current competition state.
        Args:
            current_state_inputs (List[float]): Flattened array of all past interactions.
        Returns:
            str: "cooperate" or "defect"
        """
        output = self.nn.predict(current_state_inputs)
        if NeuralNetConfig.OUTPUT_DIM == 2:
            choice = torch.argmax(output).item()
            return Choice.COOPERATE.value if choice == 1 else Choice.DEFECT.value
        elif NeuralNetConfig.OUTPUT_DIM == 1:
            return Choice.COOPERATE.value if output.item() > 0.5 else Choice.DEFECT.value
        else:
            raise ValueError("Unsupported output dimension for decision making.")

    @staticmethod
    def batch_make_decisions(agents: List['Agent'], batch_inputs: torch.Tensor) -> List[str]:
        """
        Batch decision making for a list of agents on GPU.
        Args:
            agents (List[Agent]): List of agents.
            batch_inputs (torch.Tensor): Tensor of shape (batch_size, input_dim).
        Returns:
            List[str]: List of decisions ("cooperate" or "defect").
        """
        outputs = []
        for agent, inputs in zip(agents, batch_inputs):
            out = agent.nn.predict(inputs)
            if NeuralNetConfig.OUTPUT_DIM == 2:
                choice = torch.argmax(out).item()
                outputs.append(Choice.COOPERATE.value if choice == 1 else Choice.DEFECT.value)
            elif NeuralNetConfig.OUTPUT_DIM == 1:
                outputs.append(Choice.COOPERATE.value if out.item() > 0.5 else Choice.DEFECT.value)
            else:
                raise ValueError("Unsupported output dimension for decision making.")
        return outputs

    def reset_for_new_epoch(self) -> None:
        """Reset score and competition wins for a new epoch."""
        self.score = 0
        self.competition_wins = 0

    def reset_for_new_competition(self) -> None:
        """Reset history for a new competition."""
        self.history = []

    def record_interaction(self, my_choice: str, enemy_choice: str) -> None:
        """
        Record an interaction in the agent's history.
        Args:
            my_choice (str): The agent's choice ("cooperate" or "defect").
            enemy_choice (str): The opponent's choice ("cooperate" or "defect").
        """
        if my_choice not in (Choice.COOPERATE.value, Choice.DEFECT.value):
            raise ValueError(f"Invalid my_choice: {my_choice}")
        if enemy_choice not in (Choice.COOPERATE.value, Choice.DEFECT.value):
            raise ValueError(f"Invalid enemy_choice: {enemy_choice}")
        self.history.append(Interaction(Choice(my_choice), Choice(enemy_choice)))

    def get_competition_inputs(self, max_iterations: int) -> List[float]:
        """
        Converts the competition history into the neural network input format.
        Pads with masked inputs for remaining iterations.
        Args:
            max_iterations (int): Maximum number of iterations in the competition.
        Returns:
            List[float]: Input vector for the neural network.
        """
        inputs: List[float] = []
        for i in range(max_iterations):
            if i < len(self.history):
                my_choice = self.history[i].my_choice.value
                enemy_choice = self.history[i].enemy_choice.value
                if my_choice == Choice.COOPERATE.value and enemy_choice == Choice.COOPERATE.value:
                    inputs.extend(SimulationConfig.INPUT_CC)
                elif my_choice == Choice.COOPERATE.value and enemy_choice == Choice.DEFECT.value:
                    inputs.extend(SimulationConfig.INPUT_CD)
                elif my_choice == Choice.DEFECT.value and enemy_choice == Choice.COOPERATE.value:
                    inputs.extend(SimulationConfig.INPUT_DC)
                elif my_choice == Choice.DEFECT.value and enemy_choice == Choice.DEFECT.value:
                    inputs.extend(SimulationConfig.INPUT_DD)
            else:
                inputs.extend(SimulationConfig.INPUT_MASKED)
        return inputs