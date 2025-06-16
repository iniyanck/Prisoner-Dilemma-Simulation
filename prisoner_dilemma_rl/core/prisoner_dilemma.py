"""
PrisonerDilemma class for handling payoff logic.
Improvements:
- Added type hints
- Expanded docstrings
- Made play_round a static method
- Input validation for choices
- Removed unnecessary constructor
"""
from config.simulation_config import SimulationConfig
from core.agent import Choice
from typing import Tuple

class PrisonerDilemma:
    @staticmethod
    def play_round(agent1_choice: str, agent2_choice: str) -> Tuple[int, int]:
        """
        Determines the payoffs for a single round of Prisoner's Dilemma.
        Args:
            agent1_choice (str): Choice of agent 1 ("cooperate" or "defect").
            agent2_choice (str): Choice of agent 2 ("cooperate" or "defect").
        Returns:
            Tuple[int, int]: Payoff for agent 1 and agent 2.
        """
        if agent1_choice not in (Choice.COOPERATE.value, Choice.DEFECT.value):
            raise ValueError(f"Invalid agent1_choice: {agent1_choice}")
        if agent2_choice not in (Choice.COOPERATE.value, Choice.DEFECT.value):
            raise ValueError(f"Invalid agent2_choice: {agent2_choice}")
        payoff1, payoff2 = SimulationConfig.get_payoff(agent1_choice, agent2_choice)
        return payoff1, payoff2