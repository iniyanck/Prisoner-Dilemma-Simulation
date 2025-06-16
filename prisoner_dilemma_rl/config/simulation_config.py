"""
Configuration for simulation parameters and Prisoner's Dilemma payoff matrix.
Improvements:
- Added docstrings for class and variables
- Added type hints for static methods
- Added docstrings for input encodings
"""

from typing import Tuple, List

class SimulationConfig:
    """
    Configuration for simulation parameters and Prisoner's Dilemma payoff matrix.
    """
    NUM_AGENTS: int = 100  #: Number of agents in the population
    NUM_EPOCHS: int = 100  #: Number of generations/epochs
    ITERATIONS_PER_COMPETITION: int = 10  #: Number of rounds in each Prisoner's Dilemma game

    # Genetic Algorithm Parameters
    SELECTION_RATE: float = 0.2  #: Top 20% of agents are selected for reproduction
    MUTATION_RATE: float = 0.1  #: Probability of a weight/bias being mutated

    # Scoring for competitions
    POINTS_FOR_WIN: int = 1  #: Points awarded for winning a competition

    # Prisoner's Dilemma Payoff Matrix (Row Player, Column Player)
    # Payoff (My_Score, Enemy_Score)
    # C = Cooperate, D = Defect
    # (My Choice, Enemy Choice) : (My Payoff, Enemy Payoff)
    PAYOFF_CC: Tuple[int, int] = (1, 1)  #: Both Cooperate: Reward for mutual cooperation
    PAYOFF_CD: Tuple[int, int] = (0, 2)  #: I Cooperate, Enemy Defects
    PAYOFF_DC: Tuple[int, int] = (2, 0)  #: I Defect, Enemy Cooperates
    PAYOFF_DD: Tuple[int, int] = (0, 0)  #: Both Defect: Punishment for mutual defection

    # Simulation mode: 'evolutionary' or 'rl'
    MODE: str = 'rl'  #: Set to 'rl' for pure RL mode, 'evolutionary' for genetic algorithm

    @staticmethod
    def get_payoff(my_choice: str, enemy_choice: str) -> Tuple[int, int]:
        """
        Returns the payoff for a single round of Prisoner's Dilemma.
        Args:
            my_choice (str): "cooperate" or "defect"
            enemy_choice (str): "cooperate" or "defect"
        Returns:
            Tuple[int, int]: (my_payoff, enemy_payoff)
        """
        if my_choice == "cooperate" and enemy_choice == "cooperate":
            return SimulationConfig.PAYOFF_CC
        elif my_choice == "cooperate" and enemy_choice == "defect":
            return SimulationConfig.PAYOFF_CD
        elif my_choice == "defect" and enemy_choice == "cooperate":
            return SimulationConfig.PAYOFF_DC
        elif my_choice == "defect" and enemy_choice == "defect":
            return SimulationConfig.PAYOFF_DD
        else:
            raise ValueError("Invalid choices for Prisoner's Dilemma round.")

    # Neural Network Input Encoding for historical interactions
    # Each interaction (my_choice, enemy_choice) is a 4-element vector.
    INPUT_CC: List[int] = [1, 0, 1, 0]  #: My Co, Enemy Co
    INPUT_CD: List[int] = [1, 0, 0, 1]  #: My Co, Enemy Def
    INPUT_DC: List[int] = [0, 1, 1, 0]  #: My Def, Enemy Co
    INPUT_DD: List[int] = [0, 1, 0, 1]  #: My Def, Enemy Def

    # Masked input for future/unknown rounds (e.g., all zeros or specific mask value)
    INPUT_MASKED: List[int] = [0, 0, 0, 0]  #: Represents an unplayed round