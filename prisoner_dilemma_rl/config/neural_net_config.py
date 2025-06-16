"""
Configuration for neural network hyperparameters.
Improvements:
- Added docstrings for class and variables
- Added type annotations
"""

from typing import Optional, List


class NeuralNetConfig:
    """
    Configuration for neural network hyperparameters.
    """
    INPUT_DIM: Optional[int] = None  #: Input dimension (set by main.py)
    HIDDEN_LAYERS: List[int] = [64, 32]  #: Hidden layer sizes
    OUTPUT_DIM: int = 2  #: Output dimension (2 for cooperate/defect)
    ACTIVATION_FUNCTION: str = 'relu'  #: Activation function for hidden layers
    LEARNING_RATE: float = 0.01  #: Learning rate (not used in GA)
    WEIGHT_INITIALIZATION_SCALE: float = 0.1  #: Scale for weight initialization