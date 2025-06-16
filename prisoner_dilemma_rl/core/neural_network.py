"""
PyTorch-based NeuralNetwork class for GPU acceleration.
Replaces NumPy implementation with torch and supports CUDA, mixed precision, and JIT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import sys

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, activation_function: str = 'relu'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            if activation_function == 'relu':
                layers.append(nn.ReLU())
            elif activation_function == 'tanh':
                layers.append(nn.Tanh())
            elif activation_function == 'sigmoid':
                layers.append(nn.Sigmoid())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.to(self.device)
        # JIT compile if available (PyTorch 2.0+) and not on Windows
        try:
            if sys.platform != 'win32':
                self.model = torch.compile(self.model)
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, inputs: Union[List[float], torch.Tensor], use_amp: bool = False) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            if use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.forward(inputs)
            else:
                output = self.forward(inputs)
        return output.cpu().squeeze(0)

    def mutate(self, mutation_rate: float) -> None:
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * 0.1
                param[mask] += noise[mask]

    @staticmethod
    def crossover(nn1: 'NeuralNetwork', nn2: 'NeuralNetwork') -> 'NeuralNetwork':
        child = NeuralNetwork(nn1.model[0].in_features, [l.out_features for l in nn1.model if isinstance(l, nn.Linear)][:-1], nn1.model[-1].out_features)
        with torch.no_grad():
            for p_child, p1, p2 in zip(child.parameters(), nn1.parameters(), nn2.parameters()):
                mask = torch.rand_like(p_child) < 0.5
                p_child.copy_(torch.where(mask, p1, p2))
        return child