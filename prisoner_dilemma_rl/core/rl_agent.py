"""
RLDQNAgent: Neural network-based RL agent for the Prisoner's Dilemma.
Uses DQN (Deep Q-Learning) with experience replay.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple
from core.neural_network import NeuralNetwork
from config.neural_net_config import NeuralNetConfig
from config.simulation_config import SimulationConfig

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class RLDQNAgent:
    def __init__(self, input_dim: int, action_dim: int = 2, lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = NeuralNetwork(input_dim, NeuralNetConfig.HIDDEN_LAYERS, action_dim, NeuralNetConfig.ACTIVATION_FUNCTION).to(self.device)
        self.target_net = NeuralNetwork(input_dim, NeuralNetConfig.HIDDEN_LAYERS, action_dim, NeuralNetConfig.ACTIVATION_FUNCTION).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.batch_size = 64
        self.update_target_steps = 100
        self.learn_step = 0

    def select_action(self, state: np.ndarray, use_amp: bool = False) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    q_values = self.q_net(state_tensor)
            else:
                q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values).item())

    @staticmethod
    def batch_select_actions(agents: List['RLDQNAgent'], states: np.ndarray, use_amp: bool = False) -> List[int]:
        # Assumes all agents share the same architecture
        device = agents[0].device
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        actions = []
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    q_values = torch.stack([agent.q_net(states_tensor[i].unsqueeze(0)) for i, agent in enumerate(agents)])
            else:
                q_values = torch.stack([agent.q_net(states_tensor[i].unsqueeze(0)) for i, agent in enumerate(agents)])
            actions = [int(torch.argmax(q).item()) for q in q_values]
        return actions

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self, use_amp: bool = False):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        if use_amp and self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            with torch.cuda.amp.autocast():
                q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = self.target_net(next_state).max(1)[0]
                    target = reward + self.gamma * next_q * (1 - done)
                loss = nn.MSELoss()(q_values, target)
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_state).max(1)[0]
                target = reward + self.gamma * next_q * (1 - done)
            loss = nn.MSELoss()(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_policy(self):
        return self.q_net
