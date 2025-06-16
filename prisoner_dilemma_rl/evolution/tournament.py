"""
Tournament class for running agent competitions.
Improvements:
- Added type hints
- Added/expanded docstrings
- Used logging for important events
- Avoided redundant matches (agent1 vs agent2 and agent2 vs agent1)
- Input validation for agents list
"""

import random
import logging
import torch
from typing import List
from core.agent import Agent
from core.prisoner_dilemma import PrisonerDilemma
from config.simulation_config import SimulationConfig
from config.neural_net_config import NeuralNetConfig

class Tournament:
    def __init__(self, agents: List[Agent]):
        """
        Initialize the Tournament with a list of agents.
        Args:
            agents (List[Agent]): List of agents to compete.
        """
        if not agents:
            raise ValueError("Agents list cannot be empty.")
        self.agents = agents
        self.prisoner_dilemma = PrisonerDilemma()

    def run_competition(self, agent1: Agent, agent2: Agent) -> None:
        """
        Runs a competition between two agents over a number of iterations.
        The agent with more individual iteration wins wins the competition.
        Args:
            agent1 (Agent): First agent.
            agent2 (Agent): Second agent.
        """
        agent1.reset_for_new_competition()
        agent2.reset_for_new_competition()

        agent1_iteration_wins = 0
        agent2_iteration_wins = 0

        for i in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
            agent1_inputs = agent1.get_competition_inputs(SimulationConfig.ITERATIONS_PER_COMPETITION)
            agent2_inputs = agent2.get_competition_inputs(SimulationConfig.ITERATIONS_PER_COMPETITION)

            agent1_choice = agent1.make_decision(agent1_inputs)
            agent2_choice = agent2.make_decision(agent2_inputs)

            payoff1, payoff2 = self.prisoner_dilemma.play_round(agent1_choice, agent2_choice)

            if payoff1 > payoff2:
                agent1_iteration_wins += 1
            elif payoff2 > payoff1:
                agent2_iteration_wins += 1

            agent1.record_interaction(agent1_choice, agent2_choice)
            agent2.record_interaction(agent2_choice, agent1_choice) # Record from agent2's perspective

        if agent1_iteration_wins > agent2_iteration_wins:
            agent1.competition_wins += SimulationConfig.POINTS_FOR_WIN
        elif agent2_iteration_wins > agent1_iteration_wins:
            agent2.competition_wins += SimulationConfig.POINTS_FOR_WIN
        # If tied, no one gets a point for this competition

    def run_epoch(self) -> List[Agent]:
        """
        Each agent competes with all other agents (unique pairs only), using GPU parallelization for batch decision making.
        Returns:
            List[Agent]: Agents sorted by score after the epoch.
        """
        for agent in self.agents:
            agent.reset_for_new_epoch()
        n = len(self.agents)
        matchups = []
        agent1_inputs = []
        agent2_inputs = []
        agent1_refs = []
        agent2_refs = []
        # Prepare all matchups and their input states
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = self.agents[i], self.agents[j]
                a1.reset_for_new_competition()
                a2.reset_for_new_competition()
                for _ in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
                    agent1_inputs.append(torch.tensor(a1.get_competition_inputs(SimulationConfig.ITERATIONS_PER_COMPETITION), dtype=torch.float32))
                    agent2_inputs.append(torch.tensor(a2.get_competition_inputs(SimulationConfig.ITERATIONS_PER_COMPETITION), dtype=torch.float32))
                    agent1_refs.append(a1)
                    agent2_refs.append(a2)
        # Stack inputs for batch processing
        agent1_inputs_tensor = torch.stack(agent1_inputs)
        agent2_inputs_tensor = torch.stack(agent2_inputs)
        # Batch decision making
        agent1_choices = Agent.batch_make_decisions(agent1_refs, agent1_inputs_tensor)
        agent2_choices = Agent.batch_make_decisions(agent2_refs, agent2_inputs_tensor)
        # Update histories and scores
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = self.agents[i], self.agents[j]
                a1_iteration_wins = 0
                a2_iteration_wins = 0
                for k in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
                    c1 = agent1_choices[idx]
                    c2 = agent2_choices[idx]
                    payoff1, payoff2 = self.prisoner_dilemma.play_round(c1, c2)
                    if payoff1 > payoff2:
                        a1_iteration_wins += 1
                    elif payoff2 > payoff1:
                        a2_iteration_wins += 1
                    a1.record_interaction(c1, c2)
                    a2.record_interaction(c2, c1)
                    idx += 1
                if a1_iteration_wins > a2_iteration_wins:
                    a1.competition_wins += SimulationConfig.POINTS_FOR_WIN
                elif a2_iteration_wins > a1_iteration_wins:
                    a2.competition_wins += SimulationConfig.POINTS_FOR_WIN
        for agent in self.agents:
            agent.score = agent.competition_wins
        self.agents.sort(key=lambda agent: agent.score, reverse=True)
        logging.info(f"Epoch complete. Top agent score: {self.agents[0].score}")
        return self.agents