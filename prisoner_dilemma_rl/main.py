"""
Main entry point for the Prisoner's Dilemma RL Simulation.
Improvements:
- Added logging instead of print statements
- Added type hints
- Added random seed control
- Modularized run_simulation into smaller functions
- Added error handling
"""
import random
import numpy as np
import logging
import argparse
from typing import List, Optional

from config.simulation_config import SimulationConfig
from config.neural_net_config import NeuralNetConfig
from core.agent import Agent
from evolution.tournament import Tournament
from evolution.genetic_algorithm import GeneticAlgorithm
from analysis.decision_tree_converter import DecisionTreeConverter
from analysis.llm_explainer import LLMExplainer

def setup_logging():
    import os
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'simulation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def set_random_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logging.info(f"Random seed set to {seed}")

def create_agents(genetic_algo: GeneticAlgorithm, num_agents: int) -> List[Agent]:
    return genetic_algo.create_initial_population(num_agents)

def run_epoch(tournament: Tournament, genetic_algo: GeneticAlgorithm, agents: List[Agent], epoch: int) -> List[Agent]:
    logging.info(f"--- Epoch {epoch + 1}/{SimulationConfig.NUM_EPOCHS} ---")
    agents = tournament.run_epoch()
    selected_agents = genetic_algo.select_top_agents(agents, SimulationConfig.SELECTION_RATE)
    logging.info(f"Selected {len(selected_agents)} top agents for reproduction.")
    if not selected_agents:
        logging.warning("No agents selected, cannot reproduce. Exiting.")
        return []
    num_new_agents_needed = SimulationConfig.NUM_AGENTS - len(selected_agents)
    if num_new_agents_needed > 0:
        new_agents = genetic_algo.reproduce(selected_agents, num_new_agents_needed)
        logging.info(f"Created {len(new_agents)} new agents through reproduction.")
        agents = selected_agents + new_agents
    else:
        agents = selected_agents
        logging.info("Population maintained by selected agents (no new agents created).")
    genetic_algo.mutate_population(agents, SimulationConfig.MUTATION_RATE)
    logging.info("Mutated current generation.")
    if len(agents) != SimulationConfig.NUM_AGENTS:
        logging.warning(f"Population size mismatch. Expected {SimulationConfig.NUM_AGENTS}, got {len(agents)}. Adjusting...")
        agents = agents[:SimulationConfig.NUM_AGENTS]
    if agents:
        logging.info(f"Top agent's score in epoch {epoch + 1}: {agents[0].score} (ID: {agents[0].id[:8]}...)")
    return agents

def collect_behavior_samples(best_agent: Agent, input_dim: int) -> tuple:
    sample_inputs = []
    sample_choices = []
    num_observation_competitions = 50
    for _ in range(num_observation_competitions):
        dummy_opponent = Agent(input_dim)
        best_agent.reset_for_new_competition()
        dummy_opponent.reset_for_new_competition()
        for i in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
            current_agent_inputs = best_agent.get_competition_inputs(SimulationConfig.ITERATIONS_PER_COMPETITION)
            current_opponent_inputs = dummy_opponent.get_competition_inputs(SimulationConfig.ITERATIONS_PER_COMPETITION)
            agent_choice = best_agent.make_decision(current_agent_inputs)
            opponent_choice = dummy_opponent.make_decision(current_opponent_inputs)
            best_agent.record_interaction(agent_choice, opponent_choice)
            dummy_opponent.record_interaction(opponent_choice, agent_choice)
            sample_inputs.append(np.array(current_agent_inputs).flatten())
            sample_choices.append(agent_choice)
    return np.array(sample_inputs), sample_choices

def run_simulation(seed: Optional[int] = None):
    """Run the main simulation loop."""
    setup_logging()
    set_random_seed(seed)
    logging.info("Starting Prisoner's Dilemma Reinforcement Learning Simulation...")
    input_dim = SimulationConfig.ITERATIONS_PER_COMPETITION * 4
    NeuralNetConfig.INPUT_DIM = input_dim
    genetic_algo = GeneticAlgorithm(input_dim)
    agents = create_agents(genetic_algo, SimulationConfig.NUM_AGENTS)
    tournament = Tournament(agents)
    logging.info(f"Initial population size: {len(agents)}")
    logging.info(f"Running for {SimulationConfig.NUM_EPOCHS} epochs...")
    try:
        for epoch in range(SimulationConfig.NUM_EPOCHS):
            agents = run_epoch(tournament, genetic_algo, agents, epoch)
            if not agents:
                break
        logging.info("--- Simulation Complete ---")
        if agents:
            best_agent = agents[0]
            logging.info(f"Best performing agent (ID: {best_agent.id}) has a final score of {best_agent.score}.")
            logging.info("Generating extensive behavior samples from the best agent for analysis...")
            sample_inputs, sample_choices = collect_behavior_samples(best_agent, input_dim)
            feature_names = []
            for i in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
                feature_names.extend([
                    f"Iter_{i+1}_My_CC", f"Iter_{i+1}_My_CD", f"Iter_{i+1}_My_DC", f"Iter_{i+1}_My_DD"
                ])
            dt_converter = DecisionTreeConverter()
            trained_decision_tree = dt_converter.convert_nn_to_decision_tree(best_agent.nn, sample_inputs, sample_choices)
            tree_summary = dt_converter.summarize_tree(feature_names=feature_names)
            logging.info("--- Decision Tree Rules Extracted from Best Agent's Behavior ---\n" + tree_summary)
            llm_explainer = LLMExplainer()
            nn_structure_info = f"""
            Input Dimension: {NeuralNetConfig.INPUT_DIM}
            Hidden Layers: {NeuralNetConfig.HIDDEN_LAYERS}
            Output Dimension: {NeuralNetConfig.OUTPUT_DIM}
            Activation Function: {NeuralNetConfig.ACTIVATION_FUNCTION}
            """
            simulation_context_info = {
                'iterations_per_competition': SimulationConfig.ITERATIONS_PER_COMPETITION
            }
            llm_explanation = llm_explainer.explain_agent_behavior(tree_summary, nn_structure_info, simulation_context_info)
            logging.info("--- LLM Explanation of Best Agent's Behavior ---\n" + llm_explanation)
        else:
            logging.warning("No agents found at the end of the simulation.")
    except Exception as e:
        logging.error(f"Simulation failed: {e}", exc_info=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma RL Simulation")
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_simulation(seed=args.seed)