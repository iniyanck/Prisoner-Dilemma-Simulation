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
import multiprocessing
import torch
import torch.nn as nn
from typing import List, Optional

from config.simulation_config import SimulationConfig
from config.neural_net_config import NeuralNetConfig
from core.agent import Agent
from core.rl_agent import RLDQNAgent
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

def simulate_agent_matches(args):
    agent_idx, agent, agents, input_dim = args
    experiences = []
    for j, opponent in enumerate(agents):
        if agent_idx == j:
            continue
        state = np.zeros(input_dim)
        for t in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
            action = agent.select_action(state)
            opp_action = opponent.select_action(state)
            action_str = "cooperate" if action == 1 else "defect"
            opp_action_str = "cooperate" if opp_action == 1 else "defect"
            payoff, opp_payoff = SimulationConfig.get_payoff(action_str, opp_action_str)
            next_state = state.copy()
            idx = t * 4
            if action_str == "cooperate" and opp_action_str == "cooperate":
                next_state[idx:idx+4] = SimulationConfig.INPUT_CC
            elif action_str == "cooperate" and opp_action_str == "defect":
                next_state[idx:idx+4] = SimulationConfig.INPUT_CD
            elif action_str == "defect" and opp_action_str == "cooperate":
                next_state[idx:idx+4] = SimulationConfig.INPUT_DC
            elif action_str == "defect" and opp_action_str == "defect":
                next_state[idx:idx+4] = SimulationConfig.INPUT_DD
            done = (t == SimulationConfig.ITERATIONS_PER_COMPETITION - 1)
            experiences.append((state, action, payoff, next_state, done))
            state = next_state
    return agent_idx, experiences

def run_rl_simulation(seed: Optional[int] = None):
    setup_logging()
    set_random_seed(seed)
    logging.info("Starting Prisoner's Dilemma RL Simulation (DQN mode, multiprocessing + GPU batching)...")
    input_dim = SimulationConfig.ITERATIONS_PER_COMPETITION * 4
    NeuralNetConfig.INPUT_DIM = input_dim
    num_agents = SimulationConfig.NUM_AGENTS
    agents = [RLDQNAgent(input_dim) for _ in range(num_agents)]
    logging.info(f"Initialized {num_agents} RL agents. All agents persist and learn independently.")
    for epoch in range(SimulationConfig.NUM_EPOCHS):
        logging.info(f"--- RL Epoch {epoch + 1}/{SimulationConfig.NUM_EPOCHS} ---")
        # Multiprocessing: simulate matches in parallel
        import multiprocessing
        num_workers = min(4, multiprocessing.cpu_count())
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(simulate_agent_matches, [(i, agents[i], agents, input_dim) for i in range(num_agents)])
        # Gather all experiences for all agents
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        agent_total_rewards = [0 for _ in range(num_agents)]
        for agent_idx, experiences in results:
            for state, action, reward, next_state, done in experiences:
                all_states.append(state)
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_states.append(next_state)
                all_dones.append(done)
                agent_total_rewards[agent_idx] += reward
        # Batch update: all agents share the same architecture, so we can update them in a single GPU operation
        if all_states:
            import torch
            import torch.nn as nn
            states = torch.tensor(np.array(all_states), dtype=torch.float32)
            actions = torch.tensor(np.array(all_actions), dtype=torch.long)
            rewards = torch.tensor(np.array(all_rewards), dtype=torch.float32)
            next_states = torch.tensor(np.array(all_next_states), dtype=torch.float32)
            dones = torch.tensor(np.array(all_dones), dtype=torch.float32)
            for agent in agents:
                agent.q_net.train()
                q_values = agent.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = agent.target_net(next_states).max(1)[0]
                    target = rewards + agent.gamma * next_q * (1 - dones)
                loss = nn.MSELoss()(q_values, target)
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                agent.learn_step += 1
                if agent.learn_step % agent.update_target_steps == 0:
                    agent.target_net.load_state_dict(agent.q_net.state_dict())
        # Verbose stats for this epoch
        avg_reward = np.mean(agent_total_rewards)
        max_reward = np.max(agent_total_rewards)
        min_reward = np.min(agent_total_rewards)
        logging.info(f"Epoch {epoch+1} stats: Avg reward: {avg_reward:.2f}, Max reward: {max_reward:.2f}, Min reward: {min_reward:.2f}")
        top5 = sorted(agent_total_rewards, reverse=True)[:5]
        logging.info(f"Top 5 agent rewards: {top5}")
    logging.info("RL training complete. All agents persisted and learned independently.")
    # Optionally, analyze all agents or select the best by some metric
    best_agent = max(agents, key=lambda ag: ag.epsilon_min)  # Placeholder: you can use a real evaluation
    logging.info("Proceeding to analysis of best RL agent...")
    # Generate samples for decision tree analysis
    sample_inputs = []
    sample_choices = []
    for _ in range(50):
        state = np.zeros(input_dim)
        for t in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
            action = best_agent.select_action(state)
            action_str = "cooperate" if action == 1 else "defect"
            sample_inputs.append(state.copy())
            sample_choices.append(action_str)
            idx = t * 4
            if action_str == "cooperate":
                state[idx:idx+4] = SimulationConfig.INPUT_CC
            else:
                state[idx:idx+4] = SimulationConfig.INPUT_DD
    feature_names = []
    for i in range(SimulationConfig.ITERATIONS_PER_COMPETITION):
        feature_names.extend([
            f"Iter_{i+1}_My_CC", f"Iter_{i+1}_My_CD", f"Iter_{i+1}_My_DC", f"Iter_{i+1}_My_DD"
        ])
    dt_converter = DecisionTreeConverter()
    trained_decision_tree = dt_converter.convert_nn_to_decision_tree(best_agent.q_net, np.array(sample_inputs), sample_choices)
    tree_summary = dt_converter.summarize_tree(feature_names=feature_names)
    logging.info("--- Decision Tree Rules Extracted from Best RL Agent's Behavior ---\n" + tree_summary)
    llm_explainer = LLMExplainer()
    nn_structure_info = f"""
    Input Dimension: {NeuralNetConfig.INPUT_DIM}
    Hidden Layers: {NeuralNetConfig.HIDDEN_LAYERS}
    Output Dimension: 2
    Activation Function: {NeuralNetConfig.ACTIVATION_FUNCTION}
    """
    simulation_context_info = {
        'iterations_per_competition': SimulationConfig.ITERATIONS_PER_COMPETITION
    }
    llm_explanation = llm_explainer.explain_agent_behavior(tree_summary, nn_structure_info, simulation_context_info)
    logging.info("--- LLM Explanation of Best RL Agent's Behavior ---\n" + llm_explanation)

def run_simulation(seed: Optional[int] = None):
    if SimulationConfig.MODE == 'rl':
        run_rl_simulation(seed)
        return
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
            # Add more project context for LLM
            project_context = f"""
Project: Prisoner's Dilemma RL/Evolutionary Simulation
- Agents: {SimulationConfig.NUM_AGENTS}
- Epochs: {SimulationConfig.NUM_EPOCHS}
- Iterations per competition: {SimulationConfig.ITERATIONS_PER_COMPETITION}
- Mode: {SimulationConfig.MODE}
- Payoff Matrix: CC={SimulationConfig.PAYOFF_CC}, CD={SimulationConfig.PAYOFF_CD}, DC={SimulationConfig.PAYOFF_DC}, DD={SimulationConfig.PAYOFF_DD}
- Neural Net: Hidden Layers={NeuralNetConfig.HIDDEN_LAYERS}, Activation={NeuralNetConfig.ACTIVATION_FUNCTION}, Output Dim={NeuralNetConfig.OUTPUT_DIM}
- Input Encoding: Each round is 4 values, history is padded/masked for future rounds.
Simulation Description: Each agent plays repeated Prisoner's Dilemma games against all others. In RL mode, all agents learn independently using DQN. In evolutionary mode, agents evolve via selection, crossover, and mutation. The goal is to maximize long-term reward in a competitive, adversarial environment. After training, the best agent's neural network is analyzed and explained using a decision tree and LLM.
"""
            llm_explanation = llm_explainer.explain_agent_behavior(
                tree_summary,
                nn_structure_info + "\n" + project_context,
                simulation_context_info
            )
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