"""
LLMExplainer for summarizing agent behavior using an LLM.
Improvements:
- Added type hints and docstrings
- Used logging for errors
- Optionally allow API key to be passed as argument
"""
import google.generativeai as genai
import os
import logging
from typing import Optional

class LLMExplainer:
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the LLM client.
        Args:
            api_key (Optional[str]): Google API key. If not provided, uses environment variable.
        """
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def explain_agent_behavior(self, decision_tree_summary: str, neural_net_structure_info: str, simulation_context: dict) -> str:
        """
        Uses an LLM to summarize the decision tree and explain the agent's general behavior.
        Args:
            decision_tree_summary (str): Rules from the decision tree.
            neural_net_structure_info (str): Description of the neural network.
            simulation_context (dict): Context about the simulation.
        Returns:
            str: LLM-generated explanation.
        """
        prompt = f"""
        Given the following information about a neural network-based agent trained in a Prisoner's Dilemma simulation:

        Neural Network Structure:
        {neural_net_structure_info}

        Decision Tree Rules (derived from the best agent's neural network's behavior):
        {decision_tree_summary}

        Simulation Context:
        - The agent played against all other agents in an adversarial environment.
        - Each competition involved {simulation_context['iterations_per_competition']} iterations.
        - The goal was to optimize reward, with points awarded for winning competitions (based on more iteration wins).
        - Input to the neural network included historical interactions (I cooperated/defected, Enemy cooperated/defected) and masked inputs for future rounds.
        - The decision tree rules describe the agent's choices (0 for defect, 1 for cooperate).

        Please analyze these decision tree rules and the simulation context to summarize the general behavior of this final, optimized agent.
        What strategy does it appear to have learned? Provide a concise and insightful explanation.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error communicating with LLM: {e}")
            return f"Error communicating with LLM: {e}"