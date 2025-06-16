# Prisoner's Dilemma Reinforcement Learning & Evolutionary Simulation

A modern, extensible simulation of the iterated Prisoner's Dilemma using neural network-based agents evolved via a genetic algorithm. The project is designed for research, experimentation, and educational purposes, with a focus on code clarity, modularity, and explainability.

## Key Features
- **Evolutionary Algorithm:** Agents evolve through selection, crossover, and mutation.
- **Neural Network Agents:** Each agent uses a configurable feed-forward neural network for decision making.
- **Tournament Play:** Agents compete in round-robin tournaments, optimizing for long-term reward.
- **Explainability:** Tools for converting agent behavior to decision trees and generating LLM-based explanations.
- **Highly Configurable:** All simulation and neural network parameters are easily adjustable.
- **Extensible & Well-Documented:** Modular codebase with type hints, docstrings, and logging.

## Project Structure

prisoner_dilemma_rl/
├── config/
│ ├── simulation_config.py      # Configuration for simulation parameters
│ └── neural_net_config.py      # Configuration for neural network architecture
├── core/
│ ├── agent.py                  # Defines the Agent class (holds neural network, manages state)
│ ├── neural_network.py         # Implements a basic feed-forward neural network
│ ├── prisoner_dilemma.py       # Implements the rules and payoff matrix of the Prisoner's Dilemma
│ └── data_manager.py           # Placeholder for data saving/loading (currently in-memory)
├── evolution/
│ ├── genetic_algorithm.py      # Handles selection, crossover, and mutation
│ └── tournament.py             # Orchestrates agent competitions
├── analysis/
│ ├── decision_tree_converter.py# Conceptual module for converting NN to Decision Tree
│ └── llm_explainer.py          # Conceptual module for LLM-based explanation of agent behavior
├── main.py                     # Main script to run the simulation
└── README.md                   # Project README


## Requirements
- Python 3.8+
- numpy
- scikit-learn (for decision tree analysis)
- google-generativeai (for LLM explanations, optional)

Install all requirements:
```bash
pip install numpy scikit-learn google-generativeai
```

## How to Run
1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd "Prisoner's Dilemma Simulation"/prisoner_dilemma_rl
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually as above
    ```
3. **Run the simulation:**
    ```bash
    python main.py --seed 42
    ```
    - The `--seed` argument is optional and ensures reproducibility.

## Configuration

You can modify the simulation parameters and neural network architecture by editing the files in the `config/` directory:

* `config/simulation_config.py`:
    * `NUM_AGENTS`: Number of agents in the population.
    * `NUM_EPOCHS`: Number of generations the simulation will run.
    * `ITERATIONS_PER_COMPETITION`: Number of rounds in each competition between two agents.
    * `SELECTION_RATE`: Proportion of top agents selected for reproduction.
    * `MUTATION_RATE`: Probability of a weight/bias mutating during reproduction.
    * `PAYOFF_CC`, `PAYOFF_CD`, `PAYOFF_DC`, `PAYOFF_DD`: Prisoner's Dilemma payoff matrix.
    * `INPUT_MASKED`, `INPUT_CC`, `INPUT_CD`, `INPUT_DC`, `INPUT_DD`: Encoding for neural network inputs.

* `config/neural_net_config.py`:
    * `HIDDEN_LAYERS`: A list defining the number of neurons in each hidden layer (e.g., `[16, 8]`).
    * `ACTIVATION_FUNCTION`: Activation function for hidden layers ('relu', 'sigmoid', 'tanh').
    * `LEARNING_RATE`: Learning rate (primarily for potential future gradient-based optimization within agents).
    * `WEIGHT_INITIALIZATION_SCALE`: Scale for initial random weights.

## Simulation Logic

The simulation proceeds as an evolutionary algorithm:

1.  **Initialization:** A population of agents, each with a randomly initialized neural network, is created.
2.  **Epochs:** The simulation runs for a specified number of epochs (generations).
3.  **Tournament:** In each epoch, every agent competes against every other agent in a round-robin tournament.
    * **Competition:** Each competition consists of `ITERATIONS_PER_COMPETITION` rounds of the Prisoner's Dilemma.
    * **Decision Making:** Agents use their neural networks to decide whether to "cooperate" or "defect" in each round, based on the history of interactions in the current competition. The input to the neural network includes a flattened representation of past actions and "masked" inputs for future, unplayed rounds.
    * **Winning a Competition:** The agent that wins more individual iterations (gets a higher payoff in more rounds) within a competition is awarded a point.
4.  **Scoring:** At the end of all competitions, agents are scored based on the total points accumulated.
5.  **Selection:** The top `SELECTION_RATE` percentage of agents (based on their scores) are selected to form the parent pool for the next generation.
6.  **Reproduction (Crossover):** New agents are created by combining the neural network weights and biases of two randomly selected parents.
7.  **Mutation:** The neural networks of the newly created agents undergo random mutations to introduce diversity.
8.  **Next Generation:** The new population (selected parents + new offspring) becomes the agents for the next epoch.

## Analysis & Explainability
After the simulation, the best agent's neural network is analyzed:
1. **Decision Tree Conversion:**
    - Uses scikit-learn to approximate the agent's neural network with a decision tree for interpretability.
2. **LLM Explanation:**
    - Uses Google Gemini (or another LLM) to generate a human-readable summary of the agent's learned strategy.
    - Set your API key as an environment variable: `GOOGLE_API_KEY=your_key_here`.

## Extensibility & Code Quality
- The codebase uses type hints, docstrings, and logging for maintainability.
- Easily extend with new agent types, genetic operators, or analysis tools.
- All configuration is centralized in the `config/` directory.

## Future Improvements

* **Actual Decision Tree Conversion:** Integrate a library like scikit-learn to convert the NN to a decision tree and visualize it. This would require generating a comprehensive dataset from the NN's predictions.
* **Real LLM Integration:** Use an actual LLM API (e.g., OpenAI, Google Gemini) to generate more sophisticated explanations.
* **More Sophisticated Genetic Operators:** Implement more advanced crossover and mutation strategies.
* **Gradient-based Learning within Agents:** While this is an evolutionary simulation, agents could also internally use gradient descent for learning within a competition (though this adds complexity).
* **Visualization:** Visualize agent scores over epochs, and perhaps the competition dynamics.
* **Persistence:** Save and load simulation states and best agents.
* **Parallel Processing:** Speed up tournaments by running competitions in parallel.