# Bandits for Pricing

This repository contains the code for the numerical simulations
and analysis for the work presented in the manuscript [link]().

Below see a short description of the repository.

### Contents of the scripts in the parent directory

 - utils.py - contains the utility functions for running the simulations and plotting,
 - models.py - contains the implementations of the bandit models of the environment,
 - envs.py - contains the implementation of the insurance pricing environment,
 - bandits.py - contains the implementation of bandit classes that implement the bandit decision rules and join them with the bandit models,
 - common_parameters.yaml - contains the numerical parameters of the environment to be used throughout the simulations.

### Description of the subfolders

The subfolders contain the code for particular parts of the manuscript.

- environment_analysis - code for the analysis of the environment i.e. the probability and the expected reward functions,
- model_error - code for computing the error between the true probability function and the logistic model regression + the code for computing the Nash and Pareto equilibria,
- fixed-action_agent - code for simulations between bandits and the fixed-action agent,
- fluctuating_agent - code for simulations between bandits and the agent whose margin fluctuates and follows a square wave,
- market_entering - code for simulations of competitions between bandits where one bandit is already present in the environment and then another enters the environment and starts learning.

Additionally, folders "images" and "results"  contain the plots and the text results of the simulations.


The simulations follow the same basic procedure:
1. Import the necessary bandit model.
2. Import the necessary bandit class (decision rule).
3. Import the environment.
4. Initialise the environment with the common_parameters.yaml .
5. Initialise the bandit with appriopriate model.
6. Run the simulation.
7. Save the results.

Additional files, usually named '*_plot.py', contain the code used for plotting.

### Running the code

Each script has to be run from the main directory (one that contains this readme.md) by using python -m.

```bash
python -m <folder>/<script_name>
```

For example:
```bash
python -m fixed-action_agent/epsgreedy_classic.py
```

For any questions, please message the repository owner or contact the corresponding author of the referenced manuscript.
