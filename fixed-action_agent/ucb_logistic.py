"""This file tests the epsilon greedy bandit algorithm against
a dummy agent which always quotes the same fixed action."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

# Import bandit algorithms
from bandits import UCB_Bayes

# Import fixed action agent
from bandits import FixedActionAgent

from models import StationaryLogisticModel

# Initialize the environment
from envs import InsuranceMarketCt


logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define dimless payoff function
nash_payoff = params['environment_parameters']['nash_payoff']
pareto_payoff = params['environment_parameters']['pareto_payoff']


def dimless_payoff(x):
    return (x - nash_payoff) / (pareto_payoff - nash_payoff)


# Set seed
np.random.seed(0)

# Define parameters
no_sim = 200  # Number of simulations
T = 1000  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

quantile = 0.7  # quantile for UCB-Bayes
dimension = 2  # dimension of the model

fixed_action = 0.7  # Action for fixed-action agent

# Folder parameters
result_folder = f'results/fixed-action_agent/ucb_logistic_{no_actions}'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
model = StationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension)
# Initialize epsilon-greedy bandit
bandit = UCB_Bayes(model=model, T=None, c=0, quantile=quantile)
# Initialize dummy agent
fixed_agent = FixedActionAgent(action=fixed_action)


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_action_frequencies, fixed_action_frequencies = ut.run_simulations(
    bandit, fixed_agent, env, T, no_sim)


reward_history = dimless_payoff(reward_history)

# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])
ut.create_folder(result_folder)
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    result_folder, 'total_rewards.txt'), 'Logistic UCB-Bayes', 'Fixed-Action Agent')

# Save action set to a csv file
np.savetxt(os.path.join(result_folder, 'action_set.csv'),
           action_set, delimiter=',')

# Save reward history to a csv file
np.savetxt(os.path.join(result_folder, 'reward_history.csv'),
           reward_history, delimiter=',')

# Save action frequencies to a csv file
np.savetxt(os.path.join(result_folder, 'bandit_action_frequencies.csv'),
           bandit_action_frequencies, delimiter=',')
