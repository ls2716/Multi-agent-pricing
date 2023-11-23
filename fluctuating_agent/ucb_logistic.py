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

# Import sequence action agent
from bandits import SequenceAgent

from models import NonStationaryLogisticModel

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
no_actions = 129  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

tau = 40  # Window size for sliding window method

quantile = 0.7  # quantile for UCB-Bayes
dimension = 2  # dimension of the model


# Folder parameters
result_folder = f'results/fluctuating_agent/ucb_logistic_{no_actions}'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
model = NonStationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension, method='sliding_window', tau=tau)
# Initialize epsilon-greedy bandit
bandit = UCB_Bayes(model=model, T=None, c=0, quantile=quantile)
# Initialize dummy agent with fluctuating action
low_action = 0.3
high_action = 0.7
period = 500
fluctuating_agent = SequenceAgent(
    [low_action]*(period//2) + [high_action]*(period//2)
)


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_action_frequencies, fixed_action_frequencies = ut.run_simulations(
    bandit, fluctuating_agent, env, T, no_sim)


reward_history = dimless_payoff(reward_history)

# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])
ut.create_folder(result_folder)
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    result_folder, 'total_rewards.txt'), 'Logistic UCB-Bayes', 'Fluctuating-action Agent')

logger.info(
    f'Sum of last 500 rewards for the bandits:'
    + f' {np.sum(reward_history[-500:, 0])}, {np.sum(reward_history[-500:, 1])}')
reward_1_sum_500 = np.sum(reward_history[-500:, 0])
reward_2_sum_500 = np.sum(reward_history[-500:, 1])
ut.print_result_to_file(reward_1_sum_500, reward_2_sum_500, os.path.join(
    result_folder, 'last_500.txt'), 'Logistic UCB-Bayes', 'Fluctuating-action Agent')


# Save action set to a csv file
np.savetxt(os.path.join(result_folder, 'action_set.csv'),
           action_set, delimiter=',')

# Save reward history to a csv file
np.savetxt(os.path.join(result_folder, 'reward_history.csv'),
           reward_history, delimiter=',')

# Save action frequencies to a csv file
np.savetxt(os.path.join(result_folder, 'bandit_action_frequencies.csv'),
           bandit_action_frequencies, delimiter=',')
