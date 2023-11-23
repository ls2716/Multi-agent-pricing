"""This file contains games of non-stationary agents between each other where one
agent is fully converged"""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

from copy import deepcopy

from bandits import EpsGreedy
from models import NonStationaryClassicModel

# Initialize the environment
from envs import InsuranceMarketCt

from market_entering import me_utils

logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))
nash_payoff = params['environment_parameters']['nash_payoff']
pareto_payoff = params['environment_parameters']['pareto_payoff']


def dimless_payoff(x):
    return (x - nash_payoff) / (pareto_payoff - nash_payoff)


# Set seed
np.random.seed(0)


# Define parameters
T = 2000  # Number of time steps
no_actions = 129  # Number of actions


# Plotting parameters
results_folder = f'results/market_entering/classic_enters_classic_no_actions_{no_actions}_T_{T}'
images_folder = f'images/market_entering/classic_enters_classic_no_actions_{no_actions}_T_{T}'

# Create the images folder
ut.create_folder(images_folder)

# Load the data from the results folder
action_set = np.loadtxt(os.path.join(results_folder, 'action_set.csv'),
                        delimiter=',')
reward_history = np.loadtxt(os.path.join(results_folder, 'reward_history.csv'),
                            delimiter=',')
bandit_1_action_frequencies = np.loadtxt(os.path.join(results_folder, 'bandit_1_action_frequencies.csv'),
                                         delimiter=',')
bandit_2_action_frequencies = np.loadtxt(os.path.join(results_folder, 'bandit_2_action_frequencies.csv'),
                                         delimiter=',')


# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    results_folder, 'full_episode.txt'), 'Classic Eps-Greedy', 'Classic Eps-Greedy entering')

logger.info(
    f'Sum of last 500 rewards for the bandits:'
    + f' {np.sum(reward_history[-500:, 0])}, {np.sum(reward_history[-500:, 1])}')
reward_1_sum_500 = np.sum(reward_history[-500:, 0])
reward_2_sum_500 = np.sum(reward_history[-500:, 1])
ut.print_result_to_file(reward_1_sum_500, reward_2_sum_500, os.path.join(
    results_folder, 'last_500.txt'), 'Classic Eps-Greedy', 'Classic Eps-Greedy entering')

logger.info(
    f'Mean of last 500 rewards for the bandits:'
    + f' {np.mean(reward_history[-500:, 0])}, {np.mean(reward_history[-500:, 1])}')
reward_1_mean_500 = np.mean(reward_history[-500:, 0])
reward_2_mean_500 = np.mean(reward_history[-500:, 1])
ut.print_result_to_file(reward_1_mean_500, reward_2_mean_500, os.path.join(
    results_folder, 'mean_last_500.txt'), 'Classic Eps-Greedy', 'Classic Eps-Greedy entering')


ut.plot_smooth_reward_history(
    reward_history, bandit1_name='Classic Eps-Greedy',
    bandit2_name='Classic Eps-Greedy entering', foldername=images_folder, filename='reward_plot.png', title=None, show_plot=False)
