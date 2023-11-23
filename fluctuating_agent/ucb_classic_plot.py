"""This file tests the epsilon greedy bandit algorithm against
a dummy agent which always quotes the same fixed action."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os


logger = ut.get_logger(__name__)

no_actions = 5  # Number of actions

# Folder parameters
image_folder = f'images/fluctuating_agent/ucb_classic_{no_actions}'
result_folder = f'results/fluctuating_agent/ucb_classic_{no_actions}'


# Read data for plotting from results folder
action_set = np.loadtxt(os.path.join(result_folder, 'action_set.csv'),
                        delimiter=',')
reward_history = np.loadtxt(os.path.join(result_folder, 'reward_history.csv'),
                            delimiter=',')
bandit_action_frequencies = np.loadtxt(os.path.join(result_folder, 'bandit_action_frequencies.csv'),
                                       delimiter=',')


# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])

logger.info(
    f'Sum of last 500 rewards for the bandits:'
    + f' {np.sum(reward_history[-500:, 0])}, {np.sum(reward_history[-500:, 1])}')
reward_1_sum_500 = np.sum(reward_history[-500:, 0])
reward_2_sum_500 = np.sum(reward_history[-500:, 1])
ut.print_result_to_file(reward_1_sum_500, reward_2_sum_500, os.path.join(
    result_folder, 'last_500.txt'), 'Classic UCB-Bayes', 'Fluctuating-action Agent')


# Plot results using the plot functions from utils.py
ut.create_folder(image_folder)
ut.plot_action_history(action_set, bandit_action_frequencies,
                       foldername=image_folder, filename='ucb_classic_actions')
ut.plot_smooth_reward_history(
    reward_history, bandit1_name='Classic UCB-Bayes',
    bandit2_name='Fluctuating-action Agent', foldername=image_folder, filename='ucb_classic_rewards')
