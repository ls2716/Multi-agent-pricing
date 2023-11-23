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
image_folder = f'images/fixed-action_agent/ucb_classic_{no_actions}'
result_folder = f'results/fixed-action_agent/ucb_classic_{no_actions}'


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


# Plot results using the plot functions from utils.py
ut.create_folder(image_folder)
ut.plot_action_history(action_set, bandit_action_frequencies,
                       foldername=image_folder, filename='ucb_classic_actions')
ut.plot_smooth_reward_history(
    reward_history, bandit1_name='Classic UCB-Bayes',
    bandit2_name='Fixed-Action Agent', foldername=image_folder, filename='ucb_classic_rewards')
