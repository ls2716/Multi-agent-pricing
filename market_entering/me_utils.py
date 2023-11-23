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

from bandits import FixedActionAgent

# Initialize the environment
from envs import InsuranceMarketCt


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


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment


# Define a simulation between two agents by names
def simulate_game(bandit_1, bandit_2, results_folder, bandit_1_name, bandit_2_name, simulation_setup):

    action_set = simulation_setup['action_set']
    no_sim = simulation_setup['no_sim']
    T = simulation_setup['T']

    ut.create_folder(results_folder)

    # Run initial simulation to teach the agent
    fixed_action = 1000.  # Action for fixed-action agent
    fixed_action_bandit = FixedActionAgent(action=fixed_action)

    # Run single learning simulation for first agent
    _, _, _ = ut.run_simulations(
        bandit_1, fixed_action_bandit, env, T, 1)
    bandit_1.model.info()
    bandit1_0 = deepcopy(bandit_1)
    logger.info('Bandit 1 has learned')

    # Run simulations
    reward_history, bandit_1_action_frequencies, bandit_2_action_frequencies = ut.run_simulations(
        bandit_1, bandit_2, env, T, no_sim, bandit1_0=bandit1_0)

    reward_history = dimless_payoff(reward_history)

    # Print cumulative reward for the bandit and save to a file
    logger.info(
        f'Sum of rewards for the bandits: {np.sum(reward_history[:, 0])}, {np.sum(reward_history[:, 1])}')
    reward_1_sum = np.sum(reward_history[:, 0])
    reward_2_sum = np.sum(reward_history[:, 1])
    ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
        results_folder, 'full_episode.txt'), bandit_1_name, bandit_2_name + ' entering')

    logger.info(
        f'Sum of last 500 rewards for the bandits:'
        + f' {np.sum(reward_history[-500:, 0])}, {np.sum(reward_history[-500:, 1])}')
    reward_1_sum_500 = np.sum(reward_history[-500:, 0])
    reward_2_sum_500 = np.sum(reward_history[-500:, 1])
    ut.print_result_to_file(reward_1_sum_500, reward_2_sum_500, os.path.join(
        results_folder, 'last_500.txt'), bandit_1_name, bandit_2_name + ' entering')

    # Save action set to a csv file
    np.savetxt(os.path.join(results_folder, 'action_set.csv'),
               action_set, delimiter=',')

    # Save reward history to a csv file
    np.savetxt(os.path.join(results_folder, 'reward_history.csv'),
               reward_history, delimiter=',')

    # Save action frequencies to a csv file
    np.savetxt(os.path.join(results_folder, 'bandit_1_action_frequencies.csv'),
               bandit_1_action_frequencies, delimiter=',')
    np.savetxt(os.path.join(results_folder, 'bandit_2_action_frequencies.csv'),
               bandit_1_action_frequencies, delimiter=',')
