"""This script computes the Pareto equilibrium point given two players
and a number of actions."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import logging

import json
import utils as ut

from environment_analysis import environment_functions

# Set up logger function
logger = ut.get_logger(__name__)
logger.setLevel(logging.INFO)

# Load the environment parameters
# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))

env_params = params["environment_parameters"]

# Set seed for numpy
np.random.seed(0)


def compute_pareto(no_actions):
    """Compute the Pareto equilibrium points given two players and a number of actions."""
    action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)

    pareto_reward = 0
    rewards = []

    # Find th lowest nash point
    for S_i in action_set:
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=np.array([S_i]), S=[S_i], no_samples=40000
            )
        if true_rewards[0] > pareto_reward:
            pareto_reward = true_rewards[0]
            pareto_point = S_i
        rewards.append(true_rewards[0])

    return pareto_point, pareto_reward, rewards


if __name__ == "__main__":

    nums_of_actions = [5, 9, 17, 33, 65, 129, 801]

    for no_actions in nums_of_actions:
        pareto_point, pareto_reward, rewards = compute_pareto(no_actions)
        print(
            f'No actions {no_actions}, pareto point {pareto_point} with reward {pareto_reward}.')
        plt.plot(np.linspace(0.1, 0.9, no_actions, endpoint=True), rewards)
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.grid()
        plt.show()
