"""This script computes the Nash equilibrium point given two players
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


def compute_nash(no_actions):
    """Compute the Nash equilibrium points given two players and a number of actions."""
    action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)

    nash_points = []

    # Find th lowest nash point
    S_0 = action_set[0]
    while True:
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=action_set, S=[S_0], no_samples=20000
            )
        s_0_index = np.argmax(true_rewards)
        new_S_0 = action_set[s_0_index]

        if S_0 == new_S_0:
            nash_points.append((s_0_index, new_S_0, true_rewards[s_0_index]))
            break
        S_0 = new_S_0

    for it in range(s_0_index+1, no_actions):
        action = action_set[it]
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=action_set, S=[action], no_samples=100000
            )
        index = np.argmax(true_rewards)
        new_S_0 = action_set[index]
        # print(f'Best against {S_0} is {new_S_0}')
        if action == new_S_0:
            nash_points.append((index, action, true_rewards[index]))
        elif len(nash_points) > 0:
            break

    return nash_points


if __name__ == "__main__":

    nums_of_actions = [5, 9, 17, 33, 65, 129, 801]

    for no_actions in nums_of_actions:
        nash_points = compute_nash(no_actions)
        print(
            f'No actions {no_actions}, nash_points {nash_points}')
