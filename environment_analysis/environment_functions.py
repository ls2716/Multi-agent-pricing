import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import logging

import json
import utils as ut
import time

# Set up logger function
logger = ut.get_logger(__name__)
logger.setLevel(logging.INFO)


def expected_reward_probability(env_params, S_i, S, x_samples=None, no_samples=1000):
    """Calculate the expected reward and probability
    given the environment parameters and agents' actions.
    """
    # Get parameters from the env_params dictionary
    sigma = env_params['sigma']
    tau = env_params['tau']
    rho = env_params['rho']
    S_c = env_params['S_c']
    S_i = S_i.reshape(-1, 1)

    if x_samples is None:
        # Generate Monte Carlo samples for normal distribution
        x_samples = np.random.normal(size=no_samples).reshape(1, -1)

    prob = np.ones_like(S_i)
    prob = prob * norm.cdf(-(S_i - S_c + sigma*np.sqrt(1-rho)
                           * x_samples) / np.sqrt(tau**2 + rho*sigma**2))
    for S_j in S:
        prob = prob * norm.cdf(-(S_i - S_j + sigma*rho *
                               x_samples) / np.sqrt((1-rho)*sigma**2))
    # Calculate the probability
    prob = np.mean(prob, axis=1).flatten()
    reward = prob*S_i.flatten()
    return prob, reward


# Define a function to compute S_i_star
def compute_S_i_star(S, env_params, random_sample):
    """Compute the optimal S_i given S_j"""
    S_i = np.linspace(0, 2, 201)
    # print(S_i)
    prob, reward = expected_reward_probability(
        env_params, S_i, S, x_samples=random_sample)
    S_i_star = S_i[np.argmax(reward)]
    S_i = np.linspace(S_i_star-0.01, S_i_star+0.01, 201)
    prob, reward = expected_reward_probability(
        env_params, S_i, S, x_samples=random_sample)
    S_i_star = S_i[np.argmax(reward)]

    return S_i_star


# Compute derivative given S_j
def compute_derivative(S, index, env_params, random_sample, dx=0.005):
    S0 = S.copy()
    S1 = S.copy()
    S0[index] = S0[index]-dx
    S1[index] = S1[index]+dx
    S_i_star0 = compute_S_i_star(S0, env_params, random_sample)
    S_i_star1 = compute_S_i_star(S1, env_params, random_sample)
    derivative = (S_i_star1-S_i_star0)/(2*dx)
    return derivative
