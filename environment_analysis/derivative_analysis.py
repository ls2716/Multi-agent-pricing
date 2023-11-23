import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from environment_analysis.environment_functions import \
    expected_reward_probability, compute_S_i_star, compute_derivative

import logging

import json
import utils as ut
import time

# Set up logger function
logger = ut.get_logger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":

    # Set seed for numpy
    np.random.seed(0)
    no_samples = 1000
    random_sample = np.random.normal(size=no_samples).reshape(1, -1)

    # Load the environment parameters
    # Read common parameters from yaml file
    params = ut.read_parameters('common_parameters.yaml')
    logger.info(json.dumps(params, indent=4))

    env_params = params["environment_parameters"]

    # S = [0.5]

    # tic = time.perf_counter()
    # for i in range(5):
    #     s_i_star = compute_S_i_star(S, env_params, random_sample)
    # toc = time.perf_counter()
    # logger.info(f's_i_star = {s_i_star}')
    # logger.info(f'Elapsed time {toc-tic:.4f}s')

    dim = 5

    S_js = np.linspace(-0.1, 0.1, dim)

    dSdis = np.zeros(shape=(dim, dim))
    dSdjs = np.zeros(shape=(dim, dim))
    sums = np.zeros(shape=(dim, dim))

    for i in range(dim):
        for j in range(dim):
            S = [S_js[i], S_js[j]]
            dSdi = compute_derivative(
                S, index=0, env_params=env_params, random_sample=random_sample)
            dSdj = compute_derivative(
                S, index=1, env_params=env_params, random_sample=random_sample)
            point = ', '.join([f'{item:.3f}' for item in S])
            derivatives = ', '.join([f'{item:.3f}' for item in [dSdi, dSdj]])
            logger.info(
                f'Derivatives at S=[{point}] are [{derivatives}]')
            dSdis[i, j] = dSdi
            dSdjs[i, j] = dSdj
            sums[i, j] = dSdi + dSdj

    print(' ')
    print('  ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{sums[i,j]:.2f}' for i in range(dim)]) + '\\\\')

    print(' ')
    print('  ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{dSdis[i,j]:.2f}' for i in range(dim)]) + '\\\\')

    print(' ')
    print('  ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{dSdjs[i,j]:.2f}' for i in range(dim)]) + '\\\\')
