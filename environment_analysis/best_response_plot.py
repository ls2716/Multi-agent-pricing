from environment_analysis.environment_functions import expected_reward_probability, compute_S_i_star, compute_derivative
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import logging

import os
import json
import utils as ut
import time


# Set up logger function
logger = ut.get_logger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    # Create folder for plots
    images_folder = 'images/environment_analysis'
    ut.create_folder(images_folder)

    plt.rcParams.update({'font.size': 14})
    # Set seed for numpy
    np.random.seed(0)
    no_samples = 2000
    random_sample = np.random.normal(size=no_samples).reshape(1, -1)

    # Load the environment parameters
    # Read common parameters from yaml file
    params = ut.read_parameters('common_parameters.yaml')
    logger.info(json.dumps(params, indent=4))

    env_params = params["environment_parameters"]

    n_points = 40

    S_js = np.linspace(-0.2, 1.2, n_points)
    b_i = np.zeros(n_points)

    for i in range(n_points):
        b_i[i] = compute_S_i_star([S_js[i]], env_params, random_sample)
        logger.info(
            f'Best response at S=[{S_js[i]}] is [{b_i[i]}]')

    plt.figure(figsize=(7, 5))
    plt.plot(S_js, b_i)
    # plt.plot([0, 1], [0, 1])
    plt.xlim([-0.2, 1.2])
    plt.ylim([0, 1])
    # plt.title('Best response plot')
    plt.ylabel('$B_1(S^2)$')
    plt.xlabel('$S^2$')
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.savefig(os.path.join(images_folder, 'best_response.png'), dpi=300)
    plt.show()
