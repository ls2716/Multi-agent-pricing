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

    # Set font size
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

    S_i = np.linspace(-1., 3, 1000)

    S_j = 0.8

    probs, reward = expected_reward_probability(
        env_params, S_i, [S_j], x_samples=random_sample)

    y_lims = [-0.2, 1.2]
    x_lims = [0.2, 1.4]
    S_C = 1.

    plt.figure(figsize=(6, 5))
    plt.plot(S_i, probs)
    plt.plot([S_j, S_j], y_lims, label='$S^{2}$')
    plt.plot([S_C, S_C], y_lims, label='$S^C$')
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.legend()
    # plt.title('Offer acceptance probability')
    plt.ylabel('probability')
    plt.xlabel('$S^{1}$')
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(images_folder, 'probability.png'))
    # plt.savefig(os.path.join(images_folder, 'probability.eps'))
    plt.show()

    y_lims = [-0.3, 0.9]

    plt.figure(figsize=(6, 5))
    plt.plot(S_i, reward)
    plt.plot([S_j, S_j], y_lims, label='$S^{2}$')
    plt.plot([S_C, S_C], y_lims, label='$S^C$')
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.legend()
    # plt.title('Expected reward')
    plt.ylabel('expected reward')
    plt.xlabel('$S^{1}$')
    # plt.gca().set_aspect('equal')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(images_folder, 'expected_reward.png'))
    # plt.savefig(os.path.join(images_folder, 'expected_reward.eps'))
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(S_i, probs)
    axs[0].plot([S_j, S_j], y_lims, label='$S^{2}$')
    axs[0].plot([S_C, S_C], y_lims, label='$S^C$')
    axs[0].set_xlim(x_lims)
    axs[0].set_ylim(y_lims)
    axs[0].legend()
    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('$S^{1}$')
    axs[0].grid()

    axs[1].plot(S_i, reward)
    axs[1].plot([S_j, S_j], y_lims, label='$S^{2}$')
    axs[1].plot([S_C, S_C], y_lims, label='$S^C$')
    axs[1].set_xlim(x_lims)
    axs[1].set_ylim(y_lims)
    axs[1].legend()
    axs[1].set_ylabel('expected reward')
    axs[1].set_xlabel('$S^{1}$')
    axs[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(images_folder,
                'expected_reward_and_probability.png'), dpi=300)
    # plt.savefig(os.path.join(images_folder,
    #             'expected_reward_and_probability.eps'))
    plt.show()
