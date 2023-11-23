"""This script defined utility functions for the bandit algorithms."""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Set up logging
import logging
from copy import deepcopy

# Increase font size for matplotlib
plt.rcParams.update({'font.size': 10})


# Set up logger function
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Setup format for logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Setup logging to stream
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info(f'Logger  set up with level {level}.')
    return logger


logger = get_logger(__name__)


def set_logging_level(name, level):
    t_logger = logging.getLogger(name)
    t_logger.setLevel(level)
    t_logger.info('Logging level set to {}'.format(level))


# Define a single game function between two bandits
def single_game(bandit1, bandit2, env, T):
    # logger.info('Running single game')
    no_actions = bandit1.n_actions
    reward_history = np.zeros(shape=(T, 2))
    action_bandit1_history = np.zeros(shape=(T, no_actions))
    action_bandit2_history = np.zeros(shape=(T, no_actions))
    for i in range(T):
        action_bandit1, action_bandit1_index = bandit1.get_action()
        action_bandit2, action_bandit2_index = bandit2.get_action()
        rewards, observations = env.step(
            np.array([action_bandit1, action_bandit2]))
        rewards = rewards.flatten()
        observations = observations.flatten()
        bandit1.update(action_bandit1_index, rewards[0], observations[0])
        bandit2.update(action_bandit2_index, rewards[1], observations[1])
        env.update()
        reward_history[i, :] = rewards[:]
        action_bandit1_history[i, action_bandit1_index] = 1.
        action_bandit2_history[i, action_bandit2_index] = 1.

    return reward_history, action_bandit1_history, action_bandit2_history


# Define function to run simulations
def run_simulations(bandit1, bandit2, env, T, no_sim, bandit1_0=None, bandit2_0=None):
    logger.info('Running %d simulations', no_sim)
    no_actions = bandit1.n_actions
    reward_history = np.zeros(shape=(T, 2))
    bandit1_action_frequencies = np.zeros(shape=(T, no_actions))
    bandit2_action_frequencies = np.zeros(shape=(T, no_actions))
    for i in range(no_sim):
        # Reset bandits
        if bandit1_0 is None:
            bandit1.reset()
        else:
            bandit1 = deepcopy(bandit1_0)
        if bandit2_0 is None:
            bandit2.reset()
        else:
            bandit2 = deepcopy(bandit2_0)

        # Reset the environment
        env.reset()

        logger.info('Simulation %d', i+1)
        reward_history_step, bandit1_action_history, bandit2_action_history = single_game(
            bandit1, bandit2, env, T)
        # Update reward history
        reward_history += reward_history_step/no_sim
        # Update action frequencies
        bandit1_action_frequencies += bandit1_action_history/no_sim
        bandit2_action_frequencies += bandit2_action_history/no_sim

    return reward_history, bandit1_action_frequencies, bandit2_action_frequencies


# Read parameter data from yaml file
def read_parameters(file_name):
    import yaml
    with open(file_name) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters


# Create a images folder using a foldername parameter
def create_folder(foldername):
    import os
    if not os.path.exists(foldername):
        os.makedirs(foldername)


# Define a function to plot the results
def plot_reward_history(reward_history, bandit1_name, bandit2_name,
                        foldername, filename, show_plot=True, title=None):
    # Plot the reward history
    plt.figure()
    plt.plot(reward_history[:, 0], label=bandit1_name)
    plt.plot(reward_history[:, 1], label=bandit2_name)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(foldername, filename + '_reward_history.png'))
    if show_plot:
        plt.show()
    plt.close()


def plot_smooth_reward_history(reward_history, bandit1_name, bandit2_name,
                               foldername, filename, show_plot=True, title=None, window=25):
    # Plot smoothed reward history with overlayed raw data
    plt.figure(figsize=(5, 5))
    smooth_reward_history = np.zeros(
        shape=(reward_history.shape[0]-window+1, 2))
    half_window = int(window/2)
    for i in range(reward_history.shape[1]):
        smooth_reward_history[:, i] = np.convolve(
            reward_history[:, i], np.ones(window)/window, mode='valid')
    timesteps = np.arange(
        half_window, reward_history.shape[0]-half_window + (window+1) % 2)
    plt.plot(
        timesteps, smooth_reward_history[:, 0], label=bandit1_name, color='C0')
    plt.plot(reward_history[:, 0], color='C0', alpha=0.25)
    plt.plot(
        timesteps, smooth_reward_history[:, 1], label=bandit2_name, color='C1')
    plt.plot(reward_history[:, 1], color='C1', alpha=0.25)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(foldername, filename + '_reward_history.png'))
    if show_plot:
        plt.show()
    plt.close()


# Define a function to plot the action frequencies for a single bandit
def plot_action_history(action_set, action_history, foldername, filename, title=None, show_plot=True):
    plt.figure(figsize=(5, 5))
    no_actions = action_history.shape[1]
    for i in range(no_actions):
        plt.plot(action_history[:, i], label=f'Action {action_set[i]:.2f}')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Frequency')
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(foldername, filename + '_action_frequencies.png'))
    if show_plot:
        plt.show()
    plt.close()


# Define a function to plot the action frequencies for a two bandits
def plot_action_history_two_bandits(action_set, timesteps, start_plot_index,
                                    action_history1, action_history2,
                                    foldername, filename, title1, title2, show_plot=True):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    no_actions = action_history1.shape[1]
    for i in range(no_actions):
        axs[0].plot(timesteps[start_plot_index:],
                    action_history1[start_plot_index:, i], label=f'Action {action_set[i]:.2f}')
    axs[0].legend()
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(title1)
    axs[0].grid()
    for i in range(no_actions):
        axs[1].plot(timesteps[start_plot_index:],
                    action_history2[start_plot_index:, i], label=f'Action {action_set[i]:.2f}')
    axs[1].legend()
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(title2)
    axs[1].grid()
    plt.savefig(os.path.join(foldername, filename + '_action_frequencies.png'))
    if show_plot:
        plt.show()
    plt.close()


def get_reward_profile(env, no_samples, action_set, fixed_action):
    mean_rewards = np.zeros_like(action_set)
    frequencies = np.zeros_like(action_set)
    action_indices = []
    observations = []
    rewards = []
    for j in range(no_samples):
        for i in range(action_set.shape[0]):
            rewards_step, observations_step = env.step(
                np.array([action_set[i], fixed_action]))
            frequencies[i] += observations_step[0, 0]/no_samples
            mean_rewards[i] += rewards_step[0, 0]/no_samples
            observations.append(observations_step[0, 0])
            rewards.append(rewards_step[0, 0])
            action_indices.append(i)
    return action_indices, rewards, observations, mean_rewards, frequencies


def print_result_to_file(cumulative_reward_1, cumulative_reward_2, filename, agent1_name, agent2_name):
    with open(filename, 'a+') as file:
        file.write('----------------------------------------\n')
        file.write(f'{agent1_name} reward = {cumulative_reward_1:.4f}\n')
        file.write(f'{agent2_name} reward = {cumulative_reward_2:.4f}\n')


def save_bandit(bandit, filepath):
    with open(filepath, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(bandit, f, pickle.HIGHEST_PROTOCOL)


def load_bandit(filepath):
    with open(filepath, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        bandit = pickle.load(f)
    return bandit
