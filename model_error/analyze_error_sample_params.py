"""This script analyzes discrepancy between the model and the environment"""
import os

from model_error import optimize_model
from environment_analysis import environment_functions

import numpy as np
import matplotlib.pyplot as plt

import utils as ut

logger = ut.get_logger(__name__)
logger.setLevel(ut.logging.INFO)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 14})

    # Create folder for plots
    images_folder = 'images/model_error'
    ut.create_folder(images_folder)

    # Set seed for numpy
    np.random.seed(0)

    no_actions = 5  # Number of actions
    action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

    # Read common parameters from yaml file
    params = ut.read_parameters('common_parameters.yaml')
    env_params = params['environment_parameters']

    # Define logistic model
    dimension = 2
    logistic_model = optimize_model.LogisticModel(
        action_set=action_set, dimension=dimension)

    # Define opponent actions
    opponent_action_set = np.linspace(
        0.1, 1.1, no_actions+1, endpoint=True)  # Action set

    plot_range = np.linspace(0, 1, 100)
    plotting_model = optimize_model.LogisticModel(
        action_set=plot_range, dimension=dimension)

    no_parameter_samples = 1000
    no_samples = 3000

    no_opponents = 3
    env_params['N'] = no_opponents+1

    opponent_margins = np.random.choice(
        opponent_action_set, size=(no_parameter_samples, no_opponents))

    sigmas = np.random.uniform(0.05, 0.55, size=no_parameter_samples)
    rhos = np.random.uniform(0.0, 1., size=no_parameter_samples)

    errors = np.zeros(shape=(no_actions, no_parameter_samples))
    sup_errors = np.zeros(shape=no_parameter_samples)

    env_params_sample = env_params.copy()

    for no_sample in range(no_parameter_samples):
        env_params_sample['sigma'] = sigmas[no_sample]
        env_params_sample['rho'] = rhos[no_sample]

        opponent_margins_sample = opponent_margins[no_sample, :].tolist()
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params_sample,
                S_i=action_set, S=opponent_margins_sample,
                no_samples=no_samples
            )
        error_function = optimize_model.get_error_function(
            logistic_model, true_probs)

        res = optimize_model.optimize_model(error_function=error_function)
        computed_probs = logistic_model.call(res.x)
        errors[:, no_sample] = np.abs(computed_probs - true_probs)

        # Compute supremum error
        true_probs_wide, _ = \
            environment_functions.expected_reward_probability(
                env_params=env_params_sample,
                S_i=plot_range, S=opponent_margins_sample,
                no_samples=no_samples
            )
        sup_errors[no_sample] = np.max(
            np.abs(true_probs_wide - plotting_model.call(res.x)))

    print(f'Mean error: {np.mean(errors, axis=1)}')

    mean_errors = np.mean(errors, axis=0)
    plt.figure(figsize=(6, 5))
    plt.hist(errors.flatten(), bins=50)
    plt.xlabel('Error level')
    plt.ylabel('Number of samples with error level')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(images_folder,
                'sample_params_errors.png'), dpi=300)
    plt.show()

    logger.info(f'Max error {np.max(errors, axis=1)}')
    worst_index = np.argmax(np.max(errors, axis=0))
    logger.info(f'Errors at worst index {errors[:, worst_index]}')

    opponent_margins_sample = opponent_margins[worst_index, :].tolist()
    env_params_sample['sigma'] = sigmas[worst_index]
    env_params_sample['rho'] = rhos[worst_index]
    logger.info(f'Worst opponent margins {opponent_margins_sample}')
    logger.info(f'Environment parametes {env_params_sample}')

    with open(os.path.join(images_folder, 'worst_case.txt'), 'w+') as f:
        f.write(f'Errors at worst index {errors[:, worst_index]}')
        f.write(f'Worst opponent margins {opponent_margins_sample}')
        f.write(f'Environment parametes {env_params_sample}')

    true_probs, true_rewards = \
        environment_functions.expected_reward_probability(
            env_params=env_params_sample,
            S_i=action_set, S=opponent_margins_sample, no_samples=no_samples
        )
    error_function = optimize_model.get_error_function(
        logistic_model, true_probs)

    res = optimize_model.optimize_model(error_function=error_function)
    logger.info(f'Optimization result: {res}')
    logger.info(f'Parameters {res.x}')

    plt.figure(figsize=(6, 5))
    plt.plot(plot_range, plotting_model.call(res.x),
             label='Modeled probability')
    plt.plot(plot_range, environment_functions.expected_reward_probability(
        env_params=env_params_sample, S_i=plot_range, S=opponent_margins_sample,
        no_samples=no_samples)[0],
        label='True probability')
    # plt.scatter(action_set, logistic_model.call(
    #     res.x), label='logistic collocations')
    plt.scatter(action_set, true_probs, label='Optimisation points')
    plt.xlabel('$S^i$')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(images_folder,
                'sample_params_worst_case.png'), dpi=300)
    plt.show()
