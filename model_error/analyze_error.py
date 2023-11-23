"""This script analyzes discrepancy between the model and the environment"""

from model_error import optimize_model
from environment_analysis import environment_functions

import numpy as np
import matplotlib.pyplot as plt

import utils as ut

logger = ut.get_logger(__name__)
logger.setLevel(ut.logging.INFO)


if __name__ == '__main__':
    # Create folder for plots
    images_folder = 'images/model_error'
    ut.create_folder(images_folder)

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.titlesize'] = 16

    # Set numpy seed
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

    # Set the number of samples
    no_samples = 5000

    # Define opponent actions
    opponent_action_set = np.linspace(
        0.1, 1.1, no_actions+1, endpoint=True)  # Action set

    plot_range = np.linspace(0, 1, 100)
    plotting_model = optimize_model.LogisticModel(
        action_set=plot_range, dimension=dimension)

    fig, axs = plt.subplots(2, 3, figsize=(13, 9), sharex=True, sharey=True)
    # fig.suptitle('Model discrepancy for different opponent actions')

    errors = np.zeros((no_actions+1, no_actions))
    supremum_errors = np.zeros(no_actions+1)

    for i, opponent_action in enumerate(opponent_action_set):
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=action_set, S=[opponent_action], no_samples=no_samples
            )
        error_function = optimize_model.get_error_function(
            logistic_model, true_probs)

        res = optimize_model.optimize_model(error_function=error_function)
        logger.info(f'Optimization result: {res}')
        computed_probs = logistic_model.call(res.x)
        errors[i, :] = np.sqrt(((computed_probs - true_probs)**2))

        # Compute supremum error
        true_probs_wide, _ = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=plot_range, S=[opponent_action]
            )
        supremum_errors[i] = np.max(
            np.abs(true_probs_wide - plotting_model.call(res.x)))

        plot_i, plot_j = i//3, i % 3
        axs[plot_i, plot_j].plot(plot_range, plotting_model.call(res.x),
                                 label='Modeled probability')
        axs[plot_i, plot_j].plot(plot_range, environment_functions.expected_reward_probability(
            env_params=env_params, S_i=plot_range, S=[opponent_action])[0],
            label='True probability')
        axs[plot_i, plot_j].scatter(action_set, true_probs,
                                    label='Optimisation points')
        axs[plot_i, plot_j].grid()
        # axs[plot_i, plot_j].legend()
        if plot_i == 1:
            axs[plot_i, plot_j].set_xlabel('$S^1$')
        if plot_j == 0:
            axs[plot_i, plot_j].set_ylabel('Probability')
        axs[plot_i, plot_j].set_title(
            f'Opponent action $S^2$ = {opponent_action:.2f}')
    handles, labels = plt.gca().get_legend_handles_labels()
    # plt.subplots_adjust(right=0.7)
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('images/model_error/model_discrepancy_N_2.png', dpi=300)
    plt.show()

    print("Supremum errors:")
    print(supremum_errors)

    logger.info(f'Errors:\n {errors}')

    # Print errors in a latex format
    print('Errors:')
    print('Opponent action & ' +
          ' & '.join([f'{action:.1f}' for action in action_set]))
    print('\\\\')
    print('\\hline')
    for i, error in enumerate(errors):
        print(f'{opponent_action_set[i]:.1f} & ' +
              ' & '.join([f'{error_value:.1e}' for error_value in error]) + f' & {supremum_errors[i]:.1e}', end=" \\\\\n")
