"""Implementation of bandit algorithms."""

# Import libraries
import numpy as np
import pickle

from models import BaseModel

from utils import get_logger

logger = get_logger(__name__)


class BaseBandit(object):
    """Base bandit class for bandit algorithms

    Should contain:
        - get_action
        - get_action_probabilities
        - update
    """

    def __init__(self, model: BaseModel = None) -> None:
        """Initialize the algorithm.

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
            - model - model for the environment
        """
        self.actions = model.actions
        self.n_actions = model.n_actions
        self.model = model

    def reset(self):
        """Reset agent"""
        self.model.reset()

    def get_action(self):
        """Get action.
        """
        pass

    def get_action_probabilities(self):
        """Get action probabilities.
        """
        pass

    def update(self, action_index, action_vector, reward, observation):
        """Update state of rewards.
        """
        pass

    def check_status(self):
        """Check status of agent.
        """
        pass

    def model_info(self):
        """Return model info.
        """
        return self.model.info()


class FixedActionAgent(BaseBandit):

    def __init__(self, action) -> None:
        self.action = action
        logger.debug(
            "Initialized FixedActionAgent with action {}".format(action))

    def get_action(self):
        return self.action, 0

    def update(self, *args):
        pass

    def reset(self, **kwargs):
        pass


class FluctuatingActionAgent(BaseBandit):

    def __init__(self, mean_action, half_amplitude, period) -> None:
        self.mean_action = mean_action
        self.half_amplitude = half_amplitude
        self.period = period
        self.t = 0
        logger.debug(
            f"Initialized FluctuatingActionAgent with mean action {mean_action}," +
            f" half amplitude {half_amplitude}, period {period}."
        )

    def get_action(self):
        # Compute sinusoid given current time step t and period
        sin = np.sin(self.t/self.period*2*np.pi)
        return self.mean_action + self.half_amplitude*sin, 0

    def update(self, *args):
        self.t += 1

    def reset(self, **kwargs):
        self.t = 0


class TwoActionAgent(BaseBandit):

    def __init__(self, actions) -> None:
        self.actions = actions

    def get_action(self):
        # Sample random action
        return np.random.sample(self.actions, 1)[0], 0

    def update(self, *args):
        self.t += 1

    def reset(self, **kwargs):
        self.t = 0


class SequenceAgent(BaseBandit):

    def __init__(self, action_sequence) -> None:
        self.action_sequence = action_sequence

    def get_action(self):
        # Sample random action
        return self.action_sequence[self.t], 0

    def update(self, *args):
        self.t = (self.t + 1) % len(self.action_sequence)

    def reset(self, **kwargs):
        self.t = 0


class EpsGreedy(BaseBandit):
    """eps-greed bandit algorithm as in OTC paper
    """

    def __init__(self, eps: float, model: BaseModel) -> None:
        """Initialize the algorithm.

        Arguments:
            - eps (float): epsilon
            - candidate_margins (numpy array of floats): list of possible margins
            - model - model for the environment
        """
        # Initialize super
        super().__init__(model)
        self.eps = eps
        self.actions = self.model.actions

    def reset(self):
        """Reset agent"""
        super().reset()

    def get_action(self):
        """Get action.
        """
        # Pick action
        self.r = self.model.get_expected_rewards()
        if np.random.random() > self.eps:
            # Best if greedy
            max_r = np.max(self.r)
            max_actions = [i
                           for i in range(self.n_actions) if self.r[i] > max_r-0.001]
            action_index = np.random.choice(max_actions)
        else:
            # Random uniformly if exploratory
            action_index = np.random.randint(self.n_actions)
        return self.actions[action_index], action_index

    def get_action_probabilities(self):
        """Get action probabilities.
        """
        greedy_action_index = np.argmax(self.r)
        probabilities = np.zeros_like(self.r)
        probabilities[greedy_action_index] = 1-self.eps
        probabilities += self.eps/self.n_actions
        return probabilities

    def update(self, action_index, reward, observation):
        """Update state of rewards.
        """
        self.model.update(action_index,
                          reward, observation)


class UCB_Bayes(BaseBandit):
    """Implementation of UCB-Bayes bandit"""

    def __init__(self, model: BaseModel, T, c=0, quantile=0.7) -> None:
        super().__init__(model)
        self.model = model
        self.quantile = quantile
        self.c = c
        self.T = T
        self.reset()
        if self.T is not None:
            self.inv_logTc = 1/np.power(np.log(self.T), c)

    def reset(self):
        """Reset agent"""
        super().reset()
        self.t = 0

    def get_action(self):
        if self.T is None:
            quantile = self.quantile
        else:
            quantile = 1 - np.minimum(0.5, self.inv_logTc/(self.t+1))
        action_values = self.model.get_quantiles(quantile)
        max_v = np.max(action_values)
        max_actions = [i
                       for i in range(self.n_actions) if action_values[i] > max_v-0.001]
        action_index = np.random.choice(max_actions)
        return self.actions[action_index], action_index

    def update(self, action_index, reward, observation):
        """Update state of the model"""
        self.t += 1
        self.model.update(action_index,
                          reward, observation)


class ThompsonSampling(BaseBandit):
    """Implementation of Thompson Sampling bandit"""

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model)
        self.model = model
        self.reset()

    def reset(self):
        """Reset agent"""
        super().reset()
        self.t = 0

    def get_action(self):
        action_values = self.model.sample_reward_TS()
        max_v = np.max(action_values)
        max_actions = [i
                       for i in range(self.n_actions) if action_values[i] > max_v-0.001]
        action_index = np.random.choice(max_actions)
        return self.actions[action_index], action_index

    def update(self, action_index, reward, observation):
        """Update state of the model"""
        self.t += 1
        self.model.update(action_index,
                          reward, observation)
