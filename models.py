# Implementation of the bandit environment models


# Import libraries
import os
import numpy as np
from scipy.special import expit
import scipy
import matplotlib.pyplot as plt


from utils import get_logger

logger = get_logger(__name__)
logger.setLevel('INFO')


class BaseModel(object):
    """Base model object"""

    def __init__(self, candidate_margins) -> None:
        self.actions = candidate_margins.reshape(-1)
        self.n_actions = self.actions.shape[0]
        pass

    def reset(self) -> None:
        pass

    def info(self):
        pass

    def update(self, action_index, observation) -> None:
        pass

    def get_expected_rewards(self):
        pass

    def set_action_vectors(self):
        pass

    def get_quantiles(self, quantile):
        pass

    def sample_reward_TS(self):
        pass


class StationaryClassicModel(BaseModel):
    """Classical model implementation"""

    def __init__(self, variance, candidate_margins) -> None:
        super().__init__(candidate_margins)
        self.mean = np.zeros(self.n_actions)
        self.inv_cov = np.zeros(self.n_actions)
        self.inv_variance = 1/variance
        self.set_action_vectors(candidate_margins)

    def reset(self, **model_kwargs) -> None:
        if model_kwargs.get('mean') is not None:
            self.mean = model_kwargs.get('mean')
            self.inv_cov = model_kwargs.get(
                'inv_cov')
        else:
            self.mean = np.zeros(self.n_actions)
            self.inv_cov = np.zeros(self.n_actions)

    def update(self, action_index,  reward, observation) -> None:
        new_cov = 1/(self.inv_variance + self.inv_cov[action_index])
        self.mean[action_index] = new_cov*(
            self.mean[action_index] * self.inv_cov[action_index] + reward*self.inv_variance)
        self.inv_cov[action_index] = 1/new_cov

    def set_action_vectors(self, candidate_margins):
        self.action_vectors = np.array(candidate_margins).reshape(-1)

    def get_expected_rewards(self):
        return self.mean

    def get_quantiles(self, quantile):
        inv_cov = np.maximum(self.inv_cov, 0.0001)
        return scipy.stats.norm.ppf(quantile, loc=self.mean, scale=np.sqrt(1/inv_cov))

    def sample_reward_TS(self):
        # Generate parameter sample
        inv_cov = np.maximum(self.inv_cov, 0.0001)
        mean_sample = np.random.multivariate_normal(
            mean=self.mean.reshape(-1), cov=np.diag(1/inv_cov), size=1)
        # Return the sampled means - equal to mean reward for the sample
        return mean_sample.reshape(-1)

    def info(self):
        logger.info(f'Mean: \n {self.mean}')
        logger.info(f'Inverse covariance: \n {self.inv_cov}')


class NonStationaryClassicModel(StationaryClassicModel):
    """Classical model implementation"""

    def __init__(self, variance, candidate_margins,
                 method='discounting', gamma=0.9, tau=20) -> None:
        super().__init__(variance=variance, candidate_margins=candidate_margins)
        self.method = method
        if method == 'discounting':
            self.update = self.update_discounting
            self.gamma = gamma
        elif method == "sliding_window":
            self.update = self.update_sliding_window
            self.tau = tau
        else:
            raise ValueError(
                'Invalid method for nonstationarity. Choose "discounting" or "sliding_window".')
        self.reset()
        self.set_action_vectors(candidate_margins)

    def reset(self) -> None:
        self.mean = np.zeros(self.n_actions)
        self.inv_cov = np.zeros(self.n_actions)
        self.t = 0
        self.X = None
        self.Y = None

    def update_discounting(self, action_index, reward, observation) -> None:
        new_cov = 1/(self.inv_variance + self.gamma *
                     self.inv_cov[action_index])
        self.mean[action_index] = new_cov*(
            self.gamma * self.mean[action_index] * self.inv_cov[action_index] + reward*self.inv_variance)
        self.inv_cov[action_index] = 1/new_cov

    def update_sliding_window(self, action_index, reward, observation) -> None:
        self.t += 1
        if self.X is None:
            self.X = np.zeros((1, self.n_actions))
            self.X[0, action_index] = 1.
            self.Y = np.array([reward])
        else:
            self.X = np.vstack((self.X, np.zeros(self.n_actions)))
            self.X[-1, action_index] = 1.
            self.Y = np.append(self.Y, reward)
        start_index = max(0, self.t - self.tau)

        self.inv_cov = np.sum(
            self.X[start_index:self.t, :], axis=0)*self.inv_variance
        self.mean = np.sum(self.X[start_index:self.t, :] *
                           self.Y[start_index:self.t, None], axis=0) /\
            np.where(self.inv_cov == 0, 1, self.inv_cov)*self.inv_variance
        # self.mean = np.nan_to_num(self.mean)

    def get_expected_rewards(self):
        return super().get_expected_rewards()


def sigmoid(x):
    return expit(x)


class StationaryLogisticModel(BaseModel):
    """Defines the logistic regression model for bandits
    with iterative update for the mean based on of all the observations."""

    def __init__(self, candidate_margins, dimension=2, variance=10**3) -> None:
        """Initialisation function."""
        super().__init__(candidate_margins=candidate_margins)
        self.dimension = dimension
        self.variance = variance
        self.reset()
        self.set_action_vectors()

    def reset(self):
        """Reset model."""
        self.mean = np.zeros(shape=(self.dimension, 1))
        self.cov = np.eye(self.dimension)*self.variance
        self.cov_0 = np.eye(self.dimension)*self.variance
        self.xks = None
        self.Ys = None
        self.t = 0

    def sample_rewards(self, action_vectors, no_samples):
        """Sample rewards for given action vectors."""
        # Sample parameters from current posterior
        mean_samples = np.random.multivariate_normal(
            self.mean.reshape(-1), self.cov, size=no_samples)
        probabilities = sigmoid(mean_samples @ action_vectors)
        rewards = action_vectors[1, :] * probabilities
        return rewards

    def stabilise(self, m_threshold=100, cov_threshold=50):
        """A stabilistation function for the model."""
        # If the mean is too large, set it to 0
        if np.max(np.abs(self.mean)) > m_threshold:
            self.mean = np.zeros_like(self.mean)
        if np.max(np.abs(self.cov)) > cov_threshold:
            self.mean = np.zeros_like(self.mean)

    def update(self, action_index, reward, observation):
        xk = self.action_vectors[:, action_index].reshape(-1, 1)
        Y = observation
        if self.xks is None:
            self.xks = xk
            self.Ys = np.array([Y])
            self.weights = np.ones_like(self.Ys)
        else:
            self.xks = np.hstack((self.xks, xk))
            self.Ys = np.append(self.Ys, Y)
            self.weights = np.ones_like(self.Ys)
        self.t += 1
        new_mean, new_cov = self.compute_mean_and_cov()
        self.mean = new_mean
        self.cov = new_cov
        return new_mean, new_cov

    def compute_mean_and_cov(self, m_threshold=50, cov_threshold=50):
        if np.max(np.abs(self.cov)) > cov_threshold or np.max(np.abs(self.mean)) > m_threshold:
            self.mean = np.zeros_like(self.mean)

        mutxks = self.mean.T @ self.xks
        self.S_t = np.zeros_like(self.cov_0)
        self.S_t_factors = sigmoid(mutxks)*(
            1-sigmoid(mutxks))*self.weights
        self.S_t = self.S_t + \
            self.xks @ (self.S_t_factors * self.xks).T
        self.mu_sum = np.zeros_like(self.mean)
        self.mu_sum += np.sum(self.weights * (self.Ys - sigmoid(mutxks))
                              * self.xks, axis=1, keepdims=True)
        new_cov = scipy.linalg.inv(scipy.linalg.inv(
            self.cov_0) + self.S_t)
        new_mean = new_cov @ (self.S_t @ self.mean + self.mu_sum)
        return new_mean, new_cov

    def compute_mean_and_cov_scipy(self, m_threshold=50, cov_threshold=50):
        if np.max(np.abs(self.cov)) > cov_threshold or np.max(np.abs(self.mean)) > m_threshold:
            self.mean = np.zeros_like(self.mean)
            self.mean = self.compute_mean_scipy()
            mutxks = self.mean.T @ self.xks
            self.S_t = np.zeros_like(self.cov_0)
            self.S_t_factors = sigmoid(mutxks)*(
                1-sigmoid(mutxks))*self.weights
            self.S_t = self.S_t + \
                self.xks @ (self.S_t_factors * self.xks).T
            new_cov = scipy.linalg.inv(scipy.linalg.inv(
                self.cov_0) + self.S_t)
            new_mean = self.mean
        else:
            mutxks = self.mean.T @ self.xks
            self.S_t = np.zeros_like(self.cov_0)
            self.S_t_factors = sigmoid(mutxks)*(
                1-sigmoid(mutxks))*self.weights
            self.S_t = self.S_t + \
                self.xks @ (self.S_t_factors * self.xks).T
            self.mu_sum = np.zeros_like(self.mean)
            self.mu_sum += np.sum(self.weights * (self.Ys - sigmoid(mutxks))
                                  * self.xks, axis=1, keepdims=True)
            new_cov = scipy.linalg.inv(scipy.linalg.inv(
                self.cov_0) + self.S_t)
            new_mean = new_cov @ (self.S_t @ self.mean + self.mu_sum)
        return new_mean, new_cov

    def compute_mean_scipy(self):
        """Compute the mean using scipy minimize.
        """
        # logger.debug(f'Computing mean using scipy minimize at t={self.t}')

        def loss_function(mean):
            mutxks = mean.reshape(-1, 1).T @ self.xks
            probabilities = sigmoid(mutxks)
            # print(probabilities)
            return -np.sum((self.Ys*np.log(probabilities) + (1-self.Ys)*np.log(1-probabilities)))
        result = scipy.optimize.minimize(loss_function, self.mean.flatten(), bounds=[
                                         (-50, 50), (-50, 50)])
        return result.x.reshape(-1, 1)

    def info(self):
        logger.info(f'Mean: \n {self.mean}')
        logger.info(f'Covariance: \n {self.cov}')

    def get_expected_rewards(self):
        return self.get_quantiles(0.5)
        probabilities = sigmoid(self.mean.T @ self.action_vectors)
        rewards = self.action_vectors[1, :] * probabilities
        return rewards.reshape(-1)

    def set_action_vectors(self):
        self.action_vectors = np.ones((self.dimension, self.n_actions))
        for i in range(1, self.dimension):
            self.action_vectors[i, :] = self.actions * \
                self.action_vectors[i-1, :]
        logger.info(f'Action vectors: \n {self.action_vectors}')

    def get_quantiles(self, quantile, size=1000):
        # Generate parameter samples
        mean_samples = np.random.multivariate_normal(
            mean=self.mean.reshape(-1), cov=self.cov, size=size)
        # Compute expected reward samples
        probabilities = sigmoid(mean_samples @ self.action_vectors)
        reward_samples = self.action_vectors[1, :] * probabilities
        # Return quantile of the reward samples
        return np.quantile(reward_samples, quantile, axis=0)

    def sample_reward_TS(self):
        # Generate parameter samples
        mean_sample = np.random.multivariate_normal(
            mean=self.mean.reshape(-1), cov=self.cov, size=1)
        # Compute expected reward samples
        probabilities = sigmoid(mean_sample @ self.action_vectors)
        reward_sample = self.action_vectors[1, :] * probabilities
        # Return quantile of the reward samples
        return reward_sample.reshape(-1)


class NonStationaryLogisticModel(StationaryLogisticModel):
    """Implementation of non-stationary logistic model.

    Uses either discounting or sliding window approach."""

    # Initialisation function
    def __init__(self, candidate_margins, dimension=2, variance=10**3, method='discounting',
                 gamma=0.9, tau=100) -> None:
        super().__init__(candidate_margins=candidate_margins,
                         dimension=dimension, variance=variance)
        self.method = method
        self.method = method
        if method == 'discounting':
            self.update = self.update_discounting
            self.gamma = gamma
        elif method == "sliding_window":
            self.update = self.update_sliding_window
            self.tau = tau
        else:
            raise ValueError(
                'Invalid method for nonstationarity. Choose "discounting" or "sliding_window".')
        self.reset()

    def update_discounting(self, action_index, reward, observation):
        xk = self.action_vectors[:, action_index].reshape(-1, 1)
        Y = observation
        if self.xks is None:
            self.xks = xk
            self.Ys = np.array([Y])
            self.weights = np.array([1])
        else:
            self.xks = np.hstack((self.xks, xk))
            self.Ys = np.append(self.Ys, Y)
            self.weights = np.append(self.weights*self.gamma, 1)
        self.cov_0 = self.cov_0/self.gamma
        self.t += 1
        new_mean, new_cov = self.compute_mean_and_cov()
        self.mean = new_mean
        self.cov = new_cov
        return new_mean, new_cov

    def update_sliding_window(self, action_index, reward, observation):
        xk = self.action_vectors[:, action_index].reshape(-1, 1)
        Y = observation
        if self.xks is None:
            self.xks = xk
            self.Ys = np.array([Y])
            self.weights = np.ones_like(self.Ys)
        else:
            self.xks = np.hstack((self.xks, xk))[:, -self.tau:]
            self.Ys = np.append(self.Ys, Y)[-self.tau:]
            self.weights = np.ones_like(self.Ys)
        self.t += 1
        new_mean, new_cov = self.compute_mean_and_cov()
        self.mean = new_mean
        self.cov = new_cov
        return new_mean, new_cov


def plot_logistic(mean, x_true, y_true, title, foldername, filename, show_plot=True):
    """Plot logistic function for given mean."""
    x_max = np.max(x_true)
    x_min = np.min(x_true)
    x = np.linspace(x_min-0.2, x_max+0.2, 100)
    y = sigmoid(mean[0] + mean[1]*x)
    plt.plot(x, y, label='Fitted model')
    plt.scatter(x_true, y_true, label='True values')
    plt.xlabel('Margin')
    plt.ylabel('Probability')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(foldername, filename))
    if show_plot:
        plt.show()
    plt.close()


def plot_classic(mean, x_true, y_true, title, foldername, filename, show_plot=True):
    """Plot the learned rewards"""
    plt.scatter(x_true, mean, label='Fitted model')
    plt.scatter(x_true, y_true, label='True values')
    plt.xlabel('Margin')
    plt.ylabel('Reward')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(foldername, filename))
    if show_plot:
        plt.show()
    plt.close()
