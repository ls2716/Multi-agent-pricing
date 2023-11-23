"""This file constains definitions of the environments.
"""
# Imports
import numpy as np

""""""

# Implement Calcano environment


class Calvano(object):
    """This class implements Calvano environment.

    Given n agents with product qualities a_i and prices p_i,
    the demands for ith product is:

    q_i = (exp((a_i-p_i)/mu)))/(\sum_{j=1}^n exp((a_j-p_j)/mu) + exp(a_0/mu)),
    where mu, a_0 are market parameters.   

    Then, the rewards are given by

    r_i = (p_i-c_i)q_i,
    where c_i are costs.

    """

    def __init__(self, A, mu, a_0, C) -> None:
        """Initialisation function.

        Arguments:
            - A (numpy array of floats): list of product quality indexes
            - mu (float): index of horizontal differentitation
            - a_0 (float) inverse index of aggregate demand
            - C (numpy array of floats): list of agent costs per product

        """
        self.A = A
        self.mu = mu
        self.a_0 = a_0
        self.C = C
        self.N = A.shape[0]

    def reset(self) -> None:
        """reset function

        Does nothing."""
        pass

    def step(self, P) -> np.ndarray:
        """Step function.

        Given array of prices returns array of rewards.

        Arguments:
            - P (numpy array of floats): list of agent prices
        """
        # Fix prices shape
        P = P.reshape(-1, self.N)
        # Compute demands
        demands = np.exp((self.A-P)/self.mu) / \
            (np.sum(np.exp((self.A-P)/self.mu), axis=1))
        # Return rewards
        return demands*(P-self.C)


# Implement Insurance market
class InsuranceMarketMt(object):
    """Insurance market implementation

    Each submits a margin and then the price is margin plus the estimated cost.
    The margin does not depend on estimated cost.

    The agent with the smallest price wins the customer.
    If the price is smaller than the reservation price, the customer 
    buys the insurance and receives payoff equal to price minus the true 
    cost. Other agents receive 0 reward.

    The distribution of the true cost does not matter if players margins don't depend
    on estimated costs.

    The distributions of the estimated costs are normal with mean
    m_t, variance sigma and correlation rho.

    The distribution of the reservation price is normal with mean
    m_t + S_c and variance tau.

    """

    def __init__(self, N, sigma, rho, S_c, tau) -> None:
        """Initialize object.

        Arguments:
            - N (int): number of agents
            - sigma (float): standard deviation of estimated costs
            - rho (float): correlation of estimated costs
            - S_c (float): reservation price margin
            - tau (float): variance of the reservation price

        """
        self.N = N
        self.sigma = sigma
        self.rho = rho
        self.S_c = S_c
        self.tau = tau

        self.cov = np.ones(shape=(N, N))*sigma*sigma*rho
        for i in range(N):
            self.cov[i, i] = sigma*sigma

    def reset(self):
        """Reset environment function.

        Does nothing.
        """
        pass

    def step(self, S):
        """Step function.

        The environment receives the margins and returns the rewards.

        Arguments:
            - S (numpy array of floats): list of margins from players

        Returns:
            - rewards
        """
        # Fix S shape
        S = S.reshape(-1, self.N)
        # Get batch size
        batch_size = S.shape[0]
        # Sample costs
        costs = np.random.multivariate_normal(
            np.zeros(shape=self.N), self.cov, size=batch_size).reshape(-1, self.N)
        # Get prices
        prices = costs + S

        # Sample reservation prices
        reservation_prices = np.random.normal(
            self.S_c, self.tau, size=batch_size).reshape(-1, 1)
        # Compute the minimal prices
        min_prices = np.min(prices, axis=1).reshape(-1, 1)
        # Check that the prices are bigger than the reservation price
        min_prices = np.where(min_prices <= reservation_prices, min_prices, 0)
        # Compute the rewards
        rewards = np.where(prices <= min_prices, min_prices, 0)
        observations = np.where(prices <= min_prices, 1, 0)

        return rewards, observations


# Implement Insurance market
class InsuranceMarketCt(object):
    """Insurance market implementation

    Each submits a margin and then the price is margin plus the estimated cost.
    The margin does not depend on estimated cost.

    The agent with the smallest price wins the customer.
    If the price is smaller than the reservation price, the customer 
    buys the insurance and receives payoff equal to price minus the estimated 
    cost. Other agents receive 0 reward.

    The distribution of the true cost does not matter if players margins don't depend
    on estimated costs.

    The distributions of the estimated costs are normal with mean
    m_t, variance sigma and correlation rho.

    The distribution of the reservation price is normal with mean
    m_t + S_c and variance tau.

    """

    def __init__(self, N, sigma, rho, S_c, tau, **kwargs) -> None:
        """Initialize object.

        Arguments:
            - N (int): number of agents
            - sigma (float): standard deviation of estimated costs
            - rho (float): correlation of estimated costs
            - S_c (float): reservation price margin
            - tau (float): variance of the reservation price

        """
        self.N = N
        self.sigma = sigma
        self.rho = rho
        self.S_c = S_c
        self.tau = tau

        self.cov = np.ones(shape=(N, N))*sigma*sigma*rho
        for i in range(N):
            self.cov[i, i] = sigma*sigma

    def reset(self):
        """Reset environment function.

        Does nothing.
        """
        pass

    def update(self):
        """Update the environment.

        Does nothing.
        """
        pass

    def step(self, S):
        """Step function.

        The environment receives the margins and returns the rewards.

        Arguments:
            - S (numpy array of floats): list of margins from players

        Returns:
            - rewards
        """
        # Fix S shape
        S = S.reshape(-1, self.N)
        # Get batch size
        batch_size = S.shape[0]
        # Sample costs
        costs = np.random.multivariate_normal(
            np.zeros(shape=self.N), self.cov, size=batch_size).reshape(-1, self.N)
        # Get prices
        prices = costs + S

        rewards = S

        # Sample reservation prices
        reservation_prices = np.random.normal(
            self.S_c, self.tau, size=batch_size).reshape(-1, 1)
        # Compute the minimal prices
        min_prices = np.min(prices, axis=1).reshape(-1, 1)
        # Check that the prices are bigger than the reservation price
        rewards_check_1 = np.where(
            min_prices <= reservation_prices, rewards, 0)
        # Compute the rewards
        rewards_check_2 = np.where(prices <= min_prices, rewards_check_1, 0)

        observations = np.where(prices <= min_prices, 1, 0)

        return rewards_check_2, observations


class InsuranceMarketCtNs(InsuranceMarketCt):
    """Nonstationary extension of the insurance market.

    The reservation margin S_c is fluctuating according to a sinusoid.
    """

    def __init__(self, N, sigma, rho, tau, S_c_mean, S_c_half_amplitude, S_c_period) -> None:
        super().__init__(N, sigma, rho, S_c_mean, tau)
        self.t = 0
        self.S_c_mean = S_c_mean
        self.S_c_half_amplitude = S_c_half_amplitude
        self.S_c_period = S_c_period

    def reset(self):
        """Reset the environment."""
        self.t = 0
        self.S_c = self.S_c_mean

    def update(self):
        """Update the environment.

        Increment the current time step and calculate the new reservation margin.
        """
        self.t += 1
        self.S_c = self.S_c_mean + self.S_c_half_amplitude * \
            np.sin(self.t/self.S_c_period*2*np.pi)
