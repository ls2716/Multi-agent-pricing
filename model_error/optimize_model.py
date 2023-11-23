"""This script contains definition of logistic model and its optimization
to observations.
"""


import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class LogisticModel:
    """Implementation of a logistic model"""

    def __init__(self, action_set, dimension=2) -> None:
        self.action_set = action_set
        self.dimension = dimension
        self.actions = self.generate_vectors()

    def generate_vectors(self):
        """Generate vectors of actions"""
        actions = np.ones((self.dimension, len(self.action_set)))
        for i in range(1, self.dimension):
            actions[i, :] = actions[i-1, :]*self.action_set
        return actions

    def call(self,  theta):
        """Call the model"""
        return expit(np.dot(self.actions.T, theta))


def get_error_function(model, true_values):
    """Calculate the error function"""
    def error_func(theta):
        return np.sum((true_values-model.call(theta))**2)
    return error_func


def optimize_model(error_function, theta_0=np.zeros(2)):
    """Find optimal theta for the model"""
    res = minimize(error_function, x0=theta_0,
                   method='Nelder-Mead', options={'disp': False})
    return res
