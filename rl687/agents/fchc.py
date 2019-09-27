import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable
from tqdm import trange


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """

    def __init__(self, theta: np.ndarray, sigma: float, evaluationFunction: Callable, numEpisodes: int=20):
        self._name = 'First choice hill climbing'
        self.initial_params = [theta, sigma]
        self._theta = theta
        self._sigma = sigma
        self._evaluation_function = evaluationFunction
        self._num_episodes = numEpisodes

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._theta

    def train(self) -> np.ndarray:
        j = self._evaluation_function(self._theta, self._num_episodes)
        lifetime_iterations = 200

        print('Training the {} agent...'.format(self.name))
        bar = trange(lifetime_iterations)
        for i in bar:
            theta_sample = np.random.multivariate_normal(self._theta, (self._sigma ** 2) * np.eye(self._theta.shape[0]))
            new_j = self._evaluation_function(theta_sample, self._num_episodes)
            if new_j > j:
                self._theta = theta_sample
                j = new_j
            bar.set_description("Average return: {}".format(self._evaluation_function(self._theta, self._num_episodes)))
        print('Finished training')

        return self._theta

    def reset(self) -> None:
        self._theta = self.initial_params[0]
        self._sigma = self.initial_params[1]
